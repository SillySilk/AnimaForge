"""Batch training queue: a serializable run definition and a sequential runner.

The runner reuses core.trainer.TrainingProcess so the GPU only ever runs one job at a
time. Each run snapshots everything needed to generate its config and launch training.
"""
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from core.caption_policy import ASK
from core.caption_runner import CaptionRunner
from core.config_generator import generate_configs
from core.paths import run_output_dir
from core.settings import AppSettings
from core.step_calculator import calculate_training_params
from core.trainer import TrainingProcess

# Run status values
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

# Per-run phase values (emitted via BatchRunner.run_phase)
CAPTIONING = "captioning"
TRAINING = "training"


def next_index(runs, start: int, skip_done: bool) -> int:
    """The index of the next run to execute at or after `start`.

    Walks forward over already-`DONE` runs when `skip_done` is set; returns
    `len(runs)` when the queue is exhausted (nothing left to run).
    """
    i = start
    while i < len(runs) and skip_done and runs[i].status == DONE:
        i += 1
    return i


@dataclass
class RunDefinition:
    """A full snapshot of one training run, enough to generate its config and launch it."""
    lora_name: str
    dataset_folder: str
    image_count: int
    trigger_word: str = ""
    optimizer: str = "prodigy"
    learning_rate: float = 1e-4
    network_dim: int = 16
    network_alpha: int = 8
    train_text_encoder: bool = False
    target_steps: int = 500
    enable_bucket: bool = True
    save_state: bool = True
    save_every_n_steps: int = 250
    network_weights: str = ""
    embed_metadata: bool = True
    # Full-restore extras (so a loaded set repopulates the Train tab exactly)
    sample_enabled: bool = False
    sample_prompts: list = field(default_factory=list)
    subject_type: str = ""
    # Captioning (so a queued run captions with the settings it was queued with, not
    # whatever happens to sit in the live UI when the batch reaches it — see to_caption_job)
    quality_prefix: str = ""
    caption_order: str = "nl_first"
    refine_enabled: bool = False
    lms_url: str = ""
    lms_model: str = ""
    lms_focus: str = ""
    lora_type: str = ""
    max_tokens: int = 1200
    tagger_model_id: str = ""
    tagger_threshold: float = 0.35
    tagger_use_onnx: bool = True
    style_anchor: str = ""
    caption_policy: str = ASK
    # Environment
    sdscripts_path: str = ""
    dit_path: str = ""
    qwen3_path: str = ""
    vae_path: str = ""
    output_dir: str = ""
    # Runtime
    status: str = QUEUED

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunDefinition":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_caption_job(self, sdscripts_path: str, characters_file: str, policy: str):
        """The CaptionJob this queued run should caption with — built from the run's own
        snapshot, never from whatever sits in the live UI when the batch reaches it."""
        from core.caption_runner import CaptionJob
        chain = ["tag", "describe"]
        if self.refine_enabled:
            chain.append("refine")
        chain.append("combine")
        return CaptionJob(
            dataset_folder=self.dataset_folder, sdscripts_path=sdscripts_path,
            trigger=self.trigger_word, prefix=self.quality_prefix,
            order=self.caption_order, chain=chain, policy=policy,
            lms_url=self.lms_url, lms_model=self.lms_model, lms_focus=self.lms_focus,
            lora_type=self.lora_type, max_tokens=self.max_tokens,
            characters_file=characters_file,
            tagger_model_id=self.tagger_model_id, tagger_threshold=self.tagger_threshold,
            tagger_use_onnx=self.tagger_use_onnx)


def resolve_sample_prompts(run: RunDefinition):
    """The sample prompts a queued run should actually render with.

    Prefer the run's own snapshot; when it's empty (e.g. the set was queued before the
    dataset was captioned), draw SAMPLE_COUNT random verbatim captions from the run's
    dataset at execution time. Returns a list (possibly empty for a captionless dataset).
    """
    prompts = [p.strip() for p in (run.sample_prompts or []) if p and p.strip()]
    if prompts:
        return prompts
    from core.sample_prompts import grab_caption_blocks
    from core.settings import SAMPLE_COUNT
    return grab_caption_blocks(run.dataset_folder, SAMPLE_COUNT)


def save_queue(path: str, runs) -> None:
    Path(path).write_text(
        json.dumps([r.to_dict() for r in runs], indent=2), encoding="utf-8"
    )


def load_queue(path: str):
    p = Path(path)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [RunDefinition.from_dict(d) for d in data]
    except (OSError, ValueError):
        return []


class BatchRunner(QObject):
    """Runs a list of RunDefinition sequentially, one at a time, on the single GPU."""

    run_started = Signal(int)          # index
    run_finished = Signal(int, bool)   # index, success
    run_phase = Signal(int, str)       # index, phase in {"captioning", "training"}
    batch_finished = Signal()
    log_line = Signal(str)
    progress_updated = Signal(int, int)  # index, step

    def __init__(self, parent=None):
        super().__init__(parent)
        self._runs = []
        self._idx = -1
        self._running = False
        self._continue_on_error = True
        self._skip_done = True
        self._trainer = TrainingProcess(self)
        self._trainer.log_line.connect(self.log_line)
        self._trainer.training_finished.connect(self._on_run_finished)
        self._trainer.progress_updated.connect(self._on_progress)
        # Captioner is wired ONCE here — a per-run connect/disconnect dance is how
        # double-fired slots get born. Each queued run captions itself just before
        # its own training (see _begin_run).
        self._captioner = CaptionRunner(self)
        self._captioner.log_line.connect(self.log_line)
        self._captioner.finished.connect(self._on_caption_finished)

    def is_running(self) -> bool:
        return self._running

    def start(self, runs, continue_on_error: bool = True, skip_done: bool = True):
        if self._running or not runs:
            return
        self._runs = runs
        self._continue_on_error = continue_on_error
        self._skip_done = skip_done
        self._running = True
        self._idx = -1
        self._advance()

    def reset_statuses(self, runs) -> None:
        for r in runs:
            r.status = QUEUED

    def restart(self, runs, continue_on_error: bool = True):
        """Run the whole queue again from the top — the only way back once every
        run is DONE. Resets every status to QUEUED and starts with skip_done off."""
        self.reset_statuses(runs)
        self.start(runs, continue_on_error=continue_on_error, skip_done=False)

    def stop(self):
        # Tear down _running BEFORE stopping the captioner: CaptionRunner.stop()
        # emits finished(False) synchronously, which re-enters _on_caption_finished
        # — it must find _running already False and NOT advance the queue.
        self._running = False
        if self._captioner.is_running():
            self._captioner.stop()
        if self._trainer.is_running():
            self._trainer.stop()
        self.batch_finished.emit()

    # ------------------------------------------------------------------

    def _advance(self):
        """Move through the queue as an explicit loop, never a recursive call.

        A synchronous run failure (e.g. every run refusing to caption because of
        one shared bad sdscripts_path) must not re-enter this method on the same
        call stack — with a few hundred queued runs that blew the stack with a
        RecursionError before continue_on_error ever got a chance to matter.
        `_begin_run` either hands off to async work and returns True (in which
        case we return and wait for the callback to bring us back here on a
        fresh stack), or fails the run synchronously and returns False, in which
        case we just loop to the next run.
        """
        while True:
            if not self._running:
                return
            self._idx = next_index(self._runs, self._idx + 1, self._skip_done)
            if self._idx >= len(self._runs):
                self._running = False
                self.batch_finished.emit()
                return

            run = self._runs[self._idx]
            run.status = RUNNING
            self.run_started.emit(self._idx)
            self.log_line.emit(
                f"[Batch] ({self._idx + 1}/{len(self._runs)}) starting '{run.lora_name}'")
            if self._begin_run(run):
                return                      # async work in flight; the callback re-enters
            if not self._continue_on_error:
                self._running = False
                self.batch_finished.emit()
                return
            # synchronous failure — loop straight to the next run, no recursion

    def _begin_run(self, run) -> bool:
        """Caption this run just before its own training, so a caption failure kills
        only this one run and the sample prompts (resolved later in _start_training)
        are drawn from captions that exist by then.

        Returns True once the run has been handed off to asynchronous work
        (captioning started, or training started). Returns False if it failed
        synchronously — in which case _mark_failed has already been called and
        the caller (the loop in _advance) decides whether to continue or stop.
        Must never call _advance() itself.
        """
        from core import caption_policy as cp, characters as ch
        # ASK never reaches a runner: it is a UI-only prompt. Unattended, KEEP is the
        # safe resolution — existing captions are never destroyed.
        policy = run.caption_policy if run.caption_policy != cp.ASK else cp.KEEP
        try:
            state = cp.scan(run.dataset_folder)
            if not cp.images_for(state, policy):
                return self._start_training(run)
            self.run_phase.emit(self._idx, CAPTIONING)
            job = run.to_caption_job(
                run.sdscripts_path, ch.path_for(run.dataset_folder), policy)
            started = self._captioner.start(job)
        except (ValueError, OSError) as e:
            # ValueError: an unknown stage in the caption chain. OSError: the
            # dataset folder vanished mid-batch (deleted, or a network share
            # dropped) while cp.scan() was walking it. Either way, fail this
            # run and spare the queue.
            self._mark_failed(f"captioning could not start for '{run.lora_name}': {e}")
            return False
        if not started:
            self._mark_failed(f"could not start captioning for '{run.lora_name}'")
            return False
        return True

    def _on_caption_finished(self, ok: bool):
        if not self._running:
            return
        if not ok:
            self._mark_failed(
                f"captioning failed for '{self._runs[self._idx].lora_name}'")
            self._advance_or_stop()
            return
        if not self._start_training(self._runs[self._idx]):
            self._advance_or_stop()

    def _mark_failed(self, msg: str):
        """Pure bookkeeping: mark the current run FAILED and report it. Does not
        advance the queue — callers decide whether to continue or stop."""
        self._runs[self._idx].status = FAILED
        self.log_line.emit(f"[Batch] {msg}")
        self.run_finished.emit(self._idx, False)

    def _advance_or_stop(self):
        """After an asynchronous failure: stop the batch if continue_on_error is
        False, otherwise move on to the next run. Safe to call self._advance()
        here — this runs from a real callback (a fresh, shallow stack), not from
        within _advance()'s own loop."""
        if not self._continue_on_error:
            self._running = False
            self.batch_finished.emit()
            return
        self._advance()

    def _start_training(self, run) -> bool:
        """Generate this run's config and launch training.

        Returns True once training has been handed off asynchronously to the
        trainer. Returns False if config generation failed synchronously — in
        that case _mark_failed has already been called; the caller decides
        whether to continue or stop. Must never call _advance() itself.
        """
        self.run_phase.emit(self._idx, TRAINING)
        try:
            params = calculate_training_params(run.image_count, target_steps=run.target_steps)
            self._trainer.set_total_steps(params["total_steps"])
            app = AppSettings()
            run_dir = run_output_dir(run.output_dir, run.lora_name)  # per-run folder
            extra = app.build_extra_training_args()
            extra.update(app.prepare_sample_args(
                run_dir, run.lora_name, run.trigger_word,
                prompts=resolve_sample_prompts(run)))
            from core import lowvram
            lv = lowvram.get_current() or {}  # active only if acknowledged this session
            # Aspect-ratio bucketing divides by zero on images with a side under 64 px
            # (sd-scripts floors the bucket to a multiple of 64). Drop bucketing for this
            # run rather than let it die at exit code 1 mid-queue.
            enable_bucket = run.enable_bucket
            if enable_bucket:
                from core.dataset_manager import find_undersized_images
                small = find_undersized_images(run.dataset_folder)
                if small:
                    enable_bucket = False
                    self.log_line.emit(
                        f"[Batch] bucketing disabled for '{run.lora_name}': {len(small)} "
                        "image(s) have a side under 64 px (would crash bucketing) — "
                        "center-cropping instead.")
            cfg, _ = generate_configs(
                output_dir=run_dir,
                micro_batch=lv.get("micro_batch"),
                grad_accum=lv.get("grad_accum"),
                blocks_to_swap=lv.get("blocks_to_swap"),
                fp8_base=bool(lv.get("fp8_base")),
                lora_name=run.lora_name,
                dit_path=run.dit_path,
                qwen3_path=run.qwen3_path,
                vae_path=run.vae_path,
                dataset_folder=run.dataset_folder,
                epochs=params["epochs"],
                repeats=params["repeats"],
                optimizer=run.optimizer,
                learning_rate=run.learning_rate,
                network_dim=run.network_dim,
                network_alpha=run.network_alpha,
                train_text_encoder=run.train_text_encoder,
                enable_bucket=enable_bucket,
                save_state=run.save_state,
                save_every_n_steps=run.save_every_n_steps,
                network_weights=(run.network_weights or None),
                training_comment=(run.trigger_word if run.embed_metadata and run.trigger_word else None),
                extra_args=extra,
            )
        except Exception as e:
            self._mark_failed(f"config generation failed for '{run.lora_name}': {e}")
            return False

        # Unattended: never block the queue, just warn in the log if VRAM looks low.
        try:
            from core import gpu_check
            free = gpu_check.free_vram_mb()
            need = AppSettings().get("min_free_vram_mb")
            if free is not None and need and free < need:
                apps = gpu_check.resident_gpu_apps()
                detail = (" Detected: " + ", ".join(apps) + ".") if apps else ""
                self.log_line.emit(
                    f"[Batch] WARNING: only {free / 1024:.1f} GB VRAM free "
                    f"(need ~{need / 1024:.1f} GB).{detail} May OOM.")
        except Exception:
            pass

        self._trainer.start(cfg, run.sdscripts_path)
        return True

    def _on_run_finished(self, success: bool):
        if not self._running:
            return
        run = self._runs[self._idx]
        run.status = DONE if success else FAILED
        if success:
            self._maybe_deliver(run)
        self.run_finished.emit(self._idx, success)
        if not success:
            self._advance_or_stop()
            return
        self._advance()

    def _on_progress(self, step: int):
        self.progress_updated.emit(self._idx, step)

    def _maybe_deliver(self, run):
        """Auto-deliver the finished LoRA to Forge if enabled in Settings (best-effort)."""
        try:
            app = AppSettings()
            if not app.get("forge_auto_deliver"):
                return
            lora_dir = app.get("forge_lora_dir")
            src = Path(run_output_dir(run.output_dir, run.lora_name)) / f"{run.lora_name}.safetensors"
            if lora_dir and src.is_file():
                from core import forge_api
                from core.paths import delivery_filename
                out = forge_api.deliver_lora(
                    str(src), lora_dir, app.get("forge_api_url"),
                    dest_name=delivery_filename(run.lora_name,
                                                getattr(run, "trigger_word", "")))
                self.log_line.emit(f"[Batch] Delivered '{run.lora_name}' → {out}")
        except Exception as e:
            self.log_line.emit(f"[Batch] deliver failed: {e}")
