"""Batch training queue: a serializable run definition and a sequential runner.

The runner reuses core.trainer.TrainingProcess so the GPU only ever runs one job at a
time. Each run snapshots everything needed to generate its config and launch training.
"""
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from core import models
from core.caption_policy import ASK
from core.caption_runner import CaptionRunner
from core.config_generator import generate_configs
from core.dataset_manager import combine_all
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
    # Unset (-1) resolves to the model's own default in __post_init__: Anima 500,
    # Krea 2 0 (= Auto). A queue entry lacking the key must not inherit Anima's
    # number on a Krea run. An unknown model keeps -1; that run fails at start.
    target_steps: int = -1
    resolution: int = 1024            # base training resolution (square)
    enable_bucket: bool = True
    save_state: bool = True
    save_every_n_steps: int = 250
    save_every_n_epochs: int = 0      # LoRA epoch-save cadence; 0 = auto (epochs // 5)
    save_last_n_epochs: int = 3       # epoch saves to keep; 0 = keep all
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
    tagger_model_id: str = ""
    tagger_threshold: float = 0.35
    tagger_use_onnx: bool = True
    describe_model_id: str = ""
    describe_cache_dir: str = ""
    style_anchor: str = ""
    caption_policy: str = ASK
    caption_rules: list = field(default_factory=list)
    # core.models.CAPTION_STYLES. Unset ("") resolves to the model's default in
    # __post_init__, so queue JSON without the key captions as it always did.
    caption_style: str = ""
    # Environment
    sdscripts_path: str = ""
    dit_path: str = ""
    qwen3_path: str = ""              # text encoder path (Anima Qwen3-0.6B / MOSS Qwen3.5-9B)
    vae_path: str = ""
    output_dir: str = ""
    onetrainer_root: str = ""
    # Model preset id (core/models.py). Queue JSON written before v3 has no key and
    # is Anima, which is what this default restores.
    model: str = "anima"
    # Advanced settings: {dotted.path: value} over the engine's generated config.
    config_overrides: dict = field(default_factory=dict)
    # Krea 2: write OneTrainer profiler traces (Setup switch, snapshotted per run).
    debug_profiles: bool = False
    # Caption-style comparison this run belongs to (core/compare.py); "" = none.
    compare_group: str = ""
    # Set when run safety shipped an earlier epoch of this (Anima) run; 0 = the end.
    shipped_epoch: int = 0
    # A per-character run (core/character_runs.py) for this token; "" = a normal run.
    # With a source set, its folder is rebuilt from that set's solo images of the token
    # when the run starts; without one it trains on its own set like any run.
    character_source: str = ""
    # Sub-folders of dataset_folder added after the first run (core/added_sets.py),
    # trained with extra weight. [] = the folder as it was.
    added_folders: list = field(default_factory=list)
    character_token: str = ""
    # Runtime
    status: str = QUEUED

    def __post_init__(self):
        preset = models.MODELS.get(self.model)
        if preset is not None:
            if self.target_steps == -1:
                self.target_steps = preset.default_target_steps
            if not self.caption_style:
                self.caption_style = preset.caption_style

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
        from core.caption_runner import chain_for_style
        style = self.caption_style or "prose+tags"
        # A caption-style comparison writes both sidecars whatever its own style, so
        # the other run of the pair can build its .txt from the same prose and tags.
        chain = chain_for_style("prose+tags" if self.compare_group else style)
        return CaptionJob(
            dataset_folder=self.dataset_folder, sdscripts_path=sdscripts_path,
            trigger=self.trigger_word, prefix=self.quality_prefix,
            order=self.caption_order, chain=chain, policy=policy,
            characters_file=characters_file,
            tagger_model_id=self.tagger_model_id, tagger_threshold=self.tagger_threshold,
            tagger_use_onnx=self.tagger_use_onnx,
            describe_model_id=self.describe_model_id,
            describe_cache_dir=self.describe_cache_dir,
            caption_rules=self.caption_rules,
            caption_style=style,
            name_subject=(self.subject_type or "character") == "character")


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


def _default_onetrainer_session(parent):
    from core.engines.onetrainer.session import TrainingSession
    return TrainingSession(parent)


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

    def __init__(self, parent=None, onetrainer_factory=None):
        super().__init__(parent)
        self._onetrainer_factory = onetrainer_factory or _default_onetrainer_session
        self._session = None          # the live OneTrainer session, if any
        self._session_starting = False
        self._session_refusal = ""
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
        # A stopped OneTrainer run keeps training until its process exits; starting
        # now would put a second trainer on the same GPU.
        if self._session is not None and self._session.is_running():
            self.log_line.emit("[Batch] the previous run is still shutting down; "
                               "start again in a moment.")
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
        if self._session is not None and self._session.is_running():
            self._session.cancel()
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
        if run.character_source:
            return self._begin_character_run(run)
        # ASK never reaches a runner: it is a UI-only prompt. Unattended, KEEP is the
        # safe resolution — existing captions are never destroyed.
        policy = run.caption_policy if run.caption_policy != cp.ASK else cp.KEEP
        try:
            state = cp.scan(run.dataset_folder)
            self._rebuild_if_style_changed(run, state, policy)
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

    def _begin_character_run(self, run) -> bool:
        """Copy the character's solo images and their finished captions out of the
        main set, then train. Never captions: the main set's captions are the ones
        this person was named in."""
        from core import character_runs
        try:
            run.image_count = character_runs.build_folder(run)
        except (ValueError, OSError) as e:
            self._mark_failed(f"'{run.lora_name}': {e}")
            return False
        self.log_line.emit(f"[Batch] '{run.lora_name}': {run.image_count} solo image(s) "
                           f"of {run.character_token}, {run.target_steps} steps")
        return self._start_training(run)

    def _rebuild_if_style_changed(self, run, state, policy):
        """The folder was captioned under another caption style. KEEP leaves those
        captions alone, so the run would train on the old style (tags leaking into a
        prose run). Rebuild the .txt of every image this app captioned from its
        sidecars under the run's style; hand-written captions are not ours and stay."""
        from core import caption_manifest as cm, caption_policy as cp
        from core.caption_runner import sources_for_style
        if policy != cp.KEEP or not state.captioned:
            return
        style = run.caption_style or "prose+tags"
        manifest = cm.load(run.dataset_folder)
        recorded = manifest.get("caption_style")
        if not recorded or recorded == style:
            return
        ours = set(manifest.get("images", {}))
        only = [p for p in state.captioned if Path(p).name in ours]
        job = run.to_caption_job(run.sdscripts_path, "", policy)
        written, errors = combine_all(
            run.dataset_folder, prefix=job.combine_prefix(), order=run.caption_order,
            only=only, rules=run.caption_rules, sources=sources_for_style(style))
        manifest = cm.load(run.dataset_folder)
        manifest["caption_style"] = style
        cm.save(run.dataset_folder, manifest)
        self.log_line.emit(
            f"[Batch] '{run.lora_name}': captions were built as {recorded}; rebuilt "
            f"{written} as {style} ({len(only) - written} had nothing to rebuild from).")

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
            engine = models.get(run.model).engine
        except KeyError as e:
            self._mark_failed(f"'{run.lora_name}': {e}")
            return False
        if engine == models.ONETRAINER:
            return self._start_onetrainer(run)
        try:
            from core import added_sets
            weight = added_sets.emphasis() if run.added_folders else 1.0
            params = calculate_training_params(
                added_sets.weighted_count(run.image_count, run.dataset_folder,
                                          run.added_folders, weight),
                target_steps=run.target_steps,
                                              model=run.model)
            self._trainer.set_total_steps(params["total_steps"])
            app = AppSettings()
            run_dir = run_output_dir(run.output_dir, run.lora_name)  # per-run folder
            from core.kohya_recipes import recipe
            rec = recipe(run.model)
            memory_flags = None
            auto_memory = models.get(run.model).auto_memory
            if auto_memory:
                # MOSS: files must exist (a queued run's files can be deleted later),
                # and the memory tier comes from the real GPU, never from Low-VRAM mode.
                from core import moss_memory, moss_run
                problem = moss_run.files_problem(run.dit_path, run.qwen3_path, run.vae_path)
                if problem:
                    self._mark_failed(f"'{run.lora_name}': {problem}")
                    return False
                try:
                    memory_flags, warns = moss_run.launch_plan()
                except moss_memory.MossVramError as e:
                    self._mark_failed(f"'{run.lora_name}': {e}")
                    return False
                for w in warns:
                    self.log_line.emit(f"[Batch] {w}")
            extra = app.build_extra_training_args()
            extra.update(app.prepare_sample_args(
                run_dir, run.lora_name, run.trigger_word,
                prompts=resolve_sample_prompts(run),
                force=bool(run.compare_group),   # a comparison is its previews
                sample_flags=rec.sample_flags,
                sample_every_n_epochs=rec.sample_every_n_epochs))
            from core import lowvram
            # Models that pick their own memory settings ignore Low-VRAM mode.
            lv = (lowvram.get_current() or {}) if not auto_memory else {}
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
                added_subsets=[(path, added_sets.kohya_repeats(params["repeats"], weight))
                               for path, _n in added_sets.folders(run.dataset_folder,
                                                                  run.added_folders)],
                epochs=params["epochs"],
                repeats=params["repeats"],
                optimizer=run.optimizer,
                learning_rate=run.learning_rate,
                network_dim=run.network_dim,
                network_alpha=run.network_alpha,
                train_text_encoder=run.train_text_encoder,
                resolution=run.resolution,
                enable_bucket=enable_bucket,
                save_state=run.save_state,
                save_every_n_steps=run.save_every_n_steps,
                save_every_n_epochs=run.save_every_n_epochs,
                save_last_n_epochs=run.save_last_n_epochs,
                network_weights=(run.network_weights or None),
                training_comment=(run.trigger_word if run.embed_metadata and run.trigger_word else None),
                extra_args=extra,
                overrides=run.config_overrides,
                model=run.model,
                memory_flags=memory_flags,
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

        self._trainer.set_safety(run_dir, run.lora_name, run.save_last_n_epochs)
        self._trainer.start(cfg, run.sdscripts_path, model=run.model)
        return True

    def _start_onetrainer(self, run) -> bool:
        """Hand a OneTrainer-engine run to a fresh session. Same contract as
        _start_training: True once async work is in flight; False after a
        synchronous refusal (already _mark_failed), never calling _advance()."""
        self._session_refusal = ""
        # The session's preflight fails synchronously by emitting `failed` from
        # inside start(); that must not re-enter the queue on this stack.
        self._session_starting = True
        session = None
        try:
            from core.engines.onetrainer.adapter import to_onetrainer_params
            session = self._onetrainer_factory(self)
            # Each slot carries its own session: a session's signals can arrive
            # after the queue has moved on (cancel is asynchronous), and must then
            # be ignored rather than land on whichever run is current.
            session.progress.connect(lambda d, s=session: self._on_session_progress(s, d))
            session.done.connect(lambda path, s=session: self._on_session_end(
                s, True, f"finished: {path}", path))
            session.failed.connect(lambda msg, s=session: self._on_session_end(s, False, msg))
            session.canceled.connect(lambda s=session: self._on_session_end(s, False, "canceled"))
            self._session = session
            started = session.start(to_onetrainer_params(run))
        except Exception as e:  # import, construction or mapping refused the run
            started, self._session_refusal = False, str(e)
        finally:
            self._session_starting = False
        if not started:
            self._release_session(session)
            self._mark_failed(f"'{run.lora_name}' did not start: "
                              f"{self._session_refusal or 'OneTrainer refused the run'}")
            return False
        return True

    def _release_session(self, session):
        if session is not None and session is self._session:
            self._session = None
        if session is not None and hasattr(session, "deleteLater"):
            session.deleteLater()

    def _on_session_progress(self, session, d: dict):
        if session is not self._session:
            return
        self.progress_updated.emit(self._idx, int(d.get("global_step") or 0))

    def _on_session_end(self, session, ok: bool, message: str, shipped: str = ""):
        if self._session_starting:
            self._session_refusal = message
            return
        if session is not self._session:
            return  # a stale report from a session the queue has already left
        self._release_session(session)
        if not self._running:
            return
        self.log_line.emit(f"[Batch] '{self._runs[self._idx].lora_name}': {message}")
        self._on_run_finished(ok, shipped)

    def _on_run_finished(self, success: bool, shipped: str = ""):
        if not self._running:
            return
        run = self._runs[self._idx]
        run.status = DONE if success else FAILED
        if success:
            note = getattr(self._trainer, "last_safety_note", "") if not shipped else ""
            if note:
                self.log_line.emit(f"[Batch] '{run.lora_name}': {note}")
                run.shipped_epoch = int(getattr(self._trainer, "last_shipped_epoch", 0) or 0)
            self._maybe_deliver(run, shipped)
        self.run_finished.emit(self._idx, success)
        if not success:
            self._advance_or_stop()
            return
        self._advance()

    def _on_progress(self, step: int):
        self.progress_updated.emit(self._idx, step)

    def _maybe_deliver(self, run, shipped: str = ""):
        """Auto-deliver the finished LoRA to Forge if enabled in Settings (best-effort).

        `shipped` is the file the engine reported finishing (OneTrainer runs); the
        Kohya path has no such report and rebuilds the conventional location."""
        try:
            app = AppSettings()
            if not app.get("forge_auto_deliver"):
                return
            ran = models.get(run.model)
            if not ran.forge_test:
                # Same rule as the Train tab: the Forge LoRA folder is set up for Anima.
                self.log_line.emit(
                    f"[Batch] {ran.name} LoRA '{run.lora_name}' not sent to Forge automatically "
                    "(auto-deliver is set up for Anima). Deliver it from Home when you want "
                    "it in Forge.")
                return
            lora_dir = app.get("forge_lora_dir")
            src = Path(shipped) if shipped else (
                Path(run_output_dir(run.output_dir, run.lora_name)) / f"{run.lora_name}.safetensors")
            if lora_dir and not src.is_file():
                self.log_line.emit(f"[Batch] deliver skipped: {src} not found")
                return
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
