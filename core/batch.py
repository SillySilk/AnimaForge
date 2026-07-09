"""Batch training queue: a serializable run definition and a sequential runner.

The runner reuses core.trainer.TrainingProcess so the GPU only ever runs one job at a
time. Each run snapshots everything needed to generate its config and launch training.
"""
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

from PySide6.QtCore import QObject, Signal

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
    batch_finished = Signal()
    log_line = Signal(str)
    progress_updated = Signal(int, int)  # index, step

    def __init__(self, parent=None):
        super().__init__(parent)
        self._runs = []
        self._idx = -1
        self._running = False
        self._continue_on_error = True
        self._trainer = TrainingProcess(self)
        self._trainer.log_line.connect(self.log_line)
        self._trainer.training_finished.connect(self._on_run_finished)
        self._trainer.progress_updated.connect(self._on_progress)

    def is_running(self) -> bool:
        return self._running

    def start(self, runs, continue_on_error: bool = True):
        if self._running or not runs:
            return
        self._runs = runs
        self._continue_on_error = continue_on_error
        self._running = True
        self._idx = -1
        self._advance()

    def stop(self):
        self._running = False
        if self._trainer.is_running():
            self._trainer.stop()
        self.batch_finished.emit()

    # ------------------------------------------------------------------

    def _advance(self):
        if not self._running:
            return
        self._idx += 1
        if self._idx >= len(self._runs):
            self._running = False
            self.batch_finished.emit()
            return

        run = self._runs[self._idx]
        run.status = RUNNING
        self.run_started.emit(self._idx)
        self.log_line.emit(f"[Batch] ({self._idx + 1}/{len(self._runs)}) starting '{run.lora_name}'")

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
            run.status = FAILED
            self.log_line.emit(f"[Batch] config generation failed for '{run.lora_name}': {e}")
            self.run_finished.emit(self._idx, False)
            self._advance()
            return

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

    def _on_run_finished(self, success: bool):
        if not self._running:
            return
        run = self._runs[self._idx]
        run.status = DONE if success else FAILED
        if success:
            self._maybe_deliver(run)
        self.run_finished.emit(self._idx, success)
        if not success and not self._continue_on_error:
            self._running = False
            self.batch_finished.emit()
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
