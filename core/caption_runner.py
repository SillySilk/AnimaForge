"""The tag -> describe -> refine -> combine caption chain, without a GUI.

Lifted out of ui.dataset_tab so BatchRunner and DatasetTab drive the same code.
It is a QObject, not a QWidget: it needs a Qt event loop (QProcess) but no display,
so `main.py --run-batch` can run it under a bare QCoreApplication.

`plan_stages` is pure and carries the policy decision; the runner just executes.
"""
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from core import caption_manifest, caption_policy as cp
from core.caption_progress import parse_progress
from core.dataset_manager import combine_all
from core.joycaption import JoyCaptionProcess
from core.llm_refine import LLMRefineProcess
from core.tagger import TaggerProcess


VALID_STAGES = ("tag", "describe", "refine", "combine")


def _default_chain():
    return ["tag", "describe", "combine"]


@dataclass
class CaptionJob:
    dataset_folder: str
    sdscripts_path: str
    trigger: str = ""
    prefix: str = ""
    order: str = "nl_first"
    chain: list = field(default_factory=_default_chain)
    policy: str = cp.OVERWRITE
    lms_url: str = ""
    lms_model: str = ""
    lms_focus: str = ""
    lora_type: str = ""
    max_tokens: int = 1200
    characters_file: str = ""
    tagger_model_id: str = ""
    tagger_threshold: float = 0.35
    tagger_use_onnx: bool = True

    def combine_prefix(self) -> str:
        """What `combine_all(prefix=...)` receives. The trigger is NOT a separate
        parameter of combine_all — it rides at the head of the prefix. Mirrors
        ui/dataset_tab.py::_rebuild_txt_from_sidecars."""
        return ", ".join(b for b in [self.trigger.strip(), self.prefix.strip()] if b)


def plan_stages(job: CaptionJob, state) -> list:
    """[(stage, images)] in chain order. Stages with no images are dropped, so a
    fully-captioned folder under KEEP plans nothing at all."""
    for stage in job.chain:
        if stage not in VALID_STAGES:
            raise ValueError(
                f"unknown caption stage {stage!r} in job.chain — must be one of "
                f"{VALID_STAGES}")
    images = cp.images_for(state, job.policy)
    if not images:
        return []
    return [(stage, list(images)) for stage in job.chain]


class CaptionRunner(QObject):
    log_line = Signal(str)
    tick = Signal(str, int, int, str)   # phase, done, total, filename
    stage_done = Signal(str)
    finished = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._job = None
        self._stages = []
        self._running = False
        self._only_file = None
        self._tagger = TaggerProcess(self)
        self._joy = JoyCaptionProcess(self)
        self._llm = LLMRefineProcess(self)
        for proc in (self._tagger, self._joy, self._llm):
            proc.log_line.connect(self._on_log)
        self._tagger.finished.connect(lambda ok: self._step_done("tag", ok))
        self._joy.finished.connect(lambda ok: self._step_done("describe", ok))
        self._llm.finished.connect(lambda ok: self._step_done("refine", ok))

    def is_running(self) -> bool:
        return self._running

    def start(self, job: CaptionJob) -> bool:
        if self._running or not job.sdscripts_path or not job.dataset_folder:
            return False
        state = cp.scan(job.dataset_folder)
        self._stages = plan_stages(job, state)
        self._job = job
        if not self._stages:
            self.log_line.emit("[Caption] every image already captioned — nothing to do.")
            self.finished.emit(True)
            return True
        caption_manifest.record_settings(
            job.dataset_folder, job.trigger, job.prefix, job.order, job.chain)
        self._running = True
        self._next()
        return True

    def stop(self):
        """Abort the chain. Emits finished(False) exactly once, whether or not a
        subprocess is currently alive. Tearing state down BEFORE terminating the
        processes matters: QProcess.finished fires synchronously inside
        waitForFinished(), re-entering _step_done, which must find _running False
        and no-op rather than emit a second finished()."""
        if not self._running:
            return
        self._running = False
        self._stages = []
        for proc in (self._tagger, self._joy, self._llm):
            if proc.is_running():
                proc.stop()
        self._cleanup_only_file()
        self.finished.emit(False)

    # ------------------------------------------------------------------

    def _write_only_file(self, images) -> str:
        fd, path = tempfile.mkstemp(prefix="af_only_", suffix=".txt", text=True)
        with open(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(images))
        self._only_file = path
        return path

    def _cleanup_only_file(self):
        """Remove the temp --only file, if any. Safe to call more than once."""
        if self._only_file:
            Path(self._only_file).unlink(missing_ok=True)
            self._only_file = None

    def _on_log(self, line: str):
        self.log_line.emit(line)
        t = parse_progress(line)
        if t is not None:
            self.tick.emit(t.phase, t.done, t.total, t.filename)

    def _next(self):
        if not self._running:
            return
        if not self._stages:
            self._running = False
            self.finished.emit(True)
            return
        stage, images = self._stages[0]
        job = self._job
        if stage == "tag":
            self._tagger.start(
                sdscripts_path=job.sdscripts_path, image_folder=job.dataset_folder,
                model_id=job.tagger_model_id, threshold=job.tagger_threshold,
                use_onnx=job.tagger_use_onnx,
                only_file=self._write_only_file(images))
        elif stage == "describe":
            self._joy.start(sdscripts_path=job.sdscripts_path,
                            image_folder=job.dataset_folder,
                            overwrite=(job.policy == cp.OVERWRITE))
        elif stage == "refine":
            self._llm.start(
                sdscripts_path=job.sdscripts_path, image_folder=job.dataset_folder,
                url=job.lms_url, model=job.lms_model, focus=job.lms_focus,
                lora_type=job.lora_type, max_tokens=job.max_tokens,
                characters_file=job.characters_file,
                skip_existing=(job.policy == cp.KEEP))
        elif stage == "combine":
            written, errors = combine_all(
                job.dataset_folder, prefix=job.combine_prefix(), order=job.order,
                only=images)
            self.log_line.emit(f"[Combine] {written} caption file(s) built, {errors} error(s).")
            self._step_done("combine", errors == 0)

    def _step_done(self, stage: str, ok: bool):
        if not self._running or not self._stages or self._stages[0][0] != stage:
            return
        _stage, images = self._stages.pop(0)
        self._cleanup_only_file()
        if not ok:
            self._running = False
            self.log_line.emit(f"[Caption] {stage} failed — stopping. Completed steps are kept.")
            self.finished.emit(False)
            return
        caption_manifest.mark_stage(self._job.dataset_folder, stage, images)
        self.stage_done.emit(stage)
        self._next()
