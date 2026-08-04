import os
import re
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

from utils.proc import apply_no_window


def training_env_overrides(existing=None, windows=None, force=False) -> dict:
    """Environment the training subprocess needs on top of the inherited one.

    USE_LIBUV is the important one. `accelerate launch` always builds a c10d TCPStore,
    and since torch 2.4 the store asks for the libuv backend unless USE_LIBUV=0
    (torch/distributed/rendezvous.py reads exactly that variable). Plenty of Windows
    PyTorch wheels are compiled without libuv, so the rendezvous raises "use_libuv was
    requested but PyTorch was build without libuv support" and the whole run exits with
    code 1 before a single training step — with nothing on screen but the exit code.
    The legacy store it falls back to is equivalent here: we launch --num_processes=1,
    so the store is a single-rank formality either way.

    An explicit USE_LIBUV in the user's own environment always wins, so anyone with a
    libuv-enabled build can opt back in — unless `force=True` (the Setup → Training
    Environment "ignore my environment" toggle), which makes AnimaForge set USE_LIBUV=0
    unconditionally. That's the self-service alternative to deleting the variable from
    Windows by hand.
    """
    if windows is None:
        windows = sys.platform.startswith("win")
    if existing is None:
        existing = os.environ
    env = {
        # Force UTF-8 and suppress known harmless warnings from the subprocess
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONWARNINGS": "ignore::UserWarning",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    if windows and (force or not existing.get("USE_LIBUV")):
        env["USE_LIBUV"] = "0"
    return env


def libuv_status(existing=None, windows=None, force=False) -> str:
    """One line for the training log: the USE_LIBUV the subprocess really gets, and why.

    Worth logging because the two failure modes are indistinguishable from outside.
    We deliberately never clobber a USE_LIBUV already in the user's environment unless
    `force=True`, so a stray `USE_LIBUV=1` there silently defeats the fix and the run
    still dies at exit code 1 — looking exactly like the fix never shipped.
    """
    if windows is None:
        windows = sys.platform.startswith("win")
    if existing is None:
        existing = os.environ
    if not windows:
        return "USE_LIBUV=1 (torch default; not Windows)"
    inherited = existing.get("USE_LIBUV")
    if inherited and force:
        return (f"USE_LIBUV=0 (forced — your Setup override replaces your "
                 f"environment's USE_LIBUV={inherited})")
    if inherited:
        return (f"USE_LIBUV={inherited} (from your environment — AnimaForge left it "
                f"alone; unset it to let AnimaForge choose)")
    return "USE_LIBUV=0 (set by AnimaForge for this run)"


def failure_hint(line: str):
    """Plain-English explanation for a known-cryptic launch failure, else None.

    torch's wording ("PyTorch was build without libuv support") is both misspelled and
    meaningless to a user whose only other evidence is exit code 1.
    """
    low = (line or "").lower()
    if "libuv" in low and "without libuv" in low:
        return ("[AnimaForge] That means this PyTorch build has no libuv support. "
                "AnimaForge normally sets USE_LIBUV=0 to avoid it — seeing this anyway "
                "means a USE_LIBUV variable in your own Windows environment is "
                "overriding that. Delete it (System Properties → Environment "
                "Variables) and relaunch. Rebuilding PyTorch is not necessary.")
    return None


class TrainingProcess(QObject):
    progress_updated = Signal(int)   # current step (0-based int)
    log_line = Signal(str)           # raw log line string
    training_finished = Signal(bool) # True = success, False = error/killed
    training_started = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None
        self._total_steps = 3000
        self._current_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_total_steps(self, total_steps: int):
        self._total_steps = max(1, total_steps)

    def build_command(self, config_path: str, sdscripts_path: str) -> tuple:
        """
        Build the accelerate launch command for anima_train_network.py.

        Returns (program, args_list).
        Uses the unified .venv interpreter (which has accelerate + the ML stack);
        a legacy sd-scripts/venv is honored only for old two-venv installs.
        """
        from core.env import subprocess_python
        python = subprocess_python(sdscripts_path)

        train_script = str(Path(sdscripts_path) / "anima_train_network.py")
        args = [
            "-m",
            "accelerate.commands.launch",
            "--num_cpu_threads_per_process=2",
            "--num_processes=1",
            "--num_machines=1",
            "--mixed_precision=bf16",
            "--dynamo_backend=no",
            train_script,
            f"--config_file={config_path}",
        ]
        return python, args

    def start(self, config_path: str, sdscripts_path: str):
        """Start the training process."""
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            return

        program, args = self.build_command(config_path, sdscripts_path)

        self._process = QProcess(self)
        apply_no_window(self._process)  # no console window pop-up on Windows
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_ready_read)
        self._process.finished.connect(self._on_finished)

        from PySide6.QtCore import QProcessEnvironment
        env = QProcessEnvironment.systemEnvironment()
        for key, value in training_env_overrides().items():
            env.insert(key, value)
        self._process.setProcessEnvironment(env)

        self._current_step = 0
        self._process.start(program, args)
        self.training_started.emit()
        self.log_line.emit(f"[AnimaForge] Launching: {program} {' '.join(args)}")
        self.log_line.emit(f"[AnimaForge] {libuv_status()}")

    def stop(self):
        """Terminate the running training process."""
        if self._process is not None:
            if self._process.state() != QProcess.NotRunning:
                self._process.terminate()
                if not self._process.waitForFinished(5000):
                    self._process.kill()
            self._process = None

    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.state() != QProcess.NotRunning

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_ready_read(self):
        if self._process is None:
            return
        data = self._process.readAllStandardOutput()
        text = bytes(data).decode("utf-8", errors="replace")
        for line in text.splitlines():
            if line:
                if "impl_abstract" in line:
                    continue
                self.log_line.emit(line)
                hint = failure_hint(line)
                if hint:
                    self.log_line.emit(hint)
                step = self._parse_progress(line)
                if step is not None:
                    self._current_step = step
                    self.progress_updated.emit(step)

    def _on_finished(self, exit_code: int, exit_status):
        success = exit_code == 0
        self.training_finished.emit(success)
        status = "completed successfully" if success else f"finished with exit code {exit_code}"
        self.log_line.emit(f"[AnimaForge] Training process {status}.")

    def _parse_progress(self, line: str):
        """
        Extract the current step count from Kohya output.

        Only the main training bar counts — sd-scripts names it ``desc="steps"``.
        Other tqdm bars in the same stream (latent caching, "Sampling" while a
        preview set renders, weight loading) would otherwise clobber the step,
        the total, and the dials mid-run.

        Patterns handled:
          steps:  33%|███       | 100/3000 [01:32<24:17,  1.10it/s, loss=0.0821]
          steps: 100/3000
        """
        if not line.lstrip().lower().startswith("steps"):
            return None

        # Pattern 1: tqdm bar  "current/total [..."
        m = re.search(r"(\d+)/(\d+)\s*\[", line)
        if m:
            current = int(m.group(1))
            total = int(m.group(2))
            if total > 0:
                self._total_steps = total
            return current

        # Pattern 2: plain "steps: current/total"
        m = re.search(r"steps:\s*(\d+)/(\d+)", line, re.IGNORECASE)
        if m:
            current = int(m.group(1))
            total = int(m.group(2))
            if total > 0:
                self._total_steps = total
            return current

        return None

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def current_step(self) -> int:
        return self._current_step
