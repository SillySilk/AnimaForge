import re
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

from utils.proc import apply_no_window


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

        # Force UTF-8 and suppress known harmless warnings from the subprocess
        from PySide6.QtCore import QProcessEnvironment
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONWARNINGS", "ignore::UserWarning")
        env.insert("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        self._process.setProcessEnvironment(env)

        self._current_step = 0
        self._process.start(program, args)
        self.training_started.emit()
        self.log_line.emit(f"[AnimaForge] Launching: {program} {' '.join(args)}")

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

        Patterns handled:
          100%|████| 3000/3000 [45:23<00:00,  1.10it/s, loss=0.0821]
          steps:  33%|███       | 100/3000 [01:32<24:17]
          steps: 100/3000
        """
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
