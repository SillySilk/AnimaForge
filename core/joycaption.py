import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from utils.proc import apply_no_window


class JoyCaptionProcess(QObject):
    """QProcess wrapper that runs scripts/joycaption_run.py in the sd-scripts venv.

    Mirrors core.tagger.TaggerProcess so the Dataset tab can drive both the same way.
    """

    log_line = Signal(str)
    finished = Signal(bool)  # True = success

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None

    def start(
        self,
        sdscripts_path: str,
        image_folder: str,
        ext: str = ".nl",
        overwrite: bool = False,
        prompt: str = "",
    ):
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            return

        # JoyCaption needs torch/transformers — present in the unified .venv.
        from core.env import subprocess_python
        python = subprocess_python(sdscripts_path)

        script = str(Path(__file__).resolve().parents[1] / "scripts" / "joycaption_run.py")

        args = [script, image_folder, f"--ext={ext}"]
        if overwrite:
            args.append("--overwrite")
        if prompt.strip():
            args.append(f"--prompt={prompt.strip()}")

        self._process = QProcess(self)
        apply_no_window(self._process)  # no console window pop-up on Windows
        self._process.setProcessChannelMode(QProcess.MergedChannels)

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONWARNINGS", "ignore::UserWarning")
        self._process.setProcessEnvironment(env)

        self._process.readyReadStandardOutput.connect(self._on_read)
        self._process.finished.connect(self._on_finished)

        self._process.start(python, args)
        self.log_line.emit(f"[JoyCaption] {python} {' '.join(args)}")

    def stop(self):
        if self._process and self._process.state() != QProcess.NotRunning:
            self._process.terminate()
            if not self._process.waitForFinished(5000):
                self._process.kill()

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.NotRunning

    def _on_read(self):
        if not self._process:
            return
        data = self._process.readAllStandardOutput()
        text = bytes(data).decode("utf-8", errors="replace")
        for line in text.splitlines():
            if line.strip():
                self.log_line.emit(line)

    def _on_finished(self, exit_code: int, _status):
        success = exit_code == 0
        msg = "completed successfully" if success else f"failed (exit code {exit_code})"
        self.log_line.emit(f"[JoyCaption] Captioning {msg}.")
        self.finished.emit(success)
