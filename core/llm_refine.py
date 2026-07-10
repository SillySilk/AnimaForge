import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from utils.proc import apply_no_window


class LLMRefineProcess(QObject):
    """Runs scripts/llm_refine_run.py to refine/produce captions via LM Studio.

    Mirrors core.tagger.TaggerProcess / core.joycaption.JoyCaptionProcess so the
    Dataset tab can drive it the same way.
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
        url: str,
        model: str = "",
        focus: str = "",
        lora_type: str = "",
        ext: str = ".nl",
        max_tokens: int = 1200,
        characters_file: str = "",
        skip_existing: bool = False,
    ):
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            return

        # Runs in the unified .venv (same interpreter as the GUI).
        from core.env import subprocess_python
        python = subprocess_python(sdscripts_path)

        script = str(Path(__file__).resolve().parents[1] / "scripts" / "llm_refine_run.py")
        args = [script, image_folder, f"--url={url}", f"--ext={ext}",
                f"--max_tokens={int(max_tokens)}"]
        if model.strip():
            args.append(f"--model={model.strip()}")
        if focus.strip():
            args.append(f"--focus={focus.strip()}")
        if lora_type.strip():
            args.append(f"--lora_type={lora_type.strip()}")
        if characters_file.strip():
            args.append(f"--characters_file={characters_file.strip()}")
        if skip_existing:
            args.append("--skip-existing")

        self._process = QProcess(self)
        apply_no_window(self._process)  # no console window pop-up on Windows
        self._process.setProcessChannelMode(QProcess.MergedChannels)

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        self._process.setProcessEnvironment(env)

        self._process.readyReadStandardOutput.connect(self._on_read)
        self._process.finished.connect(self._on_finished)

        self._process.start(python, args)
        self.log_line.emit(f"[LLM] {python} {' '.join(args)}")

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
        self.log_line.emit(f"[LLM] Refinement {msg}.")
        self.finished.emit(success)
