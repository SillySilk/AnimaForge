import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from utils.proc import apply_no_window


class TaggerProcess(QObject):
    log_line = Signal(str)
    finished = Signal(bool)   # True = success

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None

    def start(
        self,
        sdscripts_path: str,
        image_folder: str,
        model_id: str,
        threshold: float,
        overwrite: bool,
        use_onnx: bool = False,
        batch_size: int = 8,
    ):
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            return

        from core.env import subprocess_python
        python = subprocess_python(sdscripts_path)

        script = str(Path(sdscripts_path) / "finetune" / "tag_images_by_wd14_tagger.py")

        # model_dir: store downloaded models inside sd-scripts to keep them portable
        model_dir = str(Path(sdscripts_path) / "wd14_models")

        # Check if the model file is already fully downloaded
        repo_dir = model_id.replace("/", "_")
        if use_onnx:
            model_file = Path(model_dir) / repo_dir / "model.onnx"
        else:
            model_file = Path(model_dir) / repo_dir / "saved_model.pb"
        needs_download = not model_file.is_file()

        args = [
            script,
            image_folder,
            f"--repo_id={model_id}",
            f"--model_dir={model_dir}",
            f"--thresh={threshold:.2f}",
            f"--batch_size={batch_size}",
            "--caption_extension=.tags",
            "--remove_underscore",  # Anima wants 'side ponytail' not 'side_ponytail'
        ]
        if use_onnx:
            args.append("--onnx")
        if needs_download:
            args.append("--force_download")
        # This sd-scripts build overwrites caption files by default and has no
        # --overwrite_caption flag; use --append_tags to add to existing tags instead.
        if not overwrite:
            args.append("--append_tags")

        self._process = QProcess(self)
        apply_no_window(self._process)  # no console window pop-up on Windows
        self._process.setProcessChannelMode(QProcess.MergedChannels)

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONWARNINGS",
            "ignore::FutureWarning:xformers,"
            "ignore::FutureWarning:torch,"
            "ignore::UserWarning:xformers,"
            "ignore::UserWarning:transformers"
        )
        self._process.setProcessEnvironment(env)

        self._process.readyReadStandardOutput.connect(self._on_read)
        self._process.finished.connect(self._on_finished)

        self._process.start(python, args)
        self.log_line.emit(f"[Tagger] {python} {' '.join(args)}")

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
        self.log_line.emit(f"[Tagger] Tagging {msg}.")
        self.finished.emit(success)
