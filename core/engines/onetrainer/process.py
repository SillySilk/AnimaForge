"""Drives one OneTrainer training subprocess and translates its output.

cwd MUST be the OneTrainer root: train.py resolves its imports and its default
secrets.json against the working directory. All run-specific paths inside the
config are absolute, so artifacts land in the app's run folder regardless.

OneTrainer emits tqdm progress (epoch:/step: bars) on stderr; channels are merged
and the buffer split on both \\r and \\n because tqdm redraws lines in place.
"""
import re
import subprocess
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from core.engines.onetrainer import preset
from utils.proc import apply_no_window

# train.py is launched through this, which caps Prodigy's step size first.
TRAIN_LAUNCHER = str(Path(__file__).resolve().with_name("onetrainer_train.py"))

_PROGRESS_RE = re.compile(
    r"^(?P<name>epoch|step):\s*\d+%\|[^|]*\|\s*(?P<n>\d+)/(?P<total>\d+)")
_LOSS_RE = re.compile(r"\bloss=(?P<loss>[0-9.eE+-]+)")
_OOM_MARKERS = ("CUDA out of memory", "torch.OutOfMemoryError",
                "CUDA error: out of memory")
_CREATE_NO_WINDOW = 0x08000000


def parse_progress_line(line: str):
    """{"name","n","total","loss"} for a tqdm bar line, else None. Pure."""
    m = _PROGRESS_RE.match(line or "")
    if not m:
        return None
    loss_m = _LOSS_RE.search(line)
    return {
        "name": m.group("name"),
        "n": int(m.group("n")),
        "total": int(m.group("total")),
        "loss": float(loss_m.group("loss")) if loss_m else None,
    }


def split_lines(buf: str):
    """(complete_lines, remainder) splitting on \\r and \\n. Pure."""
    parts = re.split(r"[\r\n]", buf)
    return [p for p in parts[:-1] if p.strip()], parts[-1]


class TrainerProcess(QObject):
    line = Signal(str)
    progress = Signal(dict)
    finished = Signal(int, bool)   # (exit_code, saw_oom)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc = None
        self._buf = ""
        self._saw_oom = False

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.NotRunning

    def start(self, install, config_path: str, extra_env=None) -> bool:
        """Spawn train.py on `config_path`, through TRAIN_LAUNCHER so Prodigy's
        step size never exceeds preset.PRODIGY_D_CAP. `extra_env` is laid over the
        inherited environment -- OneTrainer's own switches (OT_DEBUG_PROFILES)
        live there."""
        return self._spawn(install, [TRAIN_LAUNCHER, install.train_script,
                                     repr(preset.PRODIGY_D_CAP),
                                     "--config-path", config_path], extra_env)

    def start_script(self, install, script: str, args, extra_env=None) -> bool:
        """Spawn any script under the install's interpreter, from its root, with
        its output lined and merged exactly like train.py's -- the final-preview
        render (core/engines/onetrainer/render.py) is one."""
        return self._spawn(install, [script, *args], extra_env)

    def _spawn(self, install, argv: list, extra_env=None) -> bool:
        if self.is_running():
            return False
        self._buf, self._saw_oom = "", False
        p = QProcess(self)
        p.setWorkingDirectory(install.root)
        p.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")   # progress lines must arrive live
        for key, value in (extra_env or {}).items():
            env.insert(key, value)
        p.setProcessEnvironment(env)
        apply_no_window(p)
        p.readyReadStandardOutput.connect(self._on_ready)
        p.finished.connect(self._on_finished)
        p.start(install.python, argv)
        if not p.waitForStarted(15000):
            self._proc = None
            return False
        self._proc = p
        return True

    def cancel(self):
        """Terminate the whole process tree — train.py forks workers that a plain
        QProcess.kill() would orphan."""
        if not self.is_running():
            return
        pid = int(self._proc.processId())
        if pid:
            self.set_paused(False)   # a suspended tree can't be torn down cleanly
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           creationflags=_CREATE_NO_WINDOW, capture_output=True)

    def set_paused(self, paused: bool) -> bool:
        """Suspend or resume the whole train.py tree. Returns whether it took."""
        if not self.is_running():
            return False
        from core.proc_pause import set_tree_suspended
        return set_tree_suspended(int(self._proc.processId()), paused)

    def wait(self, msec: int) -> bool:
        """Test-only pump: block until the process exits, then drain the buffer."""
        if self._proc is None:
            return True
        ok = self._proc.waitForFinished(msec)
        self._on_ready()
        return ok

    def _on_ready(self):
        if self._proc is None:
            return
        self._buf += bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        lines, self._buf = split_lines(self._buf)
        for text in lines:
            if any(marker in text for marker in _OOM_MARKERS):
                self._saw_oom = True
            self.line.emit(text)
            parsed = parse_progress_line(text)
            if parsed:
                self.progress.emit(parsed)

    def _on_finished(self, exit_code, _status):
        self._on_ready()   # drain whatever arrived with the exit
        if self._buf.strip():
            # Check for OOM markers in the partial line before emitting
            if any(marker in self._buf for marker in _OOM_MARKERS):
                self._saw_oom = True
            self.line.emit(self._buf.strip())
            self._buf = ""
        self.finished.emit(int(exit_code), self._saw_oom)
