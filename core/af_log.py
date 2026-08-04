"""Rotating log files: the caption log (append-only) and the training log (per run).

The app had no on-disk logging at all: every log path was an in-memory Qt signal
feeding a UI pane, so once a captioning run failed there was nothing left to
inspect and no way to ask a bug reporter for detail. This writes one file.

Deliberately tiny — no logging framework, no config. Logging must never be the
reason a caption run fails, so every write is best-effort and swallows OSError.
"""
import time
from pathlib import Path

MAX_BYTES = 2 * 1024 * 1024  # rotate at ~2MB, keep one previous file


def log_path() -> Path:
    from core.paths import app_root
    return app_root() / "logs" / "caption.log"


def train_log_path() -> Path:
    from core.paths import app_root
    return app_root() / "logs" / "training.log"


def rotate_if_needed(path: Path, max_bytes: int) -> None:
    """Move an oversized log to '<name>.log.1', replacing any previous one."""
    try:
        if path.is_file() and path.stat().st_size >= max_bytes:
            path.replace(path.with_suffix(".log.1"))
    except OSError:
        pass


def append(line: str, path=None, max_bytes: int = MAX_BYTES) -> None:
    """Append one timestamped line. Never raises."""
    p = Path(path) if path is not None else log_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        rotate_if_needed(p, max_bytes)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"{stamp}  {line}\n")
    except OSError:
        pass


class RunLog:
    """The current training run's log, on disk.

    A training run's output only ever existed in a QTextEdit, so a run that died
    before the first step left nothing to attach to a bug report — and "paste me
    the first few lines" is a poor substitute when the interesting part is a
    traceback the reporter has to scroll back to find.

    Scoped to one run rather than appended forever: the previous run moves to
    '<name>.log.1' at open, so "the training log" always means the last run and
    the one before it. Held open across the run (a training log is thousands of
    lines) but flushed per line, because the runs worth reading are the ones that
    end in a crash.

    Same contract as append(): best-effort, never raises, never the reason a run
    fails.
    """

    def __init__(self, path=None, max_bytes: int = MAX_BYTES):
        self._path = Path(path) if path is not None else train_log_path()
        self._max_bytes = max_bytes
        self._fh = None

    @property
    def path(self) -> Path:
        return self._path

    def open(self, header: str = "") -> None:
        self.close()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Unconditional, unlike append()'s size check: one run per file.
            if self._path.is_file():
                self._path.replace(self._path.with_suffix(".log.1"))
            self._fh = open(self._path, "w", encoding="utf-8")
        except OSError:
            self._fh = None
            return
        if header:
            self.write(header)

    def write(self, line: str) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")
            self._fh.flush()
        except (OSError, ValueError):
            pass

    def close(self, footer: str = "") -> None:
        if self._fh is None:
            return
        if footer:
            self.write(footer)
        try:
            self._fh.close()
        except OSError:
            pass
        self._fh = None
