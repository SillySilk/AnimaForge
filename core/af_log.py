"""Rotating caption log file.

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
