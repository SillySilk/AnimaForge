"""Resolve the Python interpreter for ML subprocesses.

Since the unified install runs the GUI from a single `.venv` that already contains
every dependency (torch, transformers, editable sd-scripts, onnxruntime), subprocesses
should reuse that interpreter (`sys.executable`). A legacy two-venv install
(`sd-scripts/venv`) is honored only as a fallback when the GUI is NOT running inside a
venv. See docs/superpowers/specs/2026-06-24-venv-install-design.md.

IMPORTANT — never hand back the windowless `pythonw.exe` for a subprocess. The GUI is
launched with `pythonw.exe` (so `sys.executable` is `pythonw.exe`), but `pythonw` is a
GUI-subsystem binary: when it spawns a child that itself spawns a child (the venv
redirector + accelerate's worker, or any `pythonw -> pythonw` chain), the *grandchild's*
stdout/stderr do NOT propagate up the QProcess pipe. The result is a training run that
works perfectly while the UI sits frozen on "Preparing… 0/steps" because no progress
line ever reaches `_parse_progress`. The console `python.exe` (paired with the
CREATE_NO_WINDOW flag every launcher already sets) propagates grandchild output and shows
no console window. So always resolve to the console interpreter.
"""
import sys
from pathlib import Path


def _to_console_python(exe: str) -> str:
    """Map a windowless `pythonw.exe` path to its console `python.exe` sibling.

    Returns `exe` unchanged when it is not a `pythonw.exe` or the sibling is missing.
    Pure/string-based so it stays unit-testable; the on-disk check is done in the caller.
    """
    p = Path(exe)
    if p.name.lower() == "pythonw.exe":
        return str(p.with_name("python.exe"))
    return exe


def _choose_python(in_venv: bool, legacy_path: str, legacy_exists: bool, sys_exe: str) -> str:
    """Pure selection logic (so it can be unit-tested without touching sys)."""
    if in_venv:
        return sys_exe
    if legacy_path and legacy_exists:
        return legacy_path
    return sys_exe


def subprocess_python(sdscripts_path: str = "") -> str:
    """Interpreter to launch ML subprocesses (tagger, JoyCaption, LLM, trainer) with."""
    legacy = ""
    if sdscripts_path:
        legacy = str(Path(sdscripts_path) / "venv" / "Scripts" / "python.exe")
    chosen = _choose_python(
        sys.prefix != sys.base_prefix,
        legacy,
        bool(legacy) and Path(legacy).is_file(),
        sys.executable,
    )
    # Prefer the console interpreter so grandchild progress reaches the UI (see module
    # docstring). Only swap when the console sibling actually exists on disk.
    console = _to_console_python(chosen)
    if console != chosen and Path(console).is_file():
        return console
    return chosen
