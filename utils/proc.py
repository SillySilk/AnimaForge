"""Windows subprocess helpers — keep child console apps from popping a window.

The app is a GUI (no console of its own when launched via pythonw, and even when
launched from a console Qt may give children their own window). Every child we
spawn — python.exe for training/tagging/captioning, nvidia-smi, tasklist, pip —
is a *console-subsystem* executable, so Windows opens a console window for each
one unless we pass CREATE_NO_WINDOW. We already capture all child output through
QProcess pipes / subprocess capture, so the window is pure noise.

CREATE_NO_WINDOW (0x08000000) suppresses the console without affecting stdout/
stderr redirection. These helpers are no-ops off Windows.
"""
import sys

CREATE_NO_WINDOW = 0x08000000


def apply_no_window(process) -> None:
    """Make a QProcess launch its child without a console window (Windows only).

    Call after constructing the QProcess and before start(). Safe to call on any
    platform / Qt build: it silently does nothing where the API is unavailable.
    """
    if sys.platform != "win32":
        return
    setter = getattr(process, "setCreateProcessArgumentsModifier", None)
    if setter is None:
        return

    def _modifier(args):
        # args is a QProcess.CreateProcessArguments; .flags is read/write.
        args.flags |= CREATE_NO_WINDOW

    setter(_modifier)


def no_window_creationflags() -> int:
    """creationflags value for subprocess.* — CREATE_NO_WINDOW on Windows, else 0."""
    return CREATE_NO_WINDOW if sys.platform == "win32" else 0
