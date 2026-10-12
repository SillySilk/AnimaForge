"""Send files to the Windows Recycle Bin instead of deleting them outright.

Lets the Dataset tab drop a picture in one click, with no confirm dialog, while
a mis-click can still be undone from the Recycle Bin.
"""
import ctypes
from ctypes import wintypes

_FO_DELETE = 0x0003
_FOF_SILENT = 0x0004
_FOF_NOCONFIRMATION = 0x0010
_FOF_ALLOWUNDO = 0x0040
_FOF_NOERRORUI = 0x0400


class _SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("wFunc", wintypes.UINT),
        ("pFrom", wintypes.LPCWSTR),
        ("pTo", wintypes.LPCWSTR),
        ("fFlags", ctypes.c_ushort),
        ("fAnyOperationsAborted", wintypes.BOOL),
        ("hNameMappings", ctypes.c_void_p),
        ("lpszProgressTitle", wintypes.LPCWSTR),
    ]


def to_recycle_bin(paths) -> None:
    """Move every path in `paths` to the Recycle Bin in one shell call.
    Raises OSError if the shell refuses."""
    paths = [str(p) for p in paths]
    if not paths:
        return
    op = _SHFILEOPSTRUCTW()
    op.wFunc = _FO_DELETE
    op.pFrom = "\0".join(paths) + "\0"     # double-NUL-terminated list
    op.fFlags = _FOF_ALLOWUNDO | _FOF_NOCONFIRMATION | _FOF_SILENT | _FOF_NOERRORUI
    rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if rc != 0 or op.fAnyOperationsAborted:
        raise OSError(f"Recycle Bin refused the delete (code {rc:#x})")
