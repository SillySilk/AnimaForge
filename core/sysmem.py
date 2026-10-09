"""How much the system can still commit, and who is holding it.

Commit charge -- physical RAM plus pagefile -- is the budget a training run
actually exhausts, and nothing else in the app can see it. A 12B run commits
~50 GB while it quantizes the model to INT8, long before a single step runs and
entirely outside VRAM, so every GPU check the app makes reports "fine" right up
to the crash.

When the system commit limit is reached, the allocation inside PyTorch returns
null and c10.dll dereferences it: Windows kills python.exe with an access
violation (0xC0000005). There is no traceback, no OOM marker, no last words --
trainer.log simply stops mid-progress-bar. The run_A run on 2026-08-17 died
that way at 36% of quantization, holding 51.6 GB, while an idle Forge Neo held
52.2 GB and the commit limit was 103.8 GB. Windows diagnosed it correctly in the
event log; the app had nothing to say.

Windows-only by design (so is the app), and every probe fails soft: a counter the
OS declines to report must never block a run that would have worked.
"""
import ctypes
import os
import sys
import time

import psutil

_GB = float(1024 ** 3)

# Enough headroom for any real machine; EnumProcesses truncates rather than fails,
# and a truncated sample still finds the multi-GB processes we care about.
_MAX_PIDS = 8192
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_MAX_PATH_W = 32768

# ProcessCommandLineInformation. The image path alone cannot identify these
# processes: a venv's Scripts\python.exe redirects to the base interpreter, so
# every Python app on the machine reports the same executable. The command line
# is what tells "an idle image generator" apart from "the trainer".
_PROCESS_COMMAND_LINE_INFORMATION = 60
_LABEL_CHARS = 90

# Below this a "close it" suggestion is noise -- nothing the user shuts down will
# free enough to matter.
_NOTE_FLOOR_GB = 5.0


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


_DLLS = {}


def _dll(name: str):
    """This module's own handle on a system DLL. Setting argtypes/restype on the
    shared ctypes.windll objects would change them for every ctypes user in the
    process (Qt, PyTorch helpers, other modules)."""
    if name not in _DLLS:
        _DLLS[name] = ctypes.WinDLL(name)
    return _DLLS[name]


class _UnicodeString(ctypes.Structure):
    # Buffer is a raw pointer: the text is exactly Length bytes and need not be
    # NUL-terminated, so it must never be read as a C string.
    _fields_ = [("Length", ctypes.c_ushort),
                ("MaximumLength", ctypes.c_ushort),
                ("Buffer", ctypes.c_void_p)]


def _unicode_string_text(us) -> str:
    if not us.Buffer or not us.Length:
        return ""
    return ctypes.wstring_at(us.Buffer, us.Length // ctypes.sizeof(ctypes.c_wchar))


class _ProcessMemoryCounters(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS. PagefileUsage is the process's commit charge --
    what it has reserved from the system-wide limit, paged out or not. Working
    set is the wrong number here: the process that took 52 GB of commit had a
    20 MB working set because it had been idle for hours."""
    _fields_ = [("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def available_commit_gb():
    """GB the system can still commit, or None when it cannot be read.

    ullAvailPageFile is exactly the pool a new process allocates from: the commit
    limit minus everything already committed by every running process.
    """
    if sys.platform != "win32":
        return None
    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not _dll("kernel32").GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.ullAvailPageFile / _GB
    except (AttributeError, OSError, ValueError):
        return None


def _process_ids():
    psapi = _dll("psapi")
    array = (ctypes.c_ulong * _MAX_PIDS)()
    returned = ctypes.c_ulong()
    if not psapi.EnumProcesses(ctypes.byref(array), ctypes.sizeof(array),
                               ctypes.byref(returned)):
        return []
    return list(array[:returned.value // ctypes.sizeof(ctypes.c_ulong)])


def _command_line(handle) -> str:
    """The process's full command line, or "" when the OS declines to give it."""
    try:
        ntdll = _dll("ntdll")
        ntdll.NtQueryInformationProcess.restype = ctypes.c_long
        ntdll.NtQueryInformationProcess.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong)]
        needed = ctypes.c_ulong()
        # First call sizes the buffer; it is expected to fail with
        # STATUS_INFO_LENGTH_MISMATCH, so only the length it reports is used.
        ntdll.NtQueryInformationProcess(
            handle, _PROCESS_COMMAND_LINE_INFORMATION, None, 0,
            ctypes.byref(needed))
        if not 0 < needed.value <= (1 << 20):
            return ""
        buf = ctypes.create_string_buffer(needed.value)
        if ntdll.NtQueryInformationProcess(
                handle, _PROCESS_COMMAND_LINE_INFORMATION, buf, needed.value,
                ctypes.byref(needed)) != 0:
            return ""
        return _unicode_string_text(ctypes.cast(buf, ctypes.POINTER(_UnicodeString)).contents)
    except (AttributeError, OSError, ValueError):
        return ""


def _label(handle, pid: int) -> str:
    """The most identifying name available, best first: command line, image
    path, bare PID. Something the user can recognize in Task Manager."""
    text = " ".join(_command_line(handle).replace('"', "").split())
    if not text:
        name = ctypes.create_unicode_buffer(_MAX_PATH_W)
        size = ctypes.c_ulong(_MAX_PATH_W)
        if _dll("kernel32").QueryFullProcessImageNameW(
                handle, 0, name, ctypes.byref(size)):
            text = name.value
    if not text:
        return f"PID {pid}"
    if len(text) > _LABEL_CHARS:
        text = text[:_LABEL_CHARS - 1] + "…"
    return f"{text} (PID {pid})"


def _commit_gb(handle):
    """Commit charge in GB for an open process handle, or None."""
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    if not _dll("psapi").GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb):
        return None
    return counters.PagefileUsage / _GB


def top_committer():
    """(name, gb) of the process holding the most commit charge, or None.

    Processes the app may not open (System, other users, protected) are skipped:
    the point is to name something the user can close, and those are not it.
    """
    if sys.platform != "win32":
        return None
    try:
        kernel32 = _dll("kernel32")
        kernel32.OpenProcess.restype = ctypes.c_void_p
        best_gb, best_name = 0.0, ""
        for pid in _process_ids():
            if not pid:
                continue
            handle = kernel32.OpenProcess(
                _PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                continue
            handle = ctypes.c_void_p(handle)
            try:
                gb = _commit_gb(handle)
                name = _label(handle, pid) if gb and gb > best_gb else ""
            finally:
                kernel32.CloseHandle(handle)
            if name:
                best_gb, best_name = gb, name
        return (best_name, best_gb) if best_name else None
    except (AttributeError, OSError, ValueError):
        return None


def biggest_consumer_note() -> str:
    """One sentence naming what to close, or "" when nothing worth closing."""
    top = top_committer()
    if not top or top[1] < _NOTE_FLOOR_GB:
        return ""
    name, gb = top
    return f" The biggest consumer right now is {name} ({gb:.0f} GB)."


# ---------------------------------------------------------------------------
# Making room. When the reading is short, the app closes whatever else is holding
# memory rather than warning and hoping: this machine trains or it runs something
# else, never both (maintainer's rule, 2026-09-04). Forge Neo idling on 52 GB is
# the case that wrote this; a browser holding 3 GB is the collateral it accepts.

# Below this a process is not worth closing -- what it gives back is noise next
# to the 50 GB the run needs, and the collateral (a chat app, a browser) is real.
CLOSE_FLOOR_GB = 2.0

# Windows keeps a desktop with these. Closing one takes the session down with it,
# and none of them is where 50 GB goes. Lower-case image names.
NEVER_CLOSE = frozenset((
    "system", "registry", "memcompression", "smss.exe", "csrss.exe",
    "wininit.exe", "winlogon.exe", "services.exe", "lsass.exe", "svchost.exe",
    "dwm.exe", "explorer.exe", "fontdrvhost.exe", "sihost.exe", "ctfmon.exe",
    "conhost.exe", "runtimebroker.exe", "taskhostw.exe", "audiodg.exe",
    "spoolsv.exe", "searchhost.exe", "searchindexer.exe", "msmpeng.exe",
    "startmenuexperiencehost.exe", "shellexperiencehost.exe", "textinputhost.exe",
    "applicationframehost.exe", "securityhealthservice.exe", "lockapp.exe",
))

# Commit is released as a process unwinds, not the instant it is signalled; the
# reading is polled this long for a closed process's share to show up.
_SETTLE_SECONDS = 10.0
_SETTLE_POLL = 0.25


def protected_pids() -> set:
    """This app, everything it launched, and everything that launched it.

    The launcher chain matters: a run started from a terminal must not close the
    terminal, and the app's own captioner (a child) is not "another program".
    """
    try:
        me = psutil.Process()
        pids = {me.pid}
        pids.update(p.pid for p in me.parents())
        pids.update(p.pid for p in me.children(recursive=True))
        return pids
    except psutil.Error:
        return {os.getpid()}


def closable(rows, protected) -> list:
    """[(pid, label, gb)] biggest first, from rows of (pid, name, cmdline, gb).

    Pure: the policy of what may be closed, kept apart from the enumeration so it
    can be tested against a made-up process table.
    """
    out = []
    for pid, name, cmdline, gb in rows:
        if pid in protected or (name or "").lower() in NEVER_CLOSE:
            continue
        if gb < CLOSE_FLOOR_GB:
            continue
        text = " ".join((cmdline or "").replace('"', "").split()) or (name or "")
        # A training run is the one thing making room for training must never
        # end -- not one this app started (protected above) and not one started
        # by hand from OneTrainer either. This rule wrote itself the day the test
        # suite closed a live run.
        if "train.py" in text.lower():
            continue
        if len(text) > _LABEL_CHARS:
            text = text[:_LABEL_CHARS - 1] + "…"
        out.append((pid, f"{text} (PID {pid})", gb))
    out.sort(key=lambda row: -row[2])
    return out


def _process_rows() -> list:
    """(pid, name, cmdline, commit_gb) for every process the OS will describe."""
    if sys.platform != "win32":
        return []
    rows = []
    for proc in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
        try:
            mem = proc.info["memory_info"]
            if mem is None:
                continue
            cmdline = " ".join(proc.info["cmdline"] or [])
            rows.append((proc.info["pid"], proc.info["name"] or "", cmdline,
                         mem.pagefile / _GB))
        except psutil.Error:
            continue
    return rows


def closable_committers() -> list:
    """Every other program worth closing to make room, biggest first."""
    return closable(_process_rows(), protected_pids())


def close_tree(pid: int) -> bool:
    """End `pid` and everything under it. True when it is gone."""
    try:
        root = psutil.Process(pid)
        procs = root.children(recursive=True) + [root]
    except psutil.Error:
        return False
    for proc in procs:
        try:
            proc.kill()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=5)
    return not alive


def plan_room(target_free_gb: float, *, free=available_commit_gb,
              candidates=closable_committers) -> tuple:
    """(free_gb | None, [(pid, label, gb)]): the biggest-first programs that would
    have to close for `target_free_gb` to be committable. Closes nothing -- the
    caller shows this list and asks before anything is ended."""
    have = free()
    if have is None or have >= target_free_gb:
        return have, []
    plan, expect = [], have
    for row in candidates():
        plan.append(row)
        expect += row[2]
        if expect >= target_free_gb:
            break
    return have, plan


def make_room(target_free_gb: float, *, close=close_tree, free=available_commit_gb,
              candidates=closable_committers, sleep=time.sleep,
              clock=time.monotonic) -> list:
    """Close the biggest other programs until `target_free_gb` can be committed.

    Returns [(label, gb)] for each process closed, in order, whether or not the
    target was reached in the end -- the caller says what was closed either way.
    Empty when there was room already, when the probe cannot answer, or when
    nothing closable is holding enough to matter. Biggest first, and it stops the
    moment the reading clears the line, so a small hog survives a big one.
    """
    have = free()
    if have is None or have >= target_free_gb:
        return []
    closed = []
    for pid, label, gb in candidates():
        before = have
        if not close(pid):
            continue
        closed.append((label, gb))
        deadline = clock() + _SETTLE_SECONDS
        while True:
            have = free()
            if have is None or have >= target_free_gb or have - before >= gb * 0.5:
                break
            if clock() >= deadline:
                break
            sleep(_SETTLE_POLL)
        if have is None or have >= target_free_gb:
            break
    return closed
