"""The commit-charge probe.

Commit charge (RAM + pagefile) is the budget a 12B run actually exhausts, and it
is invisible to every VRAM check the app makes. These tests pin the two things
that matter: the numbers are real, and a probe that cannot answer says None
instead of guessing -- an unreadable counter must never block a run.
"""
import sys

from core import sysmem


def test_available_commit_reports_a_real_number():
    gb = sysmem.available_commit_gb()
    if sys.platform != "win32":
        assert gb is None
        return
    assert gb is not None
    # Any real Windows box sits inside this range; a bad struct layout or a
    # unit slip (bytes read as GB) lands far outside it.
    assert 0.0 < gb < 100_000.0


def test_top_committer_names_a_process_and_its_size():
    top = sysmem.top_committer()
    if sys.platform != "win32":
        assert top is None
        return
    assert top is not None
    name, gb = top
    assert name.strip()
    assert gb > 0.0


def test_the_probes_fail_soft_when_the_platform_cannot_answer(monkeypatch):
    monkeypatch.setattr(sysmem.sys, "platform", "linux")
    assert sysmem.available_commit_gb() is None
    assert sysmem.top_committer() is None
    assert sysmem.biggest_consumer_note() == ""


# ---------------------------------------------------------------------------
# Making room. This machine trains or it runs something else, never both, so a
# short reading closes the other thing (Forge Neo idling on 52 GB, the case that
# wrote this) rather than warning and hoping. The policy is pure and tested
# against a made-up process table; the closing and the probe are injected.

def test_closable_skips_this_app_its_launchers_and_the_desktop():
    rows = [(1, "python.exe", "krea forge main.py", 3.0),              # us
            (2, "explorer.exe", "explorer.exe", 4.0),                   # desktop
            (3, "Python.exe", r"D:\Tools\WebUI\venv\Scripts\Python.exe launch.py", 52.2),
            (4, "msedge.exe", "msedge.exe --type=renderer", 1.9),       # under the floor
            (5, "llama-server.exe", "llama-server.exe -m big.gguf", 6.5),
            # A run started by hand from OneTrainer: not ours, not protected by
            # tree, and still the one thing making room for training must not end.
            (6, "python.exe", r"D:\Tools\OneTrainer\venv\Scripts\python.exe scripts\train.py --config-path x.json", 49.0)]
    got = sysmem.closable(rows, protected={1})
    assert [pid for pid, _, _ in got] == [3, 5]                         # biggest first
    assert "WebUI" in got[0][1] and "(PID 3)" in got[0][1]


def test_protected_pids_include_this_process_and_its_launcher():
    import os
    pids = sysmem.protected_pids()
    assert os.getpid() in pids and os.getppid() in pids


def test_the_real_table_never_lists_this_process():
    import os
    assert os.getpid() not in [pid for pid, _, _ in sysmem.closable_committers()]


def _room(free_readings, cands, close_ok=lambda pid: True):
    """make_room over a scripted probe: each free() call takes the next reading
    and the last one repeats forever. Returns (closed_labels, closed_pids)."""
    readings = iter(free_readings)
    last = [None]
    def free():
        try:
            last[0] = next(readings)
        except StopIteration:
            pass
        return last[0]
    closed = []
    def close(pid):
        ok = close_ok(pid)
        if ok:
            closed.append(pid)
        return ok
    ticks = [0.0]
    def clock():
        ticks[0] += 1.0
        return ticks[0]
    out = sysmem.make_room(55.0, close=close, free=free, candidates=lambda: cands,
                           sleep=lambda s: None, clock=clock)
    return out, closed


def test_make_room_does_nothing_with_room_to_spare_or_no_answer():
    cands = [(3, "forge (PID 3)", 52.0)]
    assert _room([80.0], cands) == ([], [])
    assert _room([None], cands) == ([], [])


def test_make_room_closes_the_biggest_and_stops_once_the_line_is_cleared():
    cands = [(3, "forge (PID 3)", 52.0), (5, "llama (PID 5)", 6.5)]
    out, closed = _room([21.0, 73.0], cands)
    assert closed == [3]
    assert out == [("forge (PID 3)", 52.0)]


def test_make_room_keeps_going_while_the_reading_stays_short():
    cands = [(3, "a (PID 3)", 10.0), (5, "b (PID 5)", 6.0)]
    out, closed = _room([40.0, 50.0, 56.0], cands)
    assert closed == [3, 5]


def test_make_room_skips_a_process_that_will_not_die():
    cands = [(3, "a (PID 3)", 10.0), (5, "b (PID 5)", 6.0)]
    out, closed = _room([40.0, 47.0, 56.0], cands, close_ok=lambda pid: pid != 3)
    assert closed == [5]
    assert [label for label, _ in out] == ["b (PID 5)"]


def test_make_room_stops_waiting_when_the_share_never_shows_up():
    # The reading never rises: the settle loop must end on the clock, not spin.
    cands = [(3, "a (PID 3)", 10.0)]
    out, closed = _room([40.0, 40.0], cands)
    assert closed == [3]
    assert out == [("a (PID 3)", 10.0)]


def test_command_line_reads_exactly_length_not_up_to_a_nul():
    # NtQueryInformationProcess's UNICODE_STRING need not be NUL-terminated inside
    # Length; reading it as a C string runs past the end of the buffer.
    import ctypes
    text = "python.exe train.py"
    raw = ctypes.create_unicode_buffer(text + "GARBAGE-PAST-LENGTH")
    us = sysmem._UnicodeString()
    us.Length = len(text) * 2
    us.MaximumLength = us.Length
    us.Buffer = ctypes.cast(raw, ctypes.c_void_p).value
    assert sysmem._unicode_string_text(us) == text
