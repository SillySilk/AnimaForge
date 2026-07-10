import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

import ui.batch_tab as batch_tab_mod
from ui.batch_tab import BatchTab
from core.batch import RunDefinition, QUEUED, DONE
from core import caption_policy as cp
from core.settings import AppSettings

_app = QApplication.instance() or QApplication([])


def _rd(**kw):
    base = dict(lora_name="lr", dataset_folder="C:/d", image_count=4, status=QUEUED)
    base.update(kw)
    return RunDefinition(**base)


def _folder(tmp_path: Path, name: str, captioned: bool = False) -> Path:
    """A real dataset folder with one image, optionally already captioned —
    cp.scan() reads the actual filesystem, so tests need genuine files, not mocks."""
    d = tmp_path / name
    d.mkdir()
    (d / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    if captioned:
        (d / "a.txt").write_text("a cat on a mat", encoding="utf-8")
    return d


def _make_tab(tmp_path: Path, monkeypatch) -> BatchTab:
    """A BatchTab whose persistence goes to a scratch file, never the repo's real
    batch_queue.json — QUEUE_PATH is a module-level global read fresh on every
    load_queue()/save_queue() call, so patching the module attribute before
    construction redirects both."""
    monkeypatch.setattr(batch_tab_mod, "QUEUE_PATH", str(tmp_path / "batch_queue.json"))
    return BatchTab()


def _with_policy(policy):
    """Force AppSettings().get('caption_existing_policy') for the duration of a
    test, restoring the real value afterward (same idiom as
    tests/test_dataset_tab_captioning.py). Caller must restore `prev` in a
    finally block — this touches the real registry-backed setting."""
    app = AppSettings()
    prev = app.get("caption_existing_policy")
    app.set("caption_existing_policy", policy)
    return app, prev


# ----------------------------------------------------------------------------
# _resolve_conflicts — pure decision logic, dialog patched via QDialog.exec
# ----------------------------------------------------------------------------

def test_resolve_conflicts_no_conflicts_returns_true_no_dialog(tmp_path, monkeypatch):
    t = _make_tab(tmp_path, monkeypatch)
    r1 = _rd(lora_name="a", dataset_folder=str(_folder(tmp_path, "a")))
    r2 = _rd(lora_name="b", dataset_folder=str(_folder(tmp_path, "b")))
    t._runs = [r1, r2]

    def _boom(self):
        raise AssertionError("no conflicts -> dialog must not be shown")
    monkeypatch.setattr(QDialog, "exec", _boom)

    assert t._resolve_conflicts([r1, r2]) is True
    assert r1.caption_policy == cp.OVERWRITE
    assert r2.caption_policy == cp.OVERWRITE


def test_resolve_conflicts_persists_the_queue(tmp_path, monkeypatch):
    from core.batch import load_queue
    t = _make_tab(tmp_path, monkeypatch)
    r1 = _rd(lora_name="a", dataset_folder=str(_folder(tmp_path, "a")))
    t._runs = [r1]

    def _boom(self):
        raise AssertionError("no conflicts -> dialog must not be shown")
    monkeypatch.setattr(QDialog, "exec", _boom)

    assert t._resolve_conflicts([r1]) is True
    reloaded = load_queue(str(tmp_path / "batch_queue.json"))
    assert reloaded[0].caption_policy == cp.OVERWRITE


def test_resolve_conflicts_scans_each_folder_exactly_once(tmp_path, monkeypatch):
    t = _make_tab(tmp_path, monkeypatch)
    r1 = _rd(lora_name="a", dataset_folder=str(_folder(tmp_path, "a", captioned=True)))
    r2 = _rd(lora_name="b", dataset_folder=str(_folder(tmp_path, "b")))
    t._runs = [r1, r2]

    calls = []
    real_scan = cp.scan

    def _counting_scan(folder, **kw):
        calls.append(folder)
        return real_scan(folder, **kw)
    monkeypatch.setattr(cp, "scan", _counting_scan)

    app, prev = _with_policy(cp.KEEP)   # skip the modal, isolate the scan count
    try:
        assert t._resolve_conflicts([r1, r2]) is True
    finally:
        app.set("caption_existing_policy", prev)

    assert len(calls) == 2   # exactly once per candidate, not once per dialog widget


def test_resolve_conflicts_default_policy_applies_silently(tmp_path, monkeypatch):
    t = _make_tab(tmp_path, monkeypatch)
    r1 = _rd(lora_name="a", dataset_folder=str(_folder(tmp_path, "a", captioned=True)))
    t._runs = [r1]

    def _boom(self):
        raise AssertionError("a non-ASK default must resolve without a dialog")
    monkeypatch.setattr(QDialog, "exec", _boom)

    app, prev = _with_policy(cp.KEEP)
    try:
        assert t._resolve_conflicts([r1]) is True
    finally:
        app.set("caption_existing_policy", prev)
    assert r1.caption_policy == cp.KEEP


def test_resolve_conflicts_user_picks_keep(tmp_path, monkeypatch):
    t = _make_tab(tmp_path, monkeypatch)
    r1 = _rd(lora_name="conflicted", dataset_folder=str(_folder(tmp_path, "c", captioned=True)))
    t._runs = [r1]

    def _fake_exec(self):
        from PySide6.QtWidgets import QComboBox
        for combo in self.findChildren(QComboBox):
            combo.setCurrentIndex(combo.findData(cp.KEEP))
        return QDialog.Accepted
    monkeypatch.setattr(QDialog, "exec", _fake_exec)

    app, prev = _with_policy(cp.ASK)
    try:
        assert t._resolve_conflicts([r1]) is True
    finally:
        app.set("caption_existing_policy", prev)
    assert r1.caption_policy == cp.KEEP


def test_resolve_conflicts_cancel_returns_false_and_does_not_mutate(tmp_path, monkeypatch):
    t = _make_tab(tmp_path, monkeypatch)
    r1 = _rd(lora_name="conflicted", dataset_folder=str(_folder(tmp_path, "c", captioned=True)),
             caption_policy=cp.ASK)
    t._runs = [r1]

    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.Rejected)

    app, prev = _with_policy(cp.ASK)
    try:
        assert t._resolve_conflicts([r1]) is False
    finally:
        app.set("caption_existing_policy", prev)
    assert r1.caption_policy == cp.ASK   # untouched — cancel must not decide for the user


# ----------------------------------------------------------------------------
# _start / _restart — the surfaces that call _resolve_conflicts
# ----------------------------------------------------------------------------

def test_start_on_all_done_queue_shows_message_and_skips_runner_start(tmp_path, monkeypatch):
    t = _make_tab(tmp_path, monkeypatch)
    r1 = _rd(lora_name="done", dataset_folder=str(_folder(tmp_path, "d")), status=DONE)
    t._runs = [r1]

    info_calls = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **kw: info_calls.append(a) or QMessageBox.Ok)
    started = []
    monkeypatch.setattr(t._runner, "start", lambda *a, **kw: started.append((a, kw)))

    t._start()

    assert len(info_calls) == 1
    assert started == []


def test_restart_resolves_conflicts_for_done_runs_too(tmp_path, monkeypatch):
    t = _make_tab(tmp_path, monkeypatch)
    # A DONE run whose folder still has captions -- restart() resets every status
    # to QUEUED, so this folder WILL be captioned again if the policy says so.
    r_done = _rd(lora_name="finished", dataset_folder=str(_folder(tmp_path, "f", captioned=True)),
                 status=DONE, caption_policy=cp.ASK)
    r_queued = _rd(lora_name="pending", dataset_folder=str(_folder(tmp_path, "p")), status=QUEUED)
    t._runs = [r_done, r_queued]

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.Yes)
    restarted = []
    monkeypatch.setattr(t._runner, "restart", lambda *a, **kw: restarted.append((a, kw)))

    app, prev = _with_policy(cp.KEEP)   # bypass the modal, isolate the "which runs" question
    try:
        t._restart()
    finally:
        app.set("caption_existing_policy", prev)

    assert r_done.caption_policy == cp.KEEP   # resolved even though it was DONE
    assert len(restarted) == 1


def test_restart_on_empty_queue_does_nothing(tmp_path, monkeypatch):
    t = _make_tab(tmp_path, monkeypatch)
    t._runs = []

    def _boom(*a, **kw):
        raise AssertionError("empty queue -> no confirm dialog, no restart call")
    monkeypatch.setattr(QMessageBox, "question", _boom)
    restarted = []
    monkeypatch.setattr(t._runner, "restart", lambda *a, **kw: restarted.append(1))

    t._restart()

    assert restarted == []


def test_restart_cancel_confirm_does_not_call_restart(tmp_path, monkeypatch):
    t = _make_tab(tmp_path, monkeypatch)
    r1 = _rd(lora_name="a", dataset_folder=str(_folder(tmp_path, "a")), status=DONE)
    t._runs = [r1]

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.No)
    restarted = []
    monkeypatch.setattr(t._runner, "restart", lambda *a, **kw: restarted.append(1))

    t._restart()

    assert restarted == []
