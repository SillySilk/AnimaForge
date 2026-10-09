import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox

_app = QApplication.instance() or QApplication([])

from core.batch import DONE, FAILED, RunDefinition


def test_home_button_asks_and_locks_while_training():
    from ui.home_tab import HomeTab
    h = HomeTab()
    seen = []
    h.compare_requested.connect(lambda: seen.append(1))
    h._compare_btn.click()
    assert seen == [1]
    h.set_training_active(True)
    assert not h._compare_btn.isEnabled()


def test_view_renders_a_row_per_prompt_and_labels_missing_sides(tmp_path):
    from PySide6.QtGui import QImage
    from ui.compare_view import CompareView
    img = tmp_path / "a.png"
    QImage(16, 16, QImage.Format_RGB32).save(str(img))
    entry = {"set": "Mr X", "runs": [{"name": "a", "style": "prose+tags"}, {"name": "b", "style": "tags"}]}
    rows = [{"prompt": "p0", "a": str(img), "b": None, "status": {"a": "done", "b": "failed"}},
            {"prompt": "p1", "a": str(img), "b": None, "status": {"a": "done", "b": "pending"}}]
    v = CompareView(entry, rows)
    texts = [l.text() for l in v.findChildren(QLabel)]
    assert "p0" in texts and "p1" in texts
    assert any("failed" in t for t in texts) and any("not finished" in t for t in texts)
    assert any("Prose + tags" in t and "Tags" in t for t in texts)


@pytest.fixture
def win(monkeypatch, tmp_path):
    from ui.main_window import MainWindow
    warns = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warns.append(a[2]) or QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.Ok)
    w = MainWindow()
    w._compare_root = tmp_path
    queued = []
    monkeypatch.setattr(w._batch_tab, "add_run", queued.append)
    ds = tmp_path / "ds"
    ds.mkdir()
    rd = RunDefinition(lora_name="Mr X", dataset_folder=str(ds), image_count=3,
                       caption_style="prose+tags", sample_prompts=["p0", "p1"])
    monkeypatch.setattr(w._train_tab, "build_run_definition", lambda: (rd, ""))
    return w, queued, warns, rd


def test_queueing_adds_two_runs_and_a_record(win, tmp_path):
    from core import compare
    w, queued, warns, _rd = win
    assert w._queue_comparison("tags") is True
    assert [r.lora_name.rsplit("-", 2)[0] for r in queued] == ["Mr X-prose-tags", "Mr X-tags"]
    assert len(compare.load_all(root=tmp_path)) == 1
    assert "start the batch" in w._status_bar.currentMessage()


def test_no_prompts_means_nothing_queued(win, monkeypatch):
    import core.batch as batch_mod
    w, queued, warns, rd = win
    rd.sample_prompts = []
    monkeypatch.setattr(batch_mod, "resolve_sample_prompts", lambda r: [])
    assert w._queue_comparison("tags") is False
    assert queued == [] and "sample prompt" in warns[-1]


def test_finishing_both_runs_opens_the_view_once(win, monkeypatch):
    w, queued, warns, _rd = win
    w._queue_comparison("tags")
    shown = []
    monkeypatch.setattr(w, "_show_comparison", lambda gid: shown.append(gid))
    a, b = queued
    monkeypatch.setattr(w._batch_tab, "runs", lambda: [a, b])
    a.status = DONE
    w._on_batch_run_done(a, True)
    assert shown == []
    b.status = FAILED
    w._on_batch_run_done(b, False)
    assert shown == [a.compare_group]
    w._on_batch_run_done(b, False)
    assert shown == [a.compare_group]
