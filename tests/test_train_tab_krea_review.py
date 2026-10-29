"""Plan 3 review fixes: the Krea 2 path through the Train tab, Home and MainWindow."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from test_train_tab_onetrainer import FakeSession, _krea_tab

_app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    shown = []
    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, **k: shown.append(a[2] if len(a) > 2 else "") or QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    return shown


def _png(path):
    from PySide6.QtGui import QImage
    path.parent.mkdir(parents=True, exist_ok=True)
    img = QImage(8, 8, QImage.Format_RGB32)
    img.fill(0)
    img.save(str(path))


# 1 ---------------------------------------------------------------------------

def test_krea_samples_in_subfolders_are_shown_and_survive_the_end_of_the_run(tmp_path, monkeypatch):
    t, s = _krea_tab(tmp_path, monkeypatch)
    s._run_dir = str(tmp_path / "out" / "Mr X")
    _png(tmp_path / "out" / "Mr X" / "workspace" / "samples" / "0 - mrx portrait" / "a.png")
    t._start_training(confirm=False)
    t._poll_sample_dir()
    assert any(p.endswith("a.png") for p in t._preview_files)
    s.finish(str(tmp_path / "out" / "Mr X" / "Mr_X.safetensors"))
    assert any(p.endswith("a.png") for p in t._preview_files)


# 2 ---------------------------------------------------------------------------

def test_krea_advisories_and_the_trigger_note_reach_the_log(tmp_path, monkeypatch):
    from core.engines.onetrainer import engine
    monkeypatch.setattr(engine, "advisories", lambda folder, trig: ["3 images were cropped to fit."])
    t, s = _krea_tab(tmp_path, monkeypatch, trigger="")
    t._start_training(confirm=False)
    log = t._log_edit.toPlainText()
    assert "3 images were cropped to fit." in log
    assert "No trigger word set" in log and "Mr X" in log


def test_forge_it_names_the_trigger_before_captioning_a_krea_set(tmp_path, monkeypatch):
    from core.settings import AppSettings
    monkeypatch.setattr(AppSettings, "set", lambda self, k, v: None)
    from ui.main_window import MainWindow
    w = MainWindow()
    w._train_tab.set_model("krea2")
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "a.png").write_bytes(b"x")
    monkeypatch.setattr(w._dataset_tab, "get_folder_path", lambda: str(ds))
    w._train_tab.set_lora_name("Mr X")
    w._on_home_trigger_changed("")
    monkeypatch.setattr(w, "_qr_advance", lambda: None)   # stop before the caption phase runs
    w._on_home_run()
    assert w._dataset_tab.get_trigger_word() == "Mr X"
    assert w._train_tab._trigger_word == "Mr X"


# 3 ---------------------------------------------------------------------------

def test_a_session_that_raises_on_start_ends_the_run(tmp_path, monkeypatch):
    class Exploding(FakeSession):
        def start(self, params):
            raise OSError("output folder is read-only")

    t, s = _krea_tab(tmp_path, monkeypatch, session=Exploding())
    ends = []
    t.training_active.connect(ends.append)
    t._start_training(confirm=False)
    assert ends == [True, False] and not t.is_training()
    assert "read-only" in t._log_edit.toPlainText()


# 4 ---------------------------------------------------------------------------

def test_the_model_cannot_change_while_a_run_is_live(tmp_path, monkeypatch):
    t, s = _krea_tab(tmp_path, monkeypatch)
    t._start_training(confirm=False)
    t.set_model("anima")
    assert t.model() == "krea2"


def test_the_run_ends_on_the_engine_it_started_with(tmp_path, monkeypatch):
    from core import run_manifest, sets
    t, s = _krea_tab(tmp_path, monkeypatch)
    writes = []
    monkeypatch.setattr(run_manifest, "mark", lambda d, status: writes.append(status))
    monkeypatch.setattr(sets, "clear_active_run", lambda *a, **k: writes.append("kohya marker"))
    t._start_training(confirm=False)
    t._model = "anima"            # however it changed, the finish must follow the run
    s.finish(str(tmp_path / "x.safetensors"))
    # Both engines record the outcome in run.json (Past runs reads it); only a
    # Kohya run clears the crash-recovery marker it set at start.
    assert writes == [run_manifest.DONE]


def test_home_locks_model_and_style_while_training():
    from ui.home_tab import HomeTab
    h = HomeTab()
    h.set_training_active(True)
    assert not h._model_combo.isEnabled() and not h._style_combo.isEnabled()
    h.set_training_active(False)
    assert h._model_combo.isEnabled() and h._style_combo.isEnabled()


# 5 ---------------------------------------------------------------------------

def test_a_loaded_sets_caption_style_reaches_home_and_dataset(monkeypatch):
    from core.settings import AppSettings
    monkeypatch.setattr(AppSettings, "set", lambda self, k, v: None)
    from core.batch import RunDefinition
    from ui.main_window import MainWindow
    w = MainWindow()
    w._train_tab.apply_run_definition(RunDefinition("K", "", 0, model="krea2", caption_style="prose"))
    assert w._home_tab._style_combo.currentData() == "prose"
    assert w._dataset_tab._caption_style_now() == "prose"
    w._train_tab.apply_run_definition(RunDefinition("K", "", 0, model="krea2", caption_style="tags"))
    assert w._home_tab._style_combo.currentData() == "tags"
    assert w._dataset_tab._caption_style_now() == "tags"


# 6 ---------------------------------------------------------------------------

def test_a_krea_lora_is_not_auto_delivered_or_test_rendered_in_forge(tmp_path, monkeypatch):
    from core.settings import DEFAULTS, AppSettings
    t, s = _krea_tab(tmp_path, monkeypatch)
    t._app_settings = AppSettings()
    monkeypatch.setattr(AppSettings, "get", lambda self, k: True if k in (
        "forge_auto_deliver", "forge_auto_test") else DEFAULTS.get(k))
    calls = []
    monkeypatch.setattr(t, "_deliver_to_forge", lambda **k: calls.append("deliver"))
    monkeypatch.setattr(t, "_test_in_forge", lambda **k: calls.append("test"))
    t._start_training(confirm=False)
    s.finish(str(tmp_path / "x.safetensors"))
    assert calls == []
    assert "not sent to Forge" in t._log_edit.toPlainText()


# 7 ---------------------------------------------------------------------------

def test_closing_mid_run_asks_and_stops_the_run(monkeypatch):
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QMessageBox
    from core.settings import AppSettings
    monkeypatch.setattr(AppSettings, "set", lambda self, k, v: None)
    from ui.main_window import MainWindow
    w = MainWindow()
    stopped = []
    monkeypatch.setattr(w._train_tab, "is_training", lambda: True)
    monkeypatch.setattr(w._train_tab, "stop_for_exit", lambda: stopped.append(True))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    e = QCloseEvent()
    w.closeEvent(e)
    assert not e.isAccepted() and stopped == []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    e = QCloseEvent()
    w.closeEvent(e)
    assert e.isAccepted() and stopped == [True]


def test_stop_for_exit_cancels_a_krea_session(tmp_path, monkeypatch):
    t, s = _krea_tab(tmp_path, monkeypatch)
    t._start_training(confirm=False)
    t.stop_for_exit()
    assert s.cancels == 1
