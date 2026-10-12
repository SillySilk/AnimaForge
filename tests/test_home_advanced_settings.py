import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def warnings(monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    seen = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: seen.append(a[2]) or QMessageBox.Ok)
    for name in ("critical", "information"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    return seen


def test_home_button_asks_for_the_editor_and_locks_while_training():
    from ui.home_tab import HomeTab
    h = HomeTab()
    seen = []
    h.advanced_settings_requested.connect(lambda: seen.append(1))
    h._advanced_btn.click()
    assert seen == [1]
    h.set_training_active(True)
    assert not h._advanced_btn.isEnabled()
    h.set_training_active(False)
    assert h._advanced_btn.isEnabled()


def test_home_shows_how_many_settings_changed():
    from ui.home_tab import HomeTab
    h = HomeTab()
    h.set_advanced_count(0)
    assert h._advanced_label.isHidden()
    h.set_advanced_count(2)
    assert not h._advanced_label.isHidden() and h._advanced_label.text() == "Advanced: 2 changed"


def _window(monkeypatch, base):
    from ui.main_window import MainWindow
    w = MainWindow()
    # MainWindow restores the last-used model from the real user settings; these
    # tests are about an Anima run (on Krea 2, "epochs" is a run-locked field, so an
    # override of it is dropped). Pin the model without signals, so the switch is not
    # written back to the user's settings.
    w._train_tab.blockSignals(True)
    w._train_tab.set_model("anima")
    w._train_tab.blockSignals(False)
    monkeypatch.setattr(w._train_tab, "base_config", lambda: base)
    return w


def test_apply_sets_the_runs_overrides_and_home_count(monkeypatch):
    w = _window(monkeypatch, ({"epochs": 1, "lr": 1.0}, ""))
    w._home_tab._advanced_btn.click()
    panel = w._advanced_panel
    panel.table.item(0, 1).setText("5")          # rows are sorted: "epochs" first
    assert w._apply_advanced() is True
    assert w._train_tab.config_overrides() == {"epochs": 5}
    assert w._home_tab._advanced_label.text() == "Advanced: 1 changed"


def test_a_bad_value_is_refused_and_the_editor_stays_open(monkeypatch, warnings):
    w = _window(monkeypatch, ({"epochs": 1}, ""))
    w._open_advanced_settings()
    w._advanced_panel.table.item(0, 1).setText("lots")
    assert w._apply_advanced() is False
    assert "epochs" in warnings[-1]
    assert w._train_tab.config_overrides() == {}


def test_not_ready_shows_why_and_opens_nothing(monkeypatch, warnings):
    w = _window(monkeypatch, (None, "Set the Anima model files first."))
    w._open_advanced_settings()
    assert warnings[-1] == "Set the Anima model files first."
    assert w._advanced_panel is None


def test_save_as_preset_stores_the_model_and_changes(monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    from core import train_presets as tp
    w = _window(monkeypatch, ({"epochs": 1}, ""))
    store = {"v": ""}
    app = w._setup_tab.get_app_settings()
    monkeypatch.setattr(app, "get", lambda k, *a: store["v"] if k == tp.SETTINGS_KEY else "")
    monkeypatch.setattr(app, "set", lambda k, v: store.__setitem__("v", v))
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("My Anima", True))
    w._open_advanced_settings()
    w._advanced_panel.table.item(0, 1).setText("7")
    assert w._save_advanced_as_preset() is True
    p = tp.find(store["v"], "My Anima")
    assert p.model == w._train_tab.model() and p.overrides == {"epochs": 7}
    assert w._train_tab.config_overrides() == {"epochs": 7}


def test_picker_lists_only_presets_for_the_current_model(monkeypatch):
    from PySide6.QtWidgets import QListWidget
    from core import train_presets as tp
    w = _window(monkeypatch, ({"epochs": 1}, ""))
    store = tp.add_custom("", tp.TrainPreset("Krea look", model="krea2"))
    monkeypatch.setattr(w, "_presets_store", lambda: store)
    opened = []
    from ui.forge_modal import ForgeModal
    monkeypatch.setattr(ForgeModal, "open", lambda self: opened.append(self))
    w._open_preset_picker()
    names = [opened[-1].findChild(QListWidget).item(i).data(256)
             for i in range(opened[-1].findChild(QListWidget).count())]
    assert "Krea look" not in names and "Person" in names
