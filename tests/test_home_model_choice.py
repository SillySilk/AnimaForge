import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)


def test_home_lists_every_model_and_style():
    from core import models
    from ui.home_tab import HomeTab
    h = HomeTab()
    ids = [h._model_combo.itemData(i) for i in range(h._model_combo.count())]
    styles = [h._style_combo.itemData(i) for i in range(h._style_combo.count())]
    assert ids == list(models.MODELS) and styles == list(models.CAPTION_STYLES)


def test_choosing_a_model_on_home_emits_it():
    from ui.home_tab import HomeTab
    h = HomeTab()
    seen = []
    h.model_changed.connect(seen.append)
    h._model_combo.setCurrentIndex(h._model_combo.findData("krea2"))
    assert seen == ["krea2"]


def test_programmatic_set_does_not_re_emit():
    from ui.home_tab import HomeTab
    h = HomeTab()
    seen = []
    h.model_changed.connect(seen.append)
    h.caption_style_changed.connect(seen.append)
    h.set_model("krea2")
    h.set_caption_style("tags")
    assert seen == [] and h._model_combo.currentData() == "krea2"
    assert h._style_combo.currentData() == "tags"


def test_main_window_keeps_home_train_and_dataset_in_step(monkeypatch):
    from core.settings import AppSettings
    store = {}
    monkeypatch.setattr(AppSettings, "set", lambda self, k, v: store.__setitem__(k, v))
    from ui.main_window import MainWindow
    w = MainWindow()
    h = w._home_tab
    h._model_combo.setCurrentIndex(h._model_combo.findData("krea2"))
    assert w._train_tab.model() == "krea2"
    assert w._dataset_tab._caption_style_now() == w._train_tab.caption_style() == "prose+tags"
    assert store.get("model") == "krea2"
    h._style_combo.setCurrentIndex(h._style_combo.findData("prose"))
    assert w._train_tab.caption_style() == "prose" and w._dataset_tab._caption_style_now() == "prose"
    from core.batch import RunDefinition
    w._train_tab.apply_run_definition(RunDefinition("A", "", 0, model="anima"))
    assert h._model_combo.currentData() == "anima"
