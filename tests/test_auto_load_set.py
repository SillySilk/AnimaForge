import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

_app = QApplication.instance() or QApplication([])

from core import sets
from core.batch import RunDefinition


@pytest.fixture
def win(monkeypatch, tmp_path):
    from ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    w = MainWindow()
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "a.png").write_bytes(b"x")
    saved = RunDefinition(lora_name="Saved Name", dataset_folder=str(ds), image_count=1,
                          trigger_word="svd", network_dim=48, network_alpha=24,
                          target_steps=1234)
    calls = []

    def fake_find(folder, root=None):
        calls.append(folder)
        return ("Saved Name", saved)
    monkeypatch.setattr(sets, "find_set_for_folder", fake_find)
    yield w, str(ds), calls
    w.close()


def test_loading_a_dataset_restores_its_saved_set(win):
    w, ds, calls = win
    w._on_dataset_loaded(ds, 1)
    assert calls == [ds]   # looked up once; the restore's own reload doesn't re-trigger it
    assert w._train_tab.get_lora_name() == "Saved Name"
    assert w._train_tab._dim_spin.value() == 48
    assert w._train_tab._target_steps_spin.value() == 1234
    assert w._dataset_tab.get_trigger_word() == "svd"
    assert w._home_tab._loaded_set == "Saved Name"


def test_hand_picked_set_is_not_swapped_for_the_folder_newest(win):
    w, ds, calls = win
    w._on_load_set_requested(ds, "picked", "")
    assert calls == []


def test_no_restore_while_training(win, monkeypatch):
    w, ds, calls = win
    monkeypatch.setattr(w._train_tab, "is_training", lambda: True)
    w._on_dataset_loaded(ds, 1)
    assert calls == []
