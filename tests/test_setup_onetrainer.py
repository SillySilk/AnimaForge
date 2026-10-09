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


def _setup(monkeypatch, store):
    from core.settings import AppSettings, DEFAULTS
    monkeypatch.setattr(AppSettings, "get", lambda self, k: store.get(k, DEFAULTS.get(k)))
    monkeypatch.setattr(AppSettings, "set", lambda self, k, v: store.__setitem__(k, v))
    from ui.setup_tab import SetupTab
    return SetupTab()


def test_onetrainer_root_round_trips_through_settings(monkeypatch):
    store = {"onetrainer_root": "D:/OT"}
    s = _setup(monkeypatch, store)
    assert s.get_onetrainer_root() == "D:/OT"
    s._ot_edit.setText("E:/OneTrainer")
    assert store["onetrainer_root"] == "E:/OneTrainer" and s.get_onetrainer_root() == "E:/OneTrainer"


def test_status_dot_reports_engine_validation(monkeypatch):
    from core.engines.onetrainer import engine
    monkeypatch.setattr(engine, "validate", lambda root: (False, "scripts\\train.py is missing."))
    s = _setup(monkeypatch, {"onetrainer_root": "D:/OT"})
    s._revalidate_onetrainer()
    assert "train.py" in s._ot_dot.toolTip()


def test_krea_weights_line_names_the_licence_when_missing(monkeypatch):
    from core import describe_models
    monkeypatch.setattr(describe_models, "has_model_snapshot", lambda cache, repo: False)
    s = _setup(monkeypatch, {})
    s._refresh_krea_weights()
    assert "huggingface.co/krea/Krea-2-Raw" in s._krea_weights_label.text()


def test_krea_weights_line_says_downloaded_when_present(monkeypatch):
    from core import describe_models
    monkeypatch.setattr(describe_models, "has_model_snapshot", lambda cache, repo: True)
    s = _setup(monkeypatch, {})
    s._refresh_krea_weights()
    assert "downloaded" in s._krea_weights_label.text()
