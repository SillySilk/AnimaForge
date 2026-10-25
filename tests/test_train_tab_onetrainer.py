import os
import pytest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    """A modal box in an offscreen test waits forever; record and answer it instead."""
    from PySide6.QtWidgets import QMessageBox
    shown = []

    def record(answer):
        return lambda *a, **k: shown.append(a[2] if len(a) > 2 else "") or answer
    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, name, record(QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "question", record(QMessageBox.Yes))
    return shown


class FakeSession(QObject):
    state_changed = Signal(str)
    progress = Signal(dict)
    failed = Signal(str)
    done = Signal(str)
    canceled = Signal()

    def __init__(self, parent=None, refuse=""):
        super().__init__(parent)
        self.params, self.cancels, self._running, self._refuse = None, 0, False, refuse
        self._run_dir = ""

    def start(self, params):
        self.params = params
        if self._refuse:
            self.failed.emit(self._refuse)
            return False
        self._running = True
        self.state_changed.emit("training")
        return True

    def cancel(self):
        self.cancels += 1

    def is_running(self):
        return self._running

    def samples_dir(self):
        return os.path.join(self._run_dir, "workspace", "samples")

    def run_dir(self):
        return self._run_dir

    def notes(self):
        return ["The end of the run overshot; shipped epoch 30 instead."]

    def finish(self, path):
        self._running = False
        self.done.emit(path)


def _krea_tab(tmp_path, monkeypatch, trigger="mrx", session=None):
    from core.engines.onetrainer import engine
    from ui.train_tab import TrainTab
    monkeypatch.setattr(engine, "validate", lambda root: (True, ""))
    t = TrainTab()
    ds = tmp_path / "ds"
    ds.mkdir()
    for i in range(4):
        (ds / f"a{i}.png").write_bytes(b"x")
        (ds / f"a{i}.txt").write_text("mrx, a man", encoding="utf-8")
    t._output_dir = str(tmp_path / "out")
    t.set_dataset(str(ds), 4)
    t.set_lora_name("Mr X")
    t.set_trigger_word(trigger)
    t.set_model("krea2")
    t.set_onetrainer_root("D:/OT")
    s = session or FakeSession()
    t._onetrainer_factory = lambda parent: s
    t.set_gpu_busy_check(lambda: "")
    return t, s


def test_start_hands_the_run_to_onetrainer(tmp_path, monkeypatch):
    t, s = _krea_tab(tmp_path, monkeypatch)
    kohya = []
    monkeypatch.setattr(t._trainer, "start", lambda *a: kohya.append(a))
    t._start_training(confirm=False)
    assert kohya == [] and s.params.lora_name == "Mr X" and s.params.onetrainer_root == "D:/OT"
    assert t.is_training()


def test_progress_drives_the_tick_bar_and_home(tmp_path, monkeypatch):
    t, s = _krea_tab(tmp_path, monkeypatch)
    payloads = []
    t.run_progress.connect(payloads.append)
    t._start_training(confirm=False)
    s.progress.emit({"epoch": 2, "epoch_total": 10, "step": 3, "step_total": 9,
                     "loss": 0.12, "global_step": 21, "global_total": 90})
    assert {"kind": "progress", "step": 21, "total": 90} in payloads


def test_done_finishes_once_and_remembers_the_shipped_file(tmp_path, monkeypatch):
    t, s = _krea_tab(tmp_path, monkeypatch)
    ends = []
    t.training_active.connect(ends.append)
    t._start_training(confirm=False)
    s.finish(str(tmp_path / "out" / "Mr X" / "Mr_X.safetensors"))
    assert ends == [True, False]
    assert str(t._lora_output_path()).endswith("Mr_X.safetensors")
    assert "overshot" in t._log_edit.toPlainText()


def test_stop_cancels_the_session_and_finishes_once(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    t, s = _krea_tab(tmp_path, monkeypatch)
    ends = []
    t.training_active.connect(ends.append)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    t._start_training(confirm=False)
    t._stop_training()
    s._running = False
    s.canceled.emit()
    assert s.cancels == 1 and ends == [True, False]


def test_preflight_refusal_is_shown_and_ends_the_run(tmp_path, monkeypatch):
    t, s = _krea_tab(tmp_path, monkeypatch, session=FakeSession(refuse="No captions mention mrx."))
    ends = []
    t.training_active.connect(ends.append)
    t._start_training(confirm=False)
    assert "No captions mention mrx." in t._log_edit.toPlainText()
    assert ends == [True, False] and not t.is_training()


def test_krea_run_without_trigger_uses_the_lora_name(tmp_path, monkeypatch):
    t, s = _krea_tab(tmp_path, monkeypatch, trigger="")
    t._start_training(confirm=False)
    assert s.params.trigger_word == "Mr X"


def test_finished_krea_run_writes_no_resume_manifest(tmp_path, monkeypatch):
    from core import run_manifest
    t, s = _krea_tab(tmp_path, monkeypatch)
    writes = []
    monkeypatch.setattr(run_manifest, "write_start", lambda *a, **k: writes.append("start"))
    monkeypatch.setattr(run_manifest, "mark", lambda *a, **k: writes.append("mark"))
    t._start_training(confirm=False)
    s.finish(str(tmp_path / "x.safetensors"))
    assert writes == []


def test_samples_are_read_from_the_sessions_folder(tmp_path, monkeypatch):
    t, s = _krea_tab(tmp_path, monkeypatch)
    s._run_dir = str(tmp_path / "out" / "Mr X")
    t._start_training(confirm=False)
    assert t._samples_dir() == s.samples_dir()


def test_anima_start_still_uses_kohya(tmp_path, monkeypatch):
    t, s = _krea_tab(tmp_path, monkeypatch)
    t.set_model("anima")

    def no_onetrainer(parent):
        raise AssertionError("an Anima run must not build a OneTrainer session")
    t._onetrainer_factory = no_onetrainer
    monkeypatch.setattr(t, "_validate_for_training", lambda: (False, "sd-scripts not set"))
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    t._start_training(confirm=False)
    assert shown == ["sd-scripts not set"]


def test_moss_step_spin_starts_from_its_own_default_not_animas_setting():
    from ui.train_tab import TrainTab

    class Settings:
        def get(self, key):
            return {"default_target_steps": 700}.get(key)
    t = TrainTab()
    t._app_settings = Settings()
    t.set_model("moss")
    assert t.get_target_steps() == 2500
    t.set_model("anima")
    assert t.get_target_steps() == 700
