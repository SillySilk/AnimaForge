from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

import core.batch as batch_mod
from core import caption_policy as cp
from core.batch import DONE, FAILED, BatchRunner, RunDefinition

app = QApplication.instance() or QApplication([])


class FakeSession(QObject):
    progress = Signal(dict)
    failed = Signal(str)
    done = Signal(str)
    canceled = Signal()
    state_changed = Signal(str)

    def __init__(self, parent=None, refuse=None):
        super().__init__(parent)
        self.started_with = None
        self.cancels = 0
        self._running = False
        self._refuse = refuse

    def start(self, params):
        self.started_with = params
        if self._refuse:
            self.failed.emit(self._refuse)   # Krea's session fails synchronously on preflight
            return False
        self._running = True
        return True

    def cancel(self):
        self.cancels += 1
        self._running = False

    def is_running(self):
        return self._running


def make(tmp_path, n=1, **kw):
    runs = []
    for i in range(n):
        ds = tmp_path / f"ds{i}"
        ds.mkdir()
        (ds / "a.png").write_bytes(b"x")
        (ds / "a.txt").write_text("amberh, a woman")
        runs.append(RunDefinition(f"K{i}", str(ds), 1, model="krea2", onetrainer_root="C:/OT",
                                  caption_policy=cp.KEEP, **kw))
    return runs


def runner_with(sessions):
    made = iter(sessions)
    return BatchRunner(onetrainer_factory=lambda parent: next(made))


def test_krea2_run_trains_through_onetrainer_not_kohya(tmp_path, monkeypatch):
    s = FakeSession()
    r = runner_with([s])
    kohya = []
    monkeypatch.setattr(r._trainer, "start", lambda cfg, sd: kohya.append(sd))
    phases = []
    r.run_phase.connect(lambda i, ph: phases.append(ph))
    r.start(make(tmp_path))
    assert kohya == []
    assert s.started_with.lora_name == "K0" and s.started_with.onetrainer_root == "C:/OT"
    assert phases == [batch_mod.TRAINING]


def test_progress_and_success_reach_the_batch_signals(tmp_path):
    s = FakeSession()
    r = runner_with([s])
    steps, finished = [], []
    r.progress_updated.connect(lambda i, n: steps.append(n))
    r.run_finished.connect(lambda i, ok: finished.append(ok))
    runs = make(tmp_path)
    r.start(runs)
    s.progress.emit({"global_step": 40, "global_total": 1000, "loss": 0.1})
    s.done.emit(str(tmp_path / "K0.safetensors"))
    assert steps == [40] and finished == [True]
    assert runs[0].status == DONE and not r.is_running()


def test_failure_marks_run_failed_and_continues(tmp_path):
    a, b = FakeSession(), FakeSession()
    r = runner_with([a, b])
    runs = make(tmp_path, n=2)
    r.start(runs)
    a.failed.emit("CUDA ran out of memory.")
    assert runs[0].status == FAILED
    assert b.started_with.lora_name == "K1"


def test_onetrainer_preflight_failure_fails_only_that_run(tmp_path):
    a = FakeSession(refuse="OneTrainer folder not found")
    b = FakeSession()
    r = runner_with([a, b])
    logs = []
    r.log_line.connect(logs.append)
    runs = make(tmp_path, n=2)
    r.start(runs)
    assert runs[0].status == FAILED
    assert any("OneTrainer folder not found" in line for line in logs)
    assert b.started_with.lora_name == "K1"


def test_canceled_session_counts_as_failed(tmp_path):
    s = FakeSession()
    r = runner_with([s])
    runs = make(tmp_path)
    r.start(runs)
    s.canceled.emit()
    assert runs[0].status == FAILED


def test_stop_during_onetrainer_run_cancels_and_does_not_advance(tmp_path):
    a, b = FakeSession(), FakeSession()
    r = runner_with([a, b])
    runs = make(tmp_path, n=2)
    r.start(runs)
    r.stop()
    a.canceled.emit()        # the real session reports its cancel afterwards
    assert a.cancels == 1
    assert b.started_with is None
    assert not r.is_running()


def test_unknown_model_fails_only_that_run(tmp_path):
    good = FakeSession()
    r = runner_with([good])
    runs = make(tmp_path, n=2)
    runs[0].model = "flux9"
    r.start(runs)
    assert runs[0].status == FAILED
    assert good.started_with.lora_name == "K1"


def test_anima_runs_still_use_kohya(tmp_path, monkeypatch):
    def no_onetrainer(parent):
        raise AssertionError("an Anima run must not build a OneTrainer session")

    r = BatchRunner(onetrainer_factory=no_onetrainer)
    kohya = []
    monkeypatch.setattr(batch_mod, "generate_configs", lambda **kw: ("cfg.toml", None))
    monkeypatch.setattr(r._trainer, "start", lambda cfg, sd: kohya.append(sd))
    run = make(tmp_path)[0]
    run.model, run.sdscripts_path = "anima", "C:/sd"
    r.start([run])
    assert kohya == ["C:/sd"]
