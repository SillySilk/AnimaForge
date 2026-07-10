import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QObject, QTimer, Signal

from core.batch import RunDefinition, save_queue
from core import headless


class FakeTrainer(QObject):
    """Stands in for core.trainer.TrainingProcess: succeeds instantly."""
    progress_updated = Signal(int)
    log_line = Signal(str)
    training_finished = Signal(bool)
    training_started = Signal()

    succeed = True

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False

    def set_total_steps(self, n):
        pass

    def is_running(self):
        return self._running

    def start(self, cfg, sdscripts):
        self._running = True
        self.training_started.emit()
        # Finish asynchronously, as QProcess would.
        QTimer.singleShot(0, self._finish)

    def _finish(self):
        self._running = False
        self.progress_updated.emit(5)
        self.training_finished.emit(FakeTrainer.succeed)

    def stop(self):
        self._running = False


def _queue_file(tmp_path, n=1) -> Path:
    runs = [
        RunDefinition(lora_name=f"lr{i}", dataset_folder=str(tmp_path),
                      image_count=4, target_steps=100,
                      output_dir=str(tmp_path / "out"))
        for i in range(n)
    ]
    qf = tmp_path / "batch_queue.json"
    save_queue(str(qf), runs)
    return qf


def test_headless_success_exit_zero_and_status_done(tmp_path, monkeypatch=None):
    import core.batch as batch_mod
    orig = batch_mod.TrainingProcess
    batch_mod.TrainingProcess = FakeTrainer
    try:
        FakeTrainer.succeed = True
        qf = _queue_file(tmp_path)
        sf = tmp_path / "batch_status.json"
        code = headless.run_headless(str(qf), str(sf))
        assert code == 0
        status = json.loads(sf.read_text(encoding="utf-8"))
        assert status[0]["state"] == "done"
        # queue file rewritten so the GUI sees the result later
        queue = json.loads(qf.read_text(encoding="utf-8"))
        assert queue[0]["status"] == "done"
    finally:
        batch_mod.TrainingProcess = orig


def test_headless_failure_exit_one(tmp_path):
    import core.batch as batch_mod
    orig = batch_mod.TrainingProcess
    batch_mod.TrainingProcess = FakeTrainer
    try:
        FakeTrainer.succeed = False
        qf = _queue_file(tmp_path)
        sf = tmp_path / "batch_status.json"
        code = headless.run_headless(str(qf), str(sf))
        assert code == 1
        status = json.loads(sf.read_text(encoding="utf-8"))
        assert status[0]["state"] == "failed"
    finally:
        batch_mod.TrainingProcess = orig


def test_headless_missing_queue_exits_one_cleanly(tmp_path):
    code = headless.run_headless(str(tmp_path / "nope.json"),
                                 str(tmp_path / "s.json"))
    assert code == 1


def test_headless_degrades_ask_policy_to_keep_before_runner_starts(tmp_path, monkeypatch):
    """'ask' is a UI-only prompt with no GUI here to answer it. Headless must rewrite
    every 'ask' run to 'keep' BEFORE the runner starts, so a queued 'ask' never blocks
    and never destroys captions already on disk. A spy on BatchRunner.start captures the
    policies at the exact moment the runner is invoked."""
    import core.batch as batch_mod

    seen = {}

    def _spy_start(self, runs, *args, **kwargs):
        seen["policies"] = [r.caption_policy for r in runs]
        self.batch_finished.emit()          # end synchronously; run nothing

    monkeypatch.setattr(batch_mod.BatchRunner, "start", _spy_start)

    runs = [
        RunDefinition(lora_name="lr0", dataset_folder=str(tmp_path), image_count=4,
                      target_steps=100, output_dir=str(tmp_path / "out"),
                      caption_policy="ask"),
        RunDefinition(lora_name="lr1", dataset_folder=str(tmp_path), image_count=4,
                      target_steps=100, output_dir=str(tmp_path / "out"),
                      caption_policy="ask"),
    ]
    qf = tmp_path / "batch_queue.json"
    save_queue(str(qf), runs)

    headless.run_headless(str(qf), str(tmp_path / "s.json"))

    assert seen["policies"] == ["keep", "keep"]
    # The rewrite is persisted back to the queue file too.
    queue = json.loads(qf.read_text(encoding="utf-8"))
    assert [r["caption_policy"] for r in queue] == ["keep", "keep"]
