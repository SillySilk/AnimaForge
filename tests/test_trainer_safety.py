import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QProcess

QCoreApplication.instance() or QCoreApplication([])

from lora_fixture import DEAD_B, write_lora

from core.trainer import TrainingProcess


class FakeQProcess:
    """Stands in for the QProcess: terminate() ends the run the way Qt does,
    by delivering finished() synchronously from waitForFinished()."""

    def __init__(self, owner):
        self.owner = owner
        self.running = True
        self.terminated = 0

    def state(self):
        return QProcess.Running if self.running else QProcess.NotRunning

    def terminate(self):
        self.terminated += 1

    def waitForFinished(self, ms):
        if self.running:
            self.running = False
            self.owner._on_finished(1, None)
        return True

    def kill(self):
        pass


def _epoch_lines(losses, total=10):
    lines = []
    for i, loss in enumerate(losses, start=1):
        lines.append(f"epoch {i}/{total}")
        lines.append(f"steps:  10%|█  | {i * 12}/120 [00:30<04:30,  2.50s/it, avr_loss={loss}]")
    return lines


def _proc(tmp_path, epochs=(1, 2, 3, 4, 5, 6), keep_last=0, final_b=0.04):
    for e in epochs:
        write_lora(tmp_path / f"X-{e:06d}.safetensors", prefix="lora_unet")
    write_lora(tmp_path / "X.safetensors", prefix="lora_unet", b_value=final_b)
    t = TrainingProcess()
    t.set_safety(str(tmp_path), "X", keep_last)
    t._process = FakeQProcess(t)
    t._arm_safety()
    seen = {"finished": [], "notes": [], "log": []}
    t.training_finished.connect(seen["finished"].append)
    t.safety_note.connect(seen["notes"].append)
    t.log_line.connect(seen["log"].append)
    return t, seen


def test_a_blowup_stops_the_run_and_ships_the_last_good_epoch(tmp_path):
    t, seen = _proc(tmp_path)
    fake = t._process
    for line in _epoch_lines([0.11, 0.10, 0.10, 0.10, 0.25, 0.10]):
        t._handle_line(line)
    assert fake.terminated == 1
    assert seen["finished"] == [True]
    assert (tmp_path / "X.overshot.safetensors").is_file()
    assert not (tmp_path / "X-000004.safetensors").exists()      # shipped as X.safetensors
    assert "epoch 4" in seen["notes"][0]


def test_an_overshoot_at_the_end_ships_the_best_epoch(tmp_path):
    t, seen = _proc(tmp_path, epochs=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    for line in _epoch_lines([0.11, 0.10, 0.09, 0.08, 0.10, 0.10, 0.11, 0.11, 0.11, 0.12]):
        t._handle_line(line)
    t._process.running = False
    t._on_finished(0, None)
    assert seen["finished"] == [True]
    assert "epoch 4" in seen["notes"][0]
    assert (tmp_path / "X.overshot.safetensors").is_file()


def test_a_dead_final_lora_fails_the_run(tmp_path):
    t, seen = _proc(tmp_path, final_b=DEAD_B)
    for line in _epoch_lines([0.11, 0.10, 0.10]):
        t._handle_line(line)
    t._process.running = False
    t._on_finished(0, None)
    assert seen["finished"] == [False]
    assert any("did not learn" in l for l in seen["log"])


def test_a_healthy_run_applies_keep_last_and_says_nothing(tmp_path):
    t, seen = _proc(tmp_path, keep_last=2)
    for line in _epoch_lines([0.11, 0.10, 0.10, 0.09, 0.09, 0.09]):
        t._handle_line(line)
    t._process.running = False
    t._on_finished(0, None)
    assert seen["finished"] == [True] and seen["notes"] == []
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "X-000005.safetensors", "X-000006.safetensors", "X.safetensors"]


def test_a_user_stop_is_not_judged(tmp_path):
    t, seen = _proc(tmp_path, keep_last=2)
    for line in _epoch_lines([0.11, 0.10]):
        t._handle_line(line)
    t.stop()
    assert seen["finished"] == [False] and seen["notes"] == []
    assert len(list(tmp_path.iterdir())) == 7                     # nothing pruned or moved


def test_without_safety_nothing_changes(tmp_path):
    write_lora(tmp_path / "X.safetensors", prefix="lora_unet", b_value=DEAD_B)
    t = TrainingProcess()
    finished = []
    t.training_finished.connect(finished.append)
    for line in _epoch_lines([0.11, 0.10, 0.10, 0.10, 0.25, 0.10]):
        t._handle_line(line)
    t._on_finished(0, None)
    assert finished == [True]


def test_safety_is_for_one_run_only(tmp_path):
    t, seen = _proc(tmp_path)
    t._process.running = False
    t._on_finished(0, None)
    assert t._safety is None
