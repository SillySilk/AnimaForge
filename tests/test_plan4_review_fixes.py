import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import subprocess
import sys
import time

import pytest
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from lora_fixture import write_lora

from core.trainer import TrainingProcess, training_env_overrides


# C1: sd-scripts prints epoch lines to stdout; without unbuffered output they only
# arrive when the process exits, and run safety never sees an epoch close.
CHILD = (
    "import sys, time\n"
    "print('epoch 1/2')\n"
    "sys.stderr.write('READY' + chr(10)); sys.stderr.flush()\n"
    "time.sleep(4)\n"
)


def test_epoch_lines_arrive_while_the_run_is_still_going():
    env = dict(os.environ)
    env.update(training_env_overrides(existing={}, windows=sys.platform.startswith("win")))
    p = subprocess.Popen([sys.executable, "-c", CHILD], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, env=env)
    start = time.monotonic()
    try:
        seen = []
        while True:
            line = p.stdout.readline().decode().strip()
            seen.append(line)
            if line == "epoch 1/2" or not line:
                break
        assert "epoch 1/2" in seen
        assert time.monotonic() - start < 3
    finally:
        p.kill()


def test_a_line_split_across_chunks_is_read_whole():
    t = TrainingProcess()
    got = []
    t.log_line.connect(got.append)
    t._feed_text("steps: 1/10 [00:01<00:09, avr_loss=0.1]\nepoch 5/")
    t._feed_text("10\nnext")
    t._flush_partial()
    assert "epoch 5/10" in got and "next" in got and "epoch 5/" not in got


# I1: a run folder re-used by an earlier run keeps that run's epoch saves.
def test_epoch_saves_from_an_earlier_run_are_ignored(tmp_path):
    from core.engines.kohya import safety
    old = write_lora(tmp_path / "X-000008.safetensors", prefix="lora_unet")
    past = time.time() - 3600
    os.utime(old, (past, past))
    write_lora(tmp_path / "X-000006.safetensors", prefix="lora_unet")
    since = time.time() - 60
    assert [s.epoch for s in safety.epoch_saves(str(tmp_path), "X", since=since)] == [6]
    safety.prune(str(tmp_path), "X", keep_last=1, since=since)
    assert (tmp_path / "X-000008.safetensors").exists()          # not this run's to prune


# I5: "already enough" only when there really is enough.
def test_free_memory_says_when_nothing_large_enough_can_close(monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox
    QApplication.instance() or QApplication([])
    from core import sysmem
    from ui.main_window import MainWindow
    info = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: info.append(a[2]) or QMessageBox.Ok)
    monkeypatch.setattr(sysmem, "plan_room", lambda target, **k: (10.0, []))
    w = MainWindow()
    w._free_memory()
    assert "already enough" not in info[-1] and "short" in info[-1]


def test_free_memory_never_offers_a_kohya_run():
    from core import sysmem
    rows = [(5, "python.exe", "python anima_train_network.py --config_file x", 30.0)]
    assert sysmem.closable(rows, set()) == []


# I3: a failing import never stops the app and a retry never duplicates.
def test_a_retry_after_a_partial_import_does_not_duplicate(tmp_path):
    import json
    from core import kreaforge_import as ki, sets
    root = tmp_path / "Krea Forge"
    run = root / "output" / "Klaus"
    run.mkdir(parents=True)
    (run / "config.json").write_text("{}", encoding="utf-8")
    (run / "concepts.json").write_text(json.dumps([{"path": "C:/gone"}]), encoding="utf-8")
    assert ki.import_runs(root, tmp_path / "a") == ["Klaus"]
    assert ki.import_runs(root, tmp_path / "a") == []
    assert sets.list_sets(root=tmp_path / "a") == ["Klaus"]


def test_one_bad_run_does_not_stop_the_others(tmp_path, monkeypatch):
    import json
    from core import kreaforge_import as ki
    root = tmp_path / "Krea Forge"
    for name in ("A", "B"):
        run = root / "output" / name
        run.mkdir(parents=True)
        (run / "config.json").write_text("{}", encoding="utf-8")
        (run / "concepts.json").write_text(json.dumps([{"path": f"C:/{name}"}]), encoding="utf-8")
    real = ki.sets.save_set

    def flaky(name, rd, root=None):
        if name == "A":
            raise OSError("denied")
        return real(name, rd, root=root)
    monkeypatch.setattr(ki.sets, "save_set", flaky)
    assert ki.import_runs(root, tmp_path / "a") == ["B"]


def test_imported_sets_do_not_point_at_krea_forges_output(tmp_path):
    import json
    from core import kreaforge_import as ki
    root = tmp_path / "Krea Forge"
    run = root / "output" / "K"
    run.mkdir(parents=True)
    (run / "config.json").write_text("{}", encoding="utf-8")
    (run / "concepts.json").write_text(json.dumps([{"path": "C:/k"}]), encoding="utf-8")
    ki.import_runs(root, tmp_path / "a")
    d = json.loads((tmp_path / "a" / "sets" / "K.json").read_text(encoding="utf-8"))
    assert d["output_dir"] == ""


# I6: the safety stop never runs inside the output handler.
def test_the_safety_stop_is_deferred_out_of_the_output_handler(tmp_path):
    from test_trainer_safety import _epoch_lines, _proc
    t, seen = _proc(tmp_path)
    fake = t._process
    for line in _epoch_lines([0.11, 0.10, 0.10, 0.10, 0.25, 0.10]):
        t._handle_line(line)
    assert fake.terminated == 0 and seen["finished"] == []
    QApplication.processEvents()
    assert seen["finished"] == [True]


def test_stopping_never_drops_a_process_started_meanwhile():
    t = TrainingProcess()

    class Proc:
        def __init__(self):
            self.running = True
        def state(self):
            from PySide6.QtCore import QProcess
            return QProcess.Running if self.running else QProcess.NotRunning
        def processId(self):
            return 0
        def terminate(self):
            pass
        def kill(self):
            pass
        def waitForFinished(self, ms):
            self.running = False
            t._process = newer          # the next queued run started from finished()
            return True
    newer = object()
    t._process = Proc()
    t._stop_process()
    assert t._process is newer


# I2: a safety ship reads as what it was, not as a plain success.
def test_the_train_tab_shows_the_safety_note_instead_of_plain_success():
    from ui.train_tab import TrainTab
    tab = TrainTab()
    msgs = []
    tab.status_message.connect(msgs.append)
    tab._trainer.last_safety_note = "Training blew up ... Shipped the epoch 4 snapshot instead."
    tab._on_training_finished(True)
    assert "epoch 4" in msgs[-1]
