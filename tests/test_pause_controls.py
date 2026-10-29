import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil
from PySide6.QtWidgets import QApplication

from core.proc_pause import set_tree_suspended
from core.trainer import TrainingProcess
from ui.home_tab import HomeTab

_app = QApplication.instance() or QApplication([])


def test_suspend_and_resume_a_process_tree():
    # A parent that spawns a sleeping child: both must stop and both must wake.
    code = ("import subprocess,sys,time;"
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
            "time.sleep(60)")
    p = subprocess.Popen([sys.executable, "-c", code])
    try:
        root = psutil.Process(p.pid)
        for _ in range(50):
            if root.children():
                break
            time.sleep(0.1)
        tree = [root] + root.children(recursive=True)
        assert len(tree) >= 2
        assert set_tree_suspended(p.pid, True)
        assert all(t.status() == psutil.STATUS_STOPPED for t in tree)
        assert set_tree_suspended(p.pid, False)
        assert all(t.status() != psutil.STATUS_STOPPED for t in tree)
    finally:
        for t in psutil.Process(p.pid).children(recursive=True):
            t.kill()
        p.kill()


def test_suspend_unknown_pid_is_false():
    assert set_tree_suspended(0, True) is False


def test_trainer_pause_is_noop_when_idle():
    tp = TrainingProcess()
    assert tp.pause() is False
    assert tp.resume() is False
    assert tp.is_paused() is False


def test_home_pause_button_follows_run_state():
    t = HomeTab()
    assert not t._pause_train_btn.isEnabled()
    t.set_training_active(True)
    assert t._pause_train_btn.isEnabled()
    t.set_training_paused(True)
    assert "Resume" in t._pause_train_btn.text()
    t.set_training_active(False)   # a run ending clears the paused look
    assert not t._pause_train_btn.isEnabled()
    assert "Pause" in t._pause_train_btn.text()


def test_home_caption_stop_follows_caption_state():
    t = HomeTab()
    stops = []
    t.stop_caption_requested.connect(lambda: stops.append(1))
    assert not t._stop_caption_btn.isEnabled()
    t.set_captioning_active(True)
    assert t._stop_caption_btn.isEnabled() and not t._run_caption_btn.isEnabled()
    t._stop_caption_btn.click()
    assert stops == [1]
    t.set_captioning_active(False)
    assert not t._stop_caption_btn.isEnabled() and t._run_caption_btn.isEnabled()
