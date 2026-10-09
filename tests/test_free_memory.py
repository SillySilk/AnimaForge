import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

_app = QApplication.instance() or QApplication([])

from core import sysmem

ROWS = [(11, "chrome.exe (PID 11)", 9.0), (12, "forge (PID 12)", 20.0), (13, "slack (PID 13)", 3.0)]


def test_plan_room_closes_biggest_first_until_the_target():
    have, plan = sysmem.plan_room(40, free=lambda: 15.0, candidates=lambda: sorted(ROWS, key=lambda r: -r[2]))
    assert have == 15.0
    assert [pid for pid, *_ in plan] == [12, 11]


def test_plan_room_with_room_or_no_probe_closes_nothing():
    assert sysmem.plan_room(10, free=lambda: 15.0, candidates=lambda: ROWS) == (15.0, [])
    assert sysmem.plan_room(10, free=lambda: None, candidates=lambda: ROWS) == (None, [])


def test_room_target_per_engine():
    from core import models, run_prep
    from core.engines.onetrainer import engine
    assert run_prep.room_target_gb(models.ONETRAINER) == engine.MIN_FREE_COMMIT_GB
    assert run_prep.room_target_gb(models.KOHYA) == run_prep.ANIMA_MIN_FREE_COMMIT_GB


def test_home_button_asks_and_locks_while_training():
    from ui.home_tab import HomeTab
    h = HomeTab()
    seen = []
    h.free_memory_requested.connect(lambda: seen.append(1))
    h._free_memory_btn.click()
    assert seen == [1]
    h.set_training_active(True)
    assert not h._free_memory_btn.isEnabled()


@pytest.fixture
def window(monkeypatch):
    from ui.main_window import MainWindow
    calls = {"info": [], "warn": [], "closed": []}
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: calls["info"].append(a[2]) or QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(sysmem, "close_tree", lambda pid: calls["closed"].append(pid) or True)
    w = MainWindow()
    return w, calls, monkeypatch


def test_nothing_to_close_only_informs(window):
    w, calls, mp = window
    mp.setattr(sysmem, "plan_room", lambda target, **k: (80.0, []))
    w._free_memory()
    assert calls["closed"] == [] and "already enough" in calls["info"][-1]


def test_saying_no_closes_nothing(window):
    w, calls, mp = window
    mp.setattr(sysmem, "plan_room", lambda target, **k: (15.0, ROWS[:2]))
    mp.setattr(QMessageBox, "warning", lambda *a, **k: calls["warn"].append(a[2]) or QMessageBox.No)
    w._free_memory()
    assert calls["closed"] == []
    assert "chrome.exe" in calls["warn"][-1] and "forge" in calls["warn"][-1]
    assert "unsaved" in calls["warn"][-1]


def test_saying_yes_closes_each_program_once(window):
    w, calls, mp = window
    mp.setattr(sysmem, "plan_room", lambda target, **k: (15.0, ROWS[:2]))
    mp.setattr(sysmem, "available_commit_gb", lambda: 44.0)
    mp.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.Yes)
    w._free_memory()
    assert calls["closed"] == [11, 12]


def test_profiler_switch_reaches_a_krea_run(tmp_path, monkeypatch):
    from core.batch import RunDefinition
    from core.engines.onetrainer.adapter import to_onetrainer_params
    rd = RunDefinition(lora_name="a", dataset_folder="d", image_count=1, model="krea2",
                       debug_profiles=True)
    assert to_onetrainer_params(rd).debug_profiles is True
    assert RunDefinition.from_dict(rd.to_dict()).debug_profiles is True
