import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.env import _choose_python, _to_console_python, subprocess_python


def test_prefers_running_venv():
    # When the GUI runs inside a venv, always use that interpreter.
    assert _choose_python(True, "C:/sd/venv/Scripts/python.exe", True,
                          "C:/proj/.venv/Scripts/python.exe") == "C:/proj/.venv/Scripts/python.exe"


def test_legacy_used_when_not_in_venv():
    # Old two-venv install, GUI not in a venv → honor the sd-scripts venv.
    assert _choose_python(False, "C:/sd/venv/Scripts/python.exe", True,
                          "C:/sys/python.exe") == "C:/sd/venv/Scripts/python.exe"


def test_fallback_to_sys_exe_when_no_legacy():
    assert _choose_python(False, "", False, "C:/sys/python.exe") == "C:/sys/python.exe"


def test_legacy_missing_falls_back():
    assert _choose_python(False, "C:/sd/venv/Scripts/python.exe", False,
                          "C:/sys/python.exe") == "C:/sys/python.exe"


def test_subprocess_python_returns_a_path():
    # Smoke: with no sdscripts path it returns the current interpreter (tests run under
    # the console python.exe, so the pythonw->python mapping is a no-op here).
    assert subprocess_python() == sys.executable


def test_console_python_maps_pythonw_to_python():
    # The windowless GUI interpreter must be swapped for the console one so a grandchild
    # process's stdout reaches the QProcess pipe (otherwise the UI freezes on "Preparing").
    assert Path(_to_console_python("C:/proj/.venv/Scripts/pythonw.exe")).name == "python.exe"


def test_console_python_is_case_insensitive():
    assert Path(_to_console_python("C:/proj/.venv/Scripts/Pythonw.EXE")).name.lower() == "python.exe"


def test_console_python_leaves_python_untouched():
    assert _to_console_python("C:/proj/.venv/Scripts/python.exe") == "C:/proj/.venv/Scripts/python.exe"


def test_console_python_leaves_other_names_untouched():
    assert _to_console_python("C:/sys/python3.10.exe") == "C:/sys/python3.10.exe"
