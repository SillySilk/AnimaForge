import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from ui.main_window import _compose_lines

_app = QApplication.instance() or QApplication([])


def test_lines_with_version_and_commits():
    lines = _compose_lines({
        "remote_version": "2.6.0", "local_version": "2.5.0",
        "ahead_by": 12, "latest_subject": "fix: batch icons", "head_sha": "f" * 40})
    body = "\n".join(lines)
    assert "AnimaForge v2.6.0 is available." in body
    assert "You have v2.5.0" in body and "12 commits behind" in body
    assert "Latest: fix: batch icons" in body
    assert "untouched" in body


def test_lines_without_version_bump():
    lines = _compose_lines({
        "remote_version": None, "local_version": "2.5.0",
        "ahead_by": 3, "latest_subject": "fix: y", "head_sha": "f" * 40})
    body = "\n".join(lines)
    assert "is available" not in body          # no version line
    assert "3 commits behind" in body


def test_lines_version_only_fallback():
    lines = _compose_lines({
        "remote_version": "2.6.0", "local_version": "2.5.0",
        "ahead_by": None, "latest_subject": None, "head_sha": None})
    body = "\n".join(lines)
    assert "AnimaForge v2.6.0 is available." in body
    assert "commits behind" not in body        # ahead_by is None
    assert "Latest:" not in body
