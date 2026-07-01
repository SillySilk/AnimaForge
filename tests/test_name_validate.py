import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from ui.name_validate_view import NameValidateView
from core import characters as ch

_app = QApplication.instance() or QApplication([])


def _seed(tmp_path):
    # one conforming file fixes the project category, two stragglers to fix
    for n in ["Bart_001_Character.png", "Lisa.png", "scene.png"]:
        (tmp_path / n).write_bytes(b"x")


def test_header_counts_and_invalid_boxes(tmp_path):
    _seed(tmp_path)
    v = NameValidateView(str(tmp_path))
    assert "Character" in v._header.text()
    assert "Lisa.png" in v._invalid_boxes and "scene.png" in v._invalid_boxes
    # box pre-fills with the current name for manual editing
    assert v._invalid_boxes["Lisa.png"].text() == "Lisa.png"


def test_auto_format_renames_on_disk(tmp_path):
    _seed(tmp_path)
    v = NameValidateView(str(tmp_path))
    v._cat_combo.setCurrentText("Character")
    v._auto_format()
    # Lisa derives a NAME -> conforms; scene has no derivable subject change but
    # still gets serial+category since its single field is the NAME
    assert (tmp_path / "Lisa_001_Character.png").is_file()
    assert not (tmp_path / "Lisa.png").exists()


def test_manual_rename_applies_on_disk(tmp_path):
    _seed(tmp_path)
    v = NameValidateView(str(tmp_path))
    v._invalid_boxes["scene.png"].setText("Scene_002_Character.png")
    v._do_rename("scene.png")
    assert (tmp_path / "Scene_002_Character.png").is_file()
    assert not (tmp_path / "scene.png").exists()


def test_done_writes_characters(tmp_path):
    _seed(tmp_path)
    v = NameValidateView(str(tmp_path))
    v._done()
    data = ch.load(str(tmp_path))
    assert "Bart" in [c.token for c in data.roster]
