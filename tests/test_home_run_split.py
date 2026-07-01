import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from ui.home_tab import HomeTab

_app = QApplication.instance() or QApplication([])


def test_has_split_run_buttons_and_signals():
    t = HomeTab()
    # Caption/train panels are stashed and shown in the Options/Presets modals; the pillars
    # carry the primary Run Captioning / Start Training buttons.
    assert hasattr(t, "mount_caption_controls") and hasattr(t, "_open_caption_modal")
    assert hasattr(t, "mount_train_controls") and hasattr(t, "_open_train_modal")
    assert hasattr(t, "run_caption_requested") and hasattr(t, "start_train_requested")


def test_autoset_type_from_filenames(tmp_path):
    # The subject combo now lives in the relocated Step Calculator (Train); Home's auto-detect
    # emits the type_changed intent that MainWindow forwards to Train.set_subject_type.
    for n in ["Tim Burton_001_Style.png", "Other_002_Style.png"]:
        (tmp_path / n).write_bytes(b"x")
    t = HomeTab()
    emitted = []
    t.type_changed.connect(lambda k: emitted.append(k))
    t._autoset_type(str(tmp_path))
    assert emitted == ["style"]


def test_autoset_type_object_maps_to_concept(tmp_path):
    (tmp_path / "teapot_001_Object.png").write_bytes(b"x")
    t = HomeTab()
    emitted = []
    t.type_changed.connect(lambda k: emitted.append(k))
    t._autoset_type(str(tmp_path))
    assert emitted == ["concept"]
