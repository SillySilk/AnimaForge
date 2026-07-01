import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication, QGroupBox
from ui.characters_tab import CharactersTab
from core import characters as C

_app = QApplication.instance() or QApplication([])


def _make_images(folder: Path, names):
    from PIL import Image
    for n in names:
        Image.new("RGB", (8, 8), (90, 90, 90)).save(folder / n)


def _bundle_titles(tab):
    from PySide6.QtWidgets import QFrame, QLabel
    titles = []
    for i in range(tab._bundles_layout.count()):
        w = tab._bundles_layout.itemAt(i).widget()
        if isinstance(w, QFrame):
            # the name label carries af_display_gold (solo/ensemble) or ready_row_err (warn)
            for lbl in w.findChildren(QLabel):
                if lbl.objectName() in ("af_display_gold", "ready_row_err"):
                    titles.append(lbl.text())
                    break
    return titles


def test_set_dataset_renders_solo_and_combined_bundles(tmp_path: Path):
    _make_images(tmp_path, ["Homer_001_Character.png", "Homer_002_Character.png",
                            "Homer-Marge_003_Character.png", "Lisa_001_Character.png"])
    tab = CharactersTab()
    tab.set_dataset(str(tmp_path))
    titles = _bundle_titles(tab)
    assert "Homer" in titles
    assert "Lisa" in titles
    assert "Homer + Marge" in titles
    assert "Marge" not in titles       # multi-subject only


def test_set_dataset_persists_names_into_prompt_roster(tmp_path: Path):
    _make_images(tmp_path, ["Homer-Marge_001_Character.png", "Lisa_002_Character.png"])
    tab = CharactersTab()
    tab.set_dataset(str(tmp_path))
    data = C.load(str(tmp_path))
    assert [c.token for c in data.roster] == ["Homer", "Marge", "Lisa"]
    assert data.assignments["Homer-Marge_001_Character.png"]["present"] == ["Homer", "Marge"]


def test_set_dataset_preserves_existing_style_anchor(tmp_path: Path):
    _make_images(tmp_path, ["Homer_001_Character.png"])
    C.save(str(tmp_path), C.DatasetCharacters(style_anchor="@mystyle"))
    tab = CharactersTab()
    tab.set_dataset(str(tmp_path))
    assert tab._style_anchor_edit.text() == "@mystyle"
    assert C.load(str(tmp_path)).style_anchor == "@mystyle"


def test_needs_naming_bundle_only_when_nonconforming(tmp_path: Path):
    _make_images(tmp_path, ["Homer_001_Character.png", "scene.png"])
    tab = CharactersTab()
    tab.set_dataset(str(tmp_path))
    assert any("Needs naming" in t for t in _bundle_titles(tab))

    clean = tmp_path / "clean"
    clean.mkdir()
    _make_images(clean, ["Bart_001_Character.png"])
    tab.set_dataset(str(clean))
    assert not any("Needs naming" in t for t in _bundle_titles(tab))


def test_style_anchor_edit_persists_and_emits(tmp_path: Path):
    _make_images(tmp_path, ["Homer_001_Character.png"])
    tab = CharactersTab()
    tab.set_dataset(str(tmp_path))
    fired = []
    tab.characters_changed.connect(lambda: fired.append(1))
    tab._style_anchor_edit.setText("@neon")
    assert C.load(str(tmp_path)).style_anchor == "@neon"
    assert fired


def test_auto_detect_from_filenames_returns_subject_count(tmp_path: Path):
    _make_images(tmp_path, ["Homer-Marge_001_Character.png", "Lisa_002_Character.png"])
    tab = CharactersTab()
    tab.set_dataset(str(tmp_path))
    assert tab.auto_detect_from_filenames() == 3
