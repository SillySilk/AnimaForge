import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from ui.image_editor import ImageEditorDialog
from core import characters as C

_app = QApplication.instance() or QApplication([])


def _items(tmp_path, n=2):
    items = []
    for i in range(n):
        txt = tmp_path / f"img{i}.txt"
        txt.write_text("", encoding="utf-8")
        items.append({
            "image_path": str(tmp_path / f"img{i}.png"),
            "txt_path": str(txt),
            "caption": f"cap{i}",
        })
    return items


def test_loads_and_navigates(tmp_path: Path):
    items = _items(tmp_path, 3)
    chars = C.DatasetCharacters(roster=[C.Character("sarah", "red hair")])
    dlg = ImageEditorDialog(items, 0, chars)
    assert dlg._counter.text() == "1 / 3"
    dlg._go(1)
    assert dlg._index == 1
    assert dlg._counter.text() == "2 / 3"


def test_caption_flush_saves_and_emits(tmp_path: Path):
    items = _items(tmp_path, 1)
    dlg = ImageEditorDialog(items, 0, C.DatasetCharacters())
    got = []
    dlg.caption_saved.connect(lambda p, t: got.append((p, t)))
    dlg._caption_edit.setPlainText("hello world")
    dlg._flush_caption()
    assert got and got[0][1] == "hello world"
    assert Path(items[0]["txt_path"]).read_text(encoding="utf-8") == "hello world"


def test_cast_toggle_updates_assignments(tmp_path: Path):
    items = _items(tmp_path, 1)
    chars = C.DatasetCharacters(roster=[C.Character("sarah")])
    dlg = ImageEditorDialog(items, 0, chars)
    changed = []
    dlg.cast_changed.connect(lambda n: changed.append(n))
    _tok, cb = dlg._cast_checks[0]
    cb.setChecked(True)
    assert chars.assignments["img0.png"]["present"] == ["sarah"]
    assert changed == ["img0.png"]
