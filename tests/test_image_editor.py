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


def _sidecars(items, i=0, tags=None, nl=None):
    stem = Path(items[i]["image_path"])
    if tags is not None:
        stem.with_suffix(".tags").write_text(tags, encoding="utf-8")
    if nl is not None:
        stem.with_suffix(".nl").write_text(nl, encoding="utf-8")
    return stem.with_suffix(".tags"), stem.with_suffix(".nl")


def test_sidecar_editors_load_and_save(tmp_path: Path):
    items = _items(tmp_path, 1)
    tags_p, nl_p = _sidecars(items, tags="1girl, park", nl="a girl in a park")
    dlg = ImageEditorDialog(items, 0, C.DatasetCharacters())
    assert dlg._tags_edit.toPlainText() == "1girl, park"
    assert dlg._nl_edit.toPlainText() == "a girl in a park"
    dlg._tags_edit.setPlainText("1girl, beach")
    dlg._nl_edit.setPlainText("a girl at the beach")
    dlg._flush_sidecars()
    assert tags_p.read_text(encoding="utf-8") == "1girl, beach"
    assert nl_p.read_text(encoding="utf-8") == "a girl at the beach"


def test_sidecar_flush_never_creates_missing_files(tmp_path: Path):
    # A dataset captioned outside AnimaForge has no sidecars — flushing empty
    # editors must not litter it with empty .tags/.nl files.
    items = _items(tmp_path, 1)
    dlg = ImageEditorDialog(items, 0, C.DatasetCharacters())
    dlg._flush_sidecars()
    stem = Path(items[0]["image_path"])
    assert not stem.with_suffix(".tags").exists()
    assert not stem.with_suffix(".nl").exists()


def test_navigation_flushes_sidecar_edits(tmp_path: Path):
    items = _items(tmp_path, 2)
    tags_p, _ = _sidecars(items, tags="old tags")
    dlg = ImageEditorDialog(items, 0, C.DatasetCharacters())
    dlg._tags_edit.setPlainText("new tags")
    dlg._go(1)
    assert tags_p.read_text(encoding="utf-8") == "new tags"


def test_recombine_button_flushes_then_emits(tmp_path: Path):
    items = _items(tmp_path, 1)
    tags_p, _ = _sidecars(items, tags="old tags")
    dlg = ImageEditorDialog(items, 0, C.DatasetCharacters())
    got = []
    dlg.recombine_requested.connect(got.append)
    dlg._tags_edit.setPlainText("fresh tags")
    dlg._recombine_btn.click()
    assert got == [items[0]["image_path"]]
    # The sidecar edit must already be on disk when the tab re-combines.
    assert tags_p.read_text(encoding="utf-8") == "fresh tags"


def test_conflict_warning_shows_and_hides(tmp_path: Path):
    items = _items(tmp_path, 1)
    dlg = ImageEditorDialog(items, 0, C.DatasetCharacters())
    dlg._caption_edit.setPlainText("1girl, black hair, brown hair")
    assert not dlg._lint_label.isHidden()
    assert "hair color" in dlg._lint_label.text()
    dlg._caption_edit.setPlainText("1girl, black hair")
    assert dlg._lint_label.isHidden()


def test_set_caption_text_updates_editor_and_item(tmp_path: Path):
    items = _items(tmp_path, 1)
    dlg = ImageEditorDialog(items, 0, C.DatasetCharacters())
    dlg.set_caption_text("rebuilt caption")
    assert dlg._caption_edit.toPlainText() == "rebuilt caption"
    assert items[0]["caption"] == "rebuilt caption"


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
