import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from ui.dataset_tab import ImageCard

_app = QApplication.instance() or QApplication([])

LONG = "\n".join(f"line {i}" for i in range(40))  # long enough to scroll once laid out


def _card():
    return ImageCard("/x/aria_03.png", "/x/aria_03.txt", LONG, 0)


def _set_cursor(edit, pos):
    cur = edit.textCursor()
    cur.setPosition(pos)
    edit.setTextCursor(cur)


def test_refresh_unchanged_is_noop():
    # Identical text must NOT re-set the box (setPlainText would slam the cursor to 0).
    c = _card()
    edit = c._caption_edit
    _set_cursor(edit, 10)
    c.refresh_caption(LONG)
    assert edit.textCursor().position() == 10


def test_refresh_changed_preserves_cursor():
    # Changed text re-sets the box but keeps the caret where it was (layout-independent
    # witness that scroll/cursor are preserved rather than reset to the top).
    c = _card()
    edit = c._caption_edit
    _set_cursor(edit, 15)
    c.refresh_caption(LONG + " extra")
    assert edit.textCursor().position() == 15


def test_refresh_changed_preserves_scroll_when_scrollable():
    # Bonus: when the box actually has a scroll range (needs layout), the position holds.
    c = _card()
    c.resize(220, 200)
    c.show()  # offscreen — forces layout so the caption box gets a scroll range
    _app.processEvents()
    sb = c._caption_edit.verticalScrollBar()
    if sb.maximum() > 0:  # only meaningful when the content overflows
        sb.setValue(min(3, sb.maximum()))
        want = sb.value()
        c.refresh_caption(LONG + "\nextra line")
        assert sb.value() == min(want, sb.maximum())


def test_filename_and_set_processing_property():
    c = _card()
    assert c.filename == "aria_03.png"
    c.set_processing(True)
    assert c.property("processing") == "true"
    c.set_processing(False)
    assert c.property("processing") == "false"


def test_set_processing_frame_moves_highlight():
    from ui.dataset_tab import DatasetTab
    t = DatasetTab()
    a, b = _card(), ImageCard("/x/b.png", "/x/b.txt", "cap", 0)
    t._cards_by_name = {"aria_03.png": a, "b.png": b}
    t._processing_card = None
    t._set_processing_frame("aria_03.png")
    assert a.property("processing") == "true"
    t._set_processing_frame("b.png")
    assert a.property("processing") == "false"
    assert b.property("processing") == "true"
    t._clear_processing_frame()
    assert b.property("processing") == "false"
