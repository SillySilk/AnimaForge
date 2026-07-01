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


def test_card_shows_readonly_caption_preview():
    # The card is a clean preview tile now — a read-only 2-line caption preview, no inline
    # editor (editing happens in the modal opened on click).
    c = _card()
    assert not hasattr(c, "_caption_edit")
    assert c._caption_preview.text() == LONG
    c.refresh_caption("a fresh caption")
    assert c._caption_preview.text() == "a fresh caption"


def test_empty_caption_preview_and_status():
    c = ImageCard("/x/bare.png", "/x/bare.txt", "", 0, status="bare")
    assert c._caption_preview.text() == "No caption yet"
    assert c._caption_preview.property("empty") == "true"
    assert c._status == "bare"
    # editing a caption flips the status dot to done
    c.refresh_caption("now captioned")
    assert c._status == "done"


def test_image_status_classification():
    from ui.dataset_tab import DatasetTab
    assert DatasetTab._image_status({"image_path": "/x/a.png", "caption": "hi"}) == "done"
    assert DatasetTab._image_status({"image_path": "/x/none.png", "caption": ""}) == "bare"


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
