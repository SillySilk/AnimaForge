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


def test_card_matches_search_and_segment():
    done = ImageCard("/x/ivy_barnes_004.png", "/x/a.txt", "a girl in a park", 0, status="done")
    bare = ImageCard("/x/ivy_barnes_009.png", "/x/b.txt", "", 0, status="bare")
    # segmented filter
    assert done.matches("", "captioned") and not bare.matches("", "captioned")
    assert bare.matches("", "needs") and not done.matches("", "needs")
    assert done.matches("", "all") and bare.matches("", "all")
    # search across filename + caption
    assert done.matches("park", "all") and not done.matches("beach", "all")
    assert bare.matches("009", "all")


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


def test_gallery_click_opens_editor_with_card_captions(monkeypatch):
    # Regression: after the gallery redesign removed the card's inline _caption_edit,
    # _open_image_editor still read it — every click raised inside the slot and the
    # editor never opened ("we lost click-to-edit").
    import ui.dataset_tab as dt_mod
    from ui.dataset_tab import DatasetTab

    t = DatasetTab()
    t._cards = [_card(), ImageCard("/x/b.png", "/x/b.txt", "second caption", 0)]

    opened = {}

    class FakeDlg:
        class _Sig:
            def connect(self, *_):
                pass
        caption_saved = _Sig()
        cast_changed = _Sig()

        def __init__(self, items, index, characters, parent):
            opened["items"], opened["index"] = items, index

        def exec(self):
            opened["shown"] = True

    monkeypatch.setattr(dt_mod, "ImageEditorDialog", FakeDlg)
    t._open_image_editor("/x/b.png")
    assert opened.get("shown")
    assert opened["index"] == 1
    assert opened["items"][1]["caption"] == "second caption"
    assert opened["items"][0]["caption"] == LONG


def test_build_caption_job_snapshots_the_live_settings(tmp_path):
    from ui.dataset_tab import DatasetTab
    from core.caption_policy import KEEP
    t = DatasetTab()
    t._folder_path = str(tmp_path)
    t._sdscripts_path = "C:/sd"
    t.set_trigger_word("manbag")
    t.set_prefix("masterpiece")
    job = t._build_caption_job(KEEP)
    assert job.dataset_folder == str(tmp_path)
    assert job.sdscripts_path == "C:/sd"
    assert job.trigger == "manbag"
    assert job.prefix == "masterpiece"
    assert job.policy == KEEP
    assert job.chain[0] == "tag" and job.chain[-1] == "combine"


def test_caption_tick_adapts_runner_four_arg_to_three():
    # CaptionRunner.tick carries (phase, done, total, filename); DatasetTab.caption_tick
    # must re-emit only the first three. If this silently regresses, Home's stage chips
    # go dark mid-run.
    from ui.dataset_tab import DatasetTab
    t = DatasetTab()
    seen = []
    t.caption_tick.connect(lambda *a: seen.append(a))
    t._runner.tick.emit("Tag", 3, 12, "a.png")
    assert seen == [("Tag", 3, 12)]


def test_captioned_milestone_fires_on_pre_combine_stage(tmp_path):
    # caption_stage_done("captioned") must fire when the last raw pass (the stage right
    # before "combine") lands — not on "combine" itself, and not on earlier stages.
    from PIL import Image
    from ui.dataset_tab import DatasetTab
    from core.caption_runner import CaptionJob
    Image.new("RGB", (8, 8), (10, 10, 10)).save(tmp_path / "a.png")
    t = DatasetTab()
    t._folder_path = str(tmp_path)
    t._runner._job = CaptionJob(
        dataset_folder=str(tmp_path), sdscripts_path="C:/sd",
        chain=["tag", "describe", "combine"])
    stages = []
    t.caption_stage_done.connect(stages.append)
    t._runner.stage_done.emit("tag")
    assert stages == []                 # tag is not the pre-combine stage
    t._runner.stage_done.emit("describe")
    assert stages == ["captioned"]      # describe precedes combine -> milestone
    t._runner.stage_done.emit("combine")
    assert stages == ["captioned"]      # combine itself does not re-emit


def test_delete_button_on_thumbnail_and_double_badge():
    c = _card()
    # trash can overlays the thumbnail (top-right), always visible
    assert c._delete_btn.parent() is c._thumb_label
    assert not hasattr(c, "delete_btn_row")  # old below-thumbnail spot is gone
    fired = []
    c.image_deleted.connect(lambda p: fired.append(p))
    c._delete_btn.click()
    assert fired == ["/x/aria_03.png"]
    # DOUBLE badge hidden by default, shown with twins, hideable again
    assert c._dup_badge.isHidden()
    c.mark_duplicate(["aria_03.jpg"])
    assert not c._dup_badge.isHidden()
    assert "aria_03.jpg" in c._dup_badge.toolTip()
    c.mark_duplicate([])
    assert c._dup_badge.isHidden()
