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


def test_full_chain_emits_stage_done_exactly_twice_in_order(tmp_path):
    # End-to-end: a full successful chain fires caption_stage_done exactly twice,
    # "captioned" then "processed" -- never more, never reordered.
    from ui.dataset_tab import DatasetTab
    from core.caption_runner import CaptionJob
    t = DatasetTab()
    t._folder_path = str(tmp_path)
    t._auto_mode = True   # skip the "Process complete" QMessageBox popup
    t._runner._job = CaptionJob(
        dataset_folder=str(tmp_path), sdscripts_path="C:/sd",
        chain=["tag", "describe", "combine"])
    stages = []
    t.caption_stage_done.connect(stages.append)
    t._runner.stage_done.emit("tag")
    t._runner.stage_done.emit("describe")
    t._runner.stage_done.emit("combine")
    t._runner.finished.emit(True)
    assert stages == ["captioned", "processed"]


def test_manual_describe_is_blocked_while_the_runner_chain_is_active(tmp_path, monkeypatch):
    # Finding 1 (CRITICAL): CaptionRunner owns its own Tagger/JoyCaption processes, so
    # the tab's own procs are idle during a chain. A guard that only checks the tab's procs
    # would let a second JoyCaption launch against the same folder mid-chain.
    import ui.dataset_tab as dt_mod
    t = dt_mod.DatasetTab()
    t._folder_path = str(tmp_path)
    t._sdscripts_path = "C:/sd"
    t._image_data = [{"image_path": str(tmp_path / "a.png"), "txt_path": str(tmp_path / "a.txt"),
                       "caption": ""}]
    monkeypatch.setattr(t._runner, "is_running", lambda: True)
    monkeypatch.setattr(dt_mod.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    # Auto-accept the "Generate natural-language captions...?" confirmation so a guard hole
    # actually reaches _start_describe() instead of silently stopping at the confirm dialog.
    monkeypatch.setattr(dt_mod.QMessageBox, "exec", lambda self: dt_mod.QMessageBox.Yes)
    launched = []
    monkeypatch.setattr(t, "_start_describe", lambda: launched.append("describe"))
    t._describe_joycaption()
    assert launched == []          # blocked


def test_manual_tagger_dialog_is_blocked_while_the_runner_chain_is_active(tmp_path, monkeypatch):
    import ui.dataset_tab as dt_mod
    t = dt_mod.DatasetTab()
    t._folder_path = str(tmp_path)
    t._sdscripts_path = "C:/sd"
    monkeypatch.setattr(t._runner, "is_running", lambda: True)
    monkeypatch.setattr(dt_mod.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    # Auto-accept the "Auto-Tag with WD14 Tagger" config dialog so a guard hole actually
    # reaches self._tagger.start(...) instead of silently stopping at the config dialog.
    monkeypatch.setattr(dt_mod.QDialog, "exec", lambda self: dt_mod.QDialog.Accepted)
    launched = []
    monkeypatch.setattr(t._tagger, "start", lambda *a, **k: launched.append("tag"))
    t._open_tagger_dialog()
    assert launched == []          # blocked


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


# ----------------------------------------------------------------------------
# Task 6: existing-captions conflict resolution (conflict_text + _resolve_policy)
# ----------------------------------------------------------------------------

def test_conflict_text_names_foreign_captions():
    from ui.dataset_tab import conflict_text
    from core.caption_policy import FolderCaptionState
    st = FolderCaptionState(total=80, captioned=["a"] * 47, partial=[],
                            untouched=["u"] * 33, foreign=47)
    msg = conflict_text(st)
    assert "47" in msg
    assert "not written by AnimaForge" in msg


def test_conflict_text_omits_foreign_wording_when_captions_are_ours():
    from ui.dataset_tab import conflict_text
    from core.caption_policy import FolderCaptionState
    st = FolderCaptionState(total=80, captioned=["a"] * 47, partial=[],
                            untouched=["u"] * 33, foreign=0)
    msg = conflict_text(st)
    assert "47" in msg
    assert "not written by AnimaForge" not in msg


def _write_conflicting_folder(tmp_path):
    """A folder with one already-captioned image and one untouched image —
    enough to trip cp.has_conflict() while still exercising the Keep-count wording."""
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "a.txt").write_text("an existing caption", encoding="utf-8")
    (tmp_path / "b.png").write_bytes(b"x")


def _button_with_role(box, role):
    for b in box.buttons():
        if box.buttonRole(b) == role:
            return b
    raise AssertionError(f"no button with role {role!r} on this QMessageBox")


def _fake_exec_clicking(role, remember=False):
    """A QMessageBox.exec replacement: instead of blocking on a real modal loop,
    it programmatically clicks the button with the given role (optionally ticking
    the 'Remember my choice' checkbox first) so _resolve_policy's post-exec()
    box.clickedButton() read reflects that choice — no real event loop needed."""
    def _exec(self):
        if remember:
            cb = self.checkBox()
            assert cb is not None, "no 'Remember my choice' checkbox on this box"
            cb.setChecked(True)
        _button_with_role(self, role).click()
        return None
    return _exec


def test_resolve_policy_no_conflict_returns_overwrite_without_dialog(tmp_path, monkeypatch):
    import ui.dataset_tab as dt_mod
    from core import caption_policy as cp

    def _boom(self):
        raise AssertionError("dialog must not be shown when there is no conflict")
    monkeypatch.setattr(dt_mod.QMessageBox, "exec", _boom)

    t = dt_mod.DatasetTab()
    t._folder_path = str(tmp_path)   # empty folder -> nothing captioned -> no conflict
    assert t._resolve_policy() == cp.OVERWRITE


def test_resolve_policy_conflict_with_keep_setting_returns_keep_without_dialog(tmp_path, monkeypatch):
    import ui.dataset_tab as dt_mod
    from core import caption_policy as cp
    from core.settings import AppSettings
    _write_conflicting_folder(tmp_path)

    def _boom(self):
        raise AssertionError("dialog must not be shown when the policy is already decided")
    monkeypatch.setattr(dt_mod.QMessageBox, "exec", _boom)

    app = AppSettings()
    prev = app.get("caption_existing_policy")
    try:
        app.set("caption_existing_policy", cp.KEEP)
        t = dt_mod.DatasetTab()
        t._folder_path = str(tmp_path)
        assert t._resolve_policy() == cp.KEEP
    finally:
        app.set("caption_existing_policy", prev)


def test_resolve_policy_conflict_ask_keep_button_returns_keep(tmp_path, monkeypatch):
    import ui.dataset_tab as dt_mod
    from core import caption_policy as cp
    from core.settings import AppSettings
    _write_conflicting_folder(tmp_path)
    monkeypatch.setattr(dt_mod.QMessageBox, "exec",
                        _fake_exec_clicking(dt_mod.QMessageBox.AcceptRole))

    app = AppSettings()
    prev = app.get("caption_existing_policy")
    try:
        app.set("caption_existing_policy", cp.ASK)
        t = dt_mod.DatasetTab()
        t._folder_path = str(tmp_path)
        assert t._resolve_policy() == cp.KEEP
        assert t.start_cancelled_by_user() is False
    finally:
        app.set("caption_existing_policy", prev)


def test_resolve_policy_conflict_ask_overwrite_button_returns_overwrite(tmp_path, monkeypatch):
    import ui.dataset_tab as dt_mod
    from core import caption_policy as cp
    from core.settings import AppSettings
    _write_conflicting_folder(tmp_path)
    monkeypatch.setattr(dt_mod.QMessageBox, "exec",
                        _fake_exec_clicking(dt_mod.QMessageBox.DestructiveRole))

    app = AppSettings()
    prev = app.get("caption_existing_policy")
    try:
        app.set("caption_existing_policy", cp.ASK)
        t = dt_mod.DatasetTab()
        t._folder_path = str(tmp_path)
        assert t._resolve_policy() == cp.OVERWRITE
        assert t.start_cancelled_by_user() is False
    finally:
        app.set("caption_existing_policy", prev)


def test_resolve_policy_conflict_ask_cancel_returns_none(tmp_path, monkeypatch):
    import ui.dataset_tab as dt_mod
    from core import caption_policy as cp
    from core.settings import AppSettings
    _write_conflicting_folder(tmp_path)
    monkeypatch.setattr(dt_mod.QMessageBox, "exec",
                        _fake_exec_clicking(dt_mod.QMessageBox.RejectRole))

    app = AppSettings()
    prev = app.get("caption_existing_policy")
    try:
        app.set("caption_existing_policy", cp.ASK)
        t = dt_mod.DatasetTab()
        t._folder_path = str(tmp_path)
        assert t._resolve_policy() is None
        assert t.start_cancelled_by_user() is True
    finally:
        app.set("caption_existing_policy", prev)


def test_resolve_policy_remember_choice_writes_setting(tmp_path, monkeypatch):
    import ui.dataset_tab as dt_mod
    from core import caption_policy as cp
    from core.settings import AppSettings
    _write_conflicting_folder(tmp_path)
    monkeypatch.setattr(dt_mod.QMessageBox, "exec",
                        _fake_exec_clicking(dt_mod.QMessageBox.DestructiveRole, remember=True))

    app = AppSettings()
    prev = app.get("caption_existing_policy")
    try:
        app.set("caption_existing_policy", cp.ASK)
        t = dt_mod.DatasetTab()
        t._folder_path = str(tmp_path)
        assert t._resolve_policy() == cp.OVERWRITE
        assert AppSettings().get("caption_existing_policy") == cp.OVERWRITE
    finally:
        app.set("caption_existing_policy", prev)


def test_resolve_policy_unchecked_remember_does_not_write_setting(tmp_path, monkeypatch):
    import ui.dataset_tab as dt_mod
    from core import caption_policy as cp
    from core.settings import AppSettings
    _write_conflicting_folder(tmp_path)
    monkeypatch.setattr(dt_mod.QMessageBox, "exec",
                        _fake_exec_clicking(dt_mod.QMessageBox.DestructiveRole, remember=False))

    app = AppSettings()
    prev = app.get("caption_existing_policy")
    try:
        app.set("caption_existing_policy", cp.ASK)
        t = dt_mod.DatasetTab()
        t._folder_path = str(tmp_path)
        assert t._resolve_policy() == cp.OVERWRITE
        assert AppSettings().get("caption_existing_policy") == cp.ASK   # unchanged
    finally:
        app.set("caption_existing_policy", prev)


def test_process_clicked_folds_conflict_into_one_dialog_not_two(tmp_path, monkeypatch):
    # The core "fold, don't stack" requirement: a conflict+ASK run shows exactly one
    # QMessageBox.exec() call (Keep/Overwrite/Cancel), never a separate plain
    # "Run all N steps?" confirm shown first or after.
    import ui.dataset_tab as dt_mod
    from core import caption_policy as cp
    from core.settings import AppSettings
    _write_conflicting_folder(tmp_path)

    exec_calls = []

    def _fake_exec(self):
        exec_calls.append(self)
        _button_with_role(self, dt_mod.QMessageBox.RejectRole).click()
        return None
    monkeypatch.setattr(dt_mod.QMessageBox, "exec", _fake_exec)

    app = AppSettings()
    prev = app.get("caption_existing_policy")
    try:
        app.set("caption_existing_policy", cp.ASK)
        t = dt_mod.DatasetTab()
        t._folder_path = str(tmp_path)
        t._sdscripts_path = "C:/sd"
        t._image_data = [
            {"image_path": str(tmp_path / "a.png"), "txt_path": str(tmp_path / "a.txt"), "caption": "an existing caption"},
            {"image_path": str(tmp_path / "b.png"), "txt_path": str(tmp_path / "b.txt"), "caption": ""},
        ]
        started = []
        monkeypatch.setattr(t, "_start_runner_or_warn", lambda job: started.append(job) or True)
        t._process_clicked()
        assert len(exec_calls) == 1     # exactly one dialog, not stacked
        assert started == []            # Cancel -> nothing started
        assert t._auto_mode is False
    finally:
        app.set("caption_existing_policy", prev)


def test_start_auto_caption_cancelled_returns_false_and_sets_flag(tmp_path, monkeypatch):
    import ui.dataset_tab as dt_mod
    from core import caption_policy as cp
    from core.settings import AppSettings
    _write_conflicting_folder(tmp_path)
    monkeypatch.setattr(dt_mod.QMessageBox, "exec",
                        _fake_exec_clicking(dt_mod.QMessageBox.RejectRole))

    app = AppSettings()
    prev = app.get("caption_existing_policy")
    try:
        app.set("caption_existing_policy", cp.ASK)
        t = dt_mod.DatasetTab()
        t._folder_path = str(tmp_path)
        t._sdscripts_path = "C:/sd"
        t._image_data = [{"image_path": str(tmp_path / "a.png"),
                           "txt_path": str(tmp_path / "a.txt"), "caption": "an existing caption"}]
        assert t.start_auto_caption() is False
        assert t.start_cancelled_by_user() is True
    finally:
        app.set("caption_existing_policy", prev)


def test_start_auto_caption_genuine_refusal_leaves_cancelled_flag_false(tmp_path):
    import ui.dataset_tab as dt_mod
    t = dt_mod.DatasetTab()
    t._folder_path = str(tmp_path)
    t._sdscripts_path = ""   # no sd-scripts path configured -> genuine refusal, not a cancel
    t._image_data = [{"image_path": str(tmp_path / "a.png"),
                       "txt_path": str(tmp_path / "a.txt"), "caption": ""}]
    assert t.start_auto_caption() is False
    assert t.start_cancelled_by_user() is False


# ----------------------------------------------------------------------------
# GPU exclusivity: MainWindow injects set_gpu_busy_check() so DatasetTab refuses
# to start captioning (any path) while the batch runner or training owns the GPU.
# ----------------------------------------------------------------------------

def _ready_tab(tmp_path):
    import ui.dataset_tab as dt_mod
    t = dt_mod.DatasetTab()
    t._folder_path = str(tmp_path)
    t._sdscripts_path = "C:/sd"
    t._image_data = [{"image_path": str(tmp_path / "a.png"),
                       "txt_path": str(tmp_path / "a.txt"), "caption": ""}]
    return t


def test_default_gpu_busy_check_is_none_for_a_standalone_tab():
    # A DatasetTab() built outside MainWindow (as every other test in this file
    # does) must keep working -- the injected check defaults to a no-op.
    import ui.dataset_tab as dt_mod
    t = dt_mod.DatasetTab()
    assert t._gpu_busy_check() is None


def test_process_clicked_refused_when_gpu_busy_elsewhere(tmp_path, monkeypatch):
    import ui.dataset_tab as dt_mod
    t = _ready_tab(tmp_path)
    t.set_gpu_busy_check(lambda: "a batch run is in progress")

    def _boom(self):
        raise AssertionError("no confirm/conflict dialog should show when GPU is busy elsewhere")
    monkeypatch.setattr(dt_mod.QMessageBox, "exec", _boom)
    monkeypatch.setattr(dt_mod.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    started = []
    monkeypatch.setattr(t, "_start_runner_or_warn", lambda job: started.append(job) or True)
    t._process_clicked()
    assert started == []


def test_start_auto_caption_refused_when_gpu_busy_elsewhere(tmp_path):
    t = _ready_tab(tmp_path)
    t.set_gpu_busy_check(lambda: "a batch run is in progress")
    assert t.start_auto_caption() is False
    # Not a user cancel -- _qr_advance must show the real "cannot caption" warning.
    assert t.start_cancelled_by_user() is False


def test_describe_joycaption_refused_when_gpu_busy_elsewhere(tmp_path, monkeypatch):
    import ui.dataset_tab as dt_mod
    t = _ready_tab(tmp_path)
    t.set_gpu_busy_check(lambda: "a batch run is in progress")
    monkeypatch.setattr(dt_mod.QMessageBox, "information", staticmethod(lambda *a, **k: None))

    def _boom(self):
        raise AssertionError("the Describe confirm dialog must not show when GPU is busy elsewhere")
    monkeypatch.setattr(dt_mod.QMessageBox, "exec", _boom)
    launched = []
    monkeypatch.setattr(t, "_start_describe", lambda: launched.append("describe"))
    t._describe_joycaption()
    assert launched == []


def test_start_describe_passes_overwrite_to_joycaption(tmp_path, monkeypatch):
    # 'Redo all' must reach JoyCaption as overwrite=True — otherwise an already-
    # captioned set silently skips every image and nothing streams in the log.
    t = _ready_tab(tmp_path)
    calls = {}
    monkeypatch.setattr(t._joycaption, "start", lambda **kw: calls.update(kw))
    t._start_describe(overwrite=True)
    assert calls.get("overwrite") is True


def test_start_describe_default_is_skip_existing(tmp_path, monkeypatch):
    t = _ready_tab(tmp_path)
    calls = {}
    monkeypatch.setattr(t._joycaption, "start", lambda **kw: calls.update(kw))
    t._start_describe()
    assert calls.get("overwrite") is False


def _click_describe_button(monkeypatch, needle):
    """Answer the Describe dialog by clicking the button whose label contains needle."""
    import ui.dataset_tab as dt_mod
    monkeypatch.setattr(dt_mod.QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(
        dt_mod.QMessageBox, "clickedButton",
        lambda self: next((b for b in self.buttons() if needle in b.text()), None))


def test_describe_dialog_redo_all_overwrites(tmp_path, monkeypatch):
    from PIL import Image
    Image.new("RGB", (8, 8), (10, 10, 10)).save(tmp_path / "a.png")
    (tmp_path / "a.nl").write_text("old prose", encoding="utf-8")
    t = _ready_tab(tmp_path)
    _click_describe_button(monkeypatch, "Redo")
    seen = {}
    monkeypatch.setattr(t, "_start_describe",
                        lambda overwrite=False: seen.setdefault("ow", overwrite))
    t._describe_joycaption()
    assert seen.get("ow") is True


def test_describe_dialog_missing_only_skips_existing(tmp_path, monkeypatch):
    from PIL import Image
    Image.new("RGB", (8, 8), (10, 10, 10)).save(tmp_path / "a.png")
    (tmp_path / "a.nl").write_text("old prose", encoding="utf-8")
    t = _ready_tab(tmp_path)
    _click_describe_button(monkeypatch, "missing")
    seen = {}
    monkeypatch.setattr(t, "_start_describe",
                        lambda overwrite=False: seen.setdefault("ow", overwrite))
    t._describe_joycaption()
    assert seen.get("ow") is False


def test_describe_dialog_cancel_launches_nothing(tmp_path, monkeypatch):
    t = _ready_tab(tmp_path)
    _click_describe_button(monkeypatch, "NO-SUCH-BUTTON")   # clickedButton -> None
    launched = []
    monkeypatch.setattr(t, "_start_describe",
                        lambda overwrite=False: launched.append(overwrite))
    t._describe_joycaption()
    assert launched == []
