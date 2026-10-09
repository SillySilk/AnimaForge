import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from core import caption_runner as cr
from core.caption_runner import CaptionJob, CaptionRunner
from core.dataset_manager import combine_all

_app = QApplication.instance() or QApplication([])


def _img(folder, stem, nl=None, tags=None, txt=None):
    (folder / f"{stem}.png").write_bytes(b"x")
    if nl is not None:
        (folder / f"{stem}.nl").write_text(nl, encoding="utf-8")
    if tags is not None:
        (folder / f"{stem}.tags").write_text(tags, encoding="utf-8")
    if txt is not None:
        (folder / f"{stem}.txt").write_text(txt, encoding="utf-8")


def _txt(folder, stem):
    return (folder / f"{stem}.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("style,chain", [
    ("prose", ["describe", "combine"]),
    ("tags", ["tag", "combine"]),
    ("prose+tags", ["tag", "describe", "combine"]),
])
def test_chain_for_style(style, chain):
    assert cr.chain_for_style(style) == chain


def test_unknown_style_is_an_error_naming_it():
    with pytest.raises(ValueError, match="haiku"):
        cr.chain_for_style("haiku")
    with pytest.raises(ValueError, match="haiku"):
        cr.sources_for_style("haiku")


def test_combine_reads_both_sidecars_by_default(tmp_path):
    _img(tmp_path, "a", nl="a woman standing", tags="1girl, red_dress")
    combine_all(str(tmp_path), prefix="trig", apply_anchors=False)
    assert _txt(tmp_path, "a") == "trig, a woman standing, 1girl, red dress"


def test_combine_prose_style_ignores_existing_tags_sidecars(tmp_path):
    _img(tmp_path, "a", nl="a woman standing", tags="1girl, red_dress")
    combine_all(str(tmp_path), prefix="trig", apply_anchors=False, sources=("nl",))
    assert _txt(tmp_path, "a") == "trig, a woman standing"
    assert (tmp_path / "a.tags").read_text(encoding="utf-8") == "1girl, red_dress"  # untouched


def test_combine_tags_style_ignores_existing_nl_sidecars(tmp_path):
    _img(tmp_path, "a", nl="a woman standing", tags="1girl, red_dress")
    combine_all(str(tmp_path), prefix="trig", apply_anchors=False, sources=("tags",))
    assert _txt(tmp_path, "a") == "trig, 1girl, red dress"
    assert (tmp_path / "a.nl").read_text(encoding="utf-8") == "a woman standing"  # untouched


def test_style_never_overwrites_a_txt_it_cannot_rebuild(tmp_path):
    # Only a .tags sidecar exists and the style reads only prose: nothing to build from.
    _img(tmp_path, "a", tags="1girl", txt="hand written caption")
    combine_all(str(tmp_path), prefix="trig", apply_anchors=False, sources=("nl",))
    assert _txt(tmp_path, "a") == "hand written caption"


def test_job_defaults_to_prose_and_tags():
    assert CaptionJob(dataset_folder="d", sdscripts_path="s").caption_style == "prose+tags"


def test_combine_stage_reads_only_the_styles_sidecars(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(cr, "combine_all", lambda folder, **kw: seen.update(kw) or (1, 0))
    runner = CaptionRunner()
    runner._job = CaptionJob(dataset_folder=str(tmp_path), sdscripts_path="s",
                             caption_style="prose")
    runner._stages = [("combine", ["a.png"])]
    runner._running = True
    runner._next()
    assert seen["sources"] == ("nl",)


def test_restricted_style_skips_an_image_it_has_nothing_to_build_from(tmp_path):
    # A describe that failed for this image left no .nl; under prose style the .tags is
    # ignored. Writing "trig" alone would mark the image captioned and resume would
    # never retry it, so nothing is written and the image stays untouched.
    _img(tmp_path, "a", tags="1girl")
    written, errors = combine_all(str(tmp_path), prefix="trig", apply_anchors=False,
                                  sources=("nl",))
    assert (written, errors) == (0, 0)
    assert not (tmp_path / "a.txt").exists()


def test_job_names_the_subject_by_default():
    assert CaptionJob(dataset_folder="d", sdscripts_path="s").name_subject is True


def test_describe_stage_leaves_the_trigger_out_when_not_naming_the_subject(monkeypatch):
    seen = {}
    runner = CaptionRunner()
    monkeypatch.setattr(runner._joy, "start", lambda **kw: seen.update(kw))
    runner._job = CaptionJob(dataset_folder="d", sdscripts_path="s", trigger="inkwash",
                             name_subject=False)
    runner._stages = [("describe", ["a.png"])]
    runner._running = True
    runner._next()
    assert seen["name"] == ""


def test_runner_records_the_caption_style(tmp_path, monkeypatch):
    from core import caption_manifest as cm
    _img(tmp_path, "a")
    runner = CaptionRunner()
    monkeypatch.setattr(runner._joy, "start", lambda **kw: None)
    runner.start(CaptionJob(dataset_folder=str(tmp_path), sdscripts_path="s",
                            chain=["describe", "combine"], caption_style="prose",
                            policy="overwrite"))
    assert cm.load(str(tmp_path))["caption_style"] == "prose"
