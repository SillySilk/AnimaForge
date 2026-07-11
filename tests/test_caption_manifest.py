import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import caption_manifest as cm


def _img(folder: Path, stem: str, tags=None, nl=None, txt=None):
    (folder / f"{stem}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    for ext, val in ((".tags", tags), (".nl", nl), (".txt", txt)):
        if val is not None:
            (folder / f"{stem}{ext}").write_text(val, encoding="utf-8")


def test_load_missing_manifest_is_empty_dict(tmp_path):
    assert cm.load(str(tmp_path)) == {}
    assert cm.images_dict(str(tmp_path)) is None


def test_load_corrupt_manifest_is_empty_dict_and_never_raises(tmp_path):
    p = tmp_path / cm.MANIFEST_REL
    p.parent.mkdir(parents=True)
    p.write_text("{ not json", encoding="utf-8")
    assert cm.load(str(tmp_path)) == {}


def test_record_settings_and_mark_stage_roundtrip(tmp_path):
    _img(tmp_path, "a")
    cm.record_settings(str(tmp_path), trigger="manbag", prefix="masterpiece",
                       order="nl_first", chain=["tag", "describe", "combine"])
    cm.mark_stage(str(tmp_path), "tag", [str(tmp_path / "a.png")])
    d = cm.load(str(tmp_path))
    assert d["trigger"] == "manbag"
    assert d["chain"] == ["tag", "describe", "combine"]
    assert d["images"]["a.png"]["tag"] == "done"
    assert "updated" in d


def test_reconcile_lets_sidecars_win_when_a_file_was_hand_deleted(tmp_path):
    _img(tmp_path, "a", tags="1girl", nl="a woman")
    cm.mark_stage(str(tmp_path), "tag", [str(tmp_path / "a.png")])
    cm.mark_stage(str(tmp_path), "describe", [str(tmp_path / "a.png")])
    (tmp_path / "a.nl").unlink()                       # user deleted the prose
    d = cm.reconcile(str(tmp_path))
    assert d["images"]["a.png"]["tag"] == "done"       # .tags still there
    assert d["images"]["a.png"]["describe"] == "pending"


def test_reconcile_preserves_describe_while_its_nl_survives(tmp_path):
    """describe stays 'done' as long as the .nl it wrote is still on disk."""
    _img(tmp_path, "a", nl="a woman")
    cm.mark_stage(str(tmp_path), "describe", [str(tmp_path / "a.png")])
    d = cm.reconcile(str(tmp_path))
    assert d["images"]["a.png"]["describe"] == "done"


def test_images_dict_feeds_caption_policy_foreign_count(tmp_path):
    from core import caption_policy as cp
    _img(tmp_path, "ours", txt="c")
    _img(tmp_path, "theirs", txt="c")
    cm.mark_stage(str(tmp_path), "combine", [str(tmp_path / "ours.png")])
    st = cp.scan(str(tmp_path))
    assert st.foreign == 1


# --- falsy-folder guards -----------------------------------------------------
# Path("") normalizes to Path("."), so a falsy folder must be guarded explicitly
# or save() would write ".animaforge/progress.json" into the process CWD
# (this exact bug shipped in caption_policy.scan() in Task 2 and was caught in
# review — do not reintroduce it here).

def test_falsy_folder_readers_return_their_empty_value(tmp_path):
    for folder in ("", None):
        assert cm.load(folder) == {}
        assert cm.images_dict(folder) is None
        assert cm.reconcile(folder) == {}


def test_falsy_folder_writers_are_no_ops_and_never_touch_cwd(tmp_path):
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        for folder in ("", None):
            cm.save(folder, {"images": {}})
            cm.record_settings(folder, trigger="t", prefix="p", order="o", chain=[])
            cm.mark_stage(folder, "tag", ["a.png"])
            assert not (tmp_path / ".animaforge").exists()
    finally:
        os.chdir(old_cwd)


def test_images_dict_distinguishes_no_manifest_from_manifest_with_no_images(tmp_path):
    """None = no manifest file at all (every caption on disk reads as foreign).
    {} = a manifest exists but has recorded no image entries yet. Collapsing
    these breaks caption_policy's provenance warning."""
    assert cm.images_dict(str(tmp_path)) is None
    cm.record_settings(str(tmp_path), trigger="t", prefix="p", order="o", chain=[])
    assert cm.images_dict(str(tmp_path)) == {}
