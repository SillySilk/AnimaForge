import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import caption_policy as cp


def _img(folder: Path, stem: str, txt=None, tags=None, nl=None):
    (folder / f"{stem}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    if txt is not None:
        (folder / f"{stem}.txt").write_text(txt, encoding="utf-8")
    if tags is not None:
        (folder / f"{stem}.tags").write_text(tags, encoding="utf-8")
    if nl is not None:
        (folder / f"{stem}.nl").write_text(nl, encoding="utf-8")


def test_scan_buckets_images(tmp_path):
    _img(tmp_path, "a", txt="a caption")             # captioned
    _img(tmp_path, "b", tags="1girl")                # partial
    _img(tmp_path, "c")                              # untouched
    _img(tmp_path, "d", txt="   ")                   # whitespace .txt is NOT captioned
    st = cp.scan(str(tmp_path))
    assert st.total == 4
    assert [Path(p).stem for p in st.captioned] == ["a"]
    assert [Path(p).stem for p in st.partial] == ["b"]
    assert sorted(Path(p).stem for p in st.untouched) == ["c", "d"]


def test_foreign_counts_captions_with_no_manifest_entry(tmp_path):
    _img(tmp_path, "a", txt="ours")
    _img(tmp_path, "b", txt="theirs")
    st = cp.scan(str(tmp_path), manifest_images={"a.png": {"combine": "done"}})
    assert st.foreign == 1          # only b.png is unaccounted for


def test_foreign_is_total_when_no_manifest(tmp_path):
    _img(tmp_path, "a", txt="theirs")
    assert cp.scan(str(tmp_path), manifest_images=None).foreign == 1


def test_images_for_overwrite_is_everything(tmp_path):
    _img(tmp_path, "a", txt="a caption")
    _img(tmp_path, "b")
    st = cp.scan(str(tmp_path))
    assert len(cp.images_for(st, cp.OVERWRITE)) == 2


def test_images_for_keep_skips_captioned(tmp_path):
    _img(tmp_path, "a", txt="a caption")
    _img(tmp_path, "b", tags="1girl")
    _img(tmp_path, "c")
    st = cp.scan(str(tmp_path))
    got = sorted(Path(p).stem for p in cp.images_for(st, cp.KEEP))
    assert got == ["b", "c"]


def test_images_for_rejects_ask(tmp_path):
    import pytest
    st = cp.scan(str(tmp_path))
    with pytest.raises(ValueError):
        cp.images_for(st, cp.ASK)


def test_has_conflict(tmp_path):
    _img(tmp_path, "a")
    assert not cp.has_conflict(cp.scan(str(tmp_path)))
    _img(tmp_path, "b", txt="x")
    assert cp.has_conflict(cp.scan(str(tmp_path)))


def test_scan_empty_or_none_folder_returns_empty_state_not_cwd(tmp_path):
    """Path("") is Path("."), so a falsy folder must be guarded BEFORE the is_dir check
    or scan() silently reports the process CWD's files as dataset images."""
    import os
    (tmp_path / "decoy.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        for folder in ("", None):
            st = cp.scan(folder)
            assert st.total == 0
            assert st.captioned == [] and st.partial == [] and st.untouched == []
    finally:
        os.chdir(old_cwd)


def test_scan_missing_path_returns_empty_state(tmp_path):
    assert cp.scan(str(tmp_path / "nope")).total == 0


def test_scan_file_path_returns_empty_state(tmp_path):
    f = tmp_path / "a.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert cp.scan(str(f)).total == 0


def test_scan_includes_gif_like_dataset_manager(tmp_path):
    from core.dataset_manager import SUPPORTED_EXTENSIONS
    assert ".gif" in SUPPORTED_EXTENSIONS
    (tmp_path / "a.gif").write_bytes(b"GIF89a")
    assert cp.scan(str(tmp_path)).total == 1


def test_nonempty_survives_a_non_utf8_sidecar(tmp_path):
    (tmp_path / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "a.txt").write_bytes(b"\xff\xfe invalid utf-8")
    st = cp.scan(str(tmp_path))          # must not raise
    assert st.total == 1
