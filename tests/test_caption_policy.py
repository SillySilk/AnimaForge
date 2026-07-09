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
