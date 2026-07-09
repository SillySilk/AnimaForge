import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.dataset_manager import (
    combine_caption,
    combine_all,
    apply_prefix,
    read_sidecar,
    write_sidecar,
    latest_files,
    TAGS_EXT,
    NL_EXT,
)


def test_combine_nl_first():
    out = combine_caption("a girl on a bench", "1girl, bench, outdoors", prefix="mychar")
    assert out == "mychar, a girl on a bench, 1girl, bench, outdoors"


def test_combine_tags_first():
    out = combine_caption("a girl", "1girl, solo", prefix="", order="tags_first")
    assert out == "1girl, solo, a girl"


def test_combine_drops_empty():
    assert combine_caption("", "1girl", prefix="") == "1girl"
    assert combine_caption("a girl", "", prefix="trig") == "trig, a girl"
    assert combine_caption("", "", prefix="") == ""


def test_combine_lead_after_prefix():
    # lead tokens (character name) sit right after the trigger prefix, before nl/tags
    out = combine_caption("a girl", "1girl, solo", prefix="ohwx", order="nl_first", lead="Homer")
    assert out == "ohwx, Homer, a girl, 1girl, solo"


def test_combine_lead_hoisted_when_already_in_tags():
    # a name already buried in the tags is hoisted to the front, not duplicated
    out = combine_caption("", "1girl, homer", prefix="", order="nl_first", lead="homer")
    assert out == "homer, 1girl"


def test_combine_no_lead_is_unchanged():
    out = combine_caption("a girl on a bench", "1girl, bench", prefix="mychar")
    assert out == "mychar, a girl on a bench, 1girl, bench"


def test_sidecar_roundtrip(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    write_sidecar(str(img), NL_EXT, "a description")
    write_sidecar(str(img), TAGS_EXT, "1girl, solo")
    assert read_sidecar(str(img), NL_EXT) == "a description"
    assert read_sidecar(str(img), TAGS_EXT) == "1girl, solo"
    assert read_sidecar(str(img), ".missing") == ""


def test_combine_all_writes_txt(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    write_sidecar(str(img), NL_EXT, "a cat sitting")
    write_sidecar(str(img), TAGS_EXT, "cat, sitting")
    written, errors = combine_all(str(tmp_path), prefix="trig", order="nl_first")
    assert written == 1 and errors == 0
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "trig, a cat sitting, cat, sitting"


def test_combine_all_missing_sidecar(tmp_path):
    img = tmp_path / "b.png"
    img.write_bytes(b"fake")
    write_sidecar(str(img), TAGS_EXT, "dog")  # no .nl
    written, errors = combine_all(str(tmp_path))
    assert written == 1 and errors == 0
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "dog"


def test_combine_all_applies_character_anchors(tmp_path):
    from core import characters as C
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    write_sidecar(str(img), NL_EXT, "a red-haired girl kneeling")
    write_sidecar(str(img), TAGS_EXT, "1girl, kneeling")

    data = C.DatasetCharacters(
        roster=[C.Character("asuka", "red hair")],
        style_anchor="@mystyle",
        assignments={"a.png": {"present": ["asuka"], "oneoffs": []}},
    )
    C.save(str(tmp_path), data)

    combine_all(str(tmp_path), prefix="", order="nl_first")
    txt = (tmp_path / "a.txt").read_text(encoding="utf-8")
    # character name leads, followed by the style anchor, then the caption body
    assert txt.startswith("asuka, @mystyle, ")
    assert txt == "asuka, @mystyle, a red-haired girl kneeling, 1girl, kneeling"


def test_combine_all_no_characters_file_is_unchanged(tmp_path):
    img = tmp_path / "b.png"
    img.write_bytes(b"fake")
    write_sidecar(str(img), NL_EXT, "a dog")
    write_sidecar(str(img), TAGS_EXT, "dog, outdoors")

    combine_all(str(tmp_path), prefix="", order="nl_first")
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "a dog, dog, outdoors"


def test_latest_files_missing_dir():
    assert latest_files("C:/no/such/dir") == []


def test_latest_files_newest_first(tmp_path):
    import time
    for name in ("a.png", "b.png", "c.png"):
        (tmp_path / name).write_bytes(b"x")
        time.sleep(0.01)
    got = latest_files(str(tmp_path), n=2)
    assert [p.split("\\")[-1].split("/")[-1] for p in got] == ["c.png", "b.png"]


def test_apply_prefix_skips_duplicate(tmp_path):
    txt = tmp_path / "c.txt"
    txt.write_text("some caption", encoding="utf-8")
    m, s, e = apply_prefix(str(tmp_path), prefix_text="", trigger_word="mychar")
    assert (m, s, e) == (1, 0, 0)
    assert txt.read_text(encoding="utf-8") == "mychar, some caption"
    # second run should skip
    m, s, e = apply_prefix(str(tmp_path), prefix_text="", trigger_word="mychar")
    assert (m, s, e) == (0, 1, 0)


# ---- empty-caption guard (pre-training) ----

def _img(tmp_path, name, caption=None):
    p = tmp_path / name
    p.write_bytes(b"fake")
    if caption is not None:
        p.with_suffix(".txt").write_text(caption, encoding="utf-8")
    return p


def test_find_empty_captions_missing_and_whitespace(tmp_path):
    from core.dataset_manager import find_empty_captions
    _img(tmp_path, "a.png")                    # no .txt at all
    _img(tmp_path, "b.png", "  \n\t ")         # whitespace-only
    _img(tmp_path, "c.png", "a real caption")  # fine
    (tmp_path / "notes.txt").write_text("stray txt, no image", encoding="utf-8")
    empty = find_empty_captions(str(tmp_path))
    assert [Path(p).name for p in empty] == ["a.png", "b.png"]
    assert find_empty_captions(str(tmp_path / "nope")) == []


def test_fill_empty_captions_writes_trigger_only_where_empty(tmp_path):
    from core.dataset_manager import find_empty_captions, fill_empty_captions
    _img(tmp_path, "a.png")
    _img(tmp_path, "b.png", "")
    _img(tmp_path, "c.png", "keep me")
    assert fill_empty_captions(str(tmp_path), "  mychar  ") == 2
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "mychar"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "mychar"
    assert (tmp_path / "c.txt").read_text(encoding="utf-8") == "keep me"
    assert find_empty_captions(str(tmp_path)) == []


def test_fill_empty_captions_noop_without_trigger(tmp_path):
    from core.dataset_manager import fill_empty_captions
    _img(tmp_path, "a.png")
    assert fill_empty_captions(str(tmp_path), "   ") == 0
    assert not (tmp_path / "a.txt").exists()


# ---- undersized-image guard (bucketing division-by-zero, pre-training) ----

def _png(tmp_path, name, w, h):
    from PIL import Image
    p = tmp_path / name
    Image.new("RGB", (w, h), (128, 128, 128)).save(p)
    return p


def test_find_undersized_images_flags_short_side_under_step(tmp_path):
    # A side < bucket_reso_steps (64) makes kohya floor that dim to 0 and then
    # divide by it -> "division by zero" when aspect-ratio bucketing is on.
    from core.dataset_manager import find_undersized_images
    _png(tmp_path, "tiny.png", 48, 48)     # both sides too small
    _png(tmp_path, "strip.png", 1024, 40)  # thin banner: short side 40
    _png(tmp_path, "edge_ok.png", 64, 900)  # exactly 64 -> safe
    _png(tmp_path, "good.png", 896, 1152)  # normal
    (tmp_path / "notes.txt").write_text("stray", encoding="utf-8")

    found = find_undersized_images(str(tmp_path))
    # returns (path, width, height), sorted by filename, only the offenders
    assert [(Path(p).name, w, h) for p, w, h in found] == [
        ("strip.png", 1024, 40),
        ("tiny.png", 48, 48),
    ]


def test_find_undersized_images_boundary_and_param(tmp_path):
    from core.dataset_manager import find_undersized_images
    _png(tmp_path, "at_64.png", 64, 200)   # == step: safe
    _png(tmp_path, "below.png", 63, 200)   # one under: flagged
    names = lambda res: [Path(p).name for p, _, _ in res]
    assert names(find_undersized_images(str(tmp_path))) == ["below.png"]
    # honors a custom minimum
    assert names(find_undersized_images(str(tmp_path), min_side=128)) == [
        "at_64.png", "below.png",
    ]


def test_find_undersized_images_missing_dir(tmp_path):
    from core.dataset_manager import find_undersized_images
    assert find_undersized_images(str(tmp_path / "nope")) == []


def test_combine_all_only_restricts_which_txt_are_written(tmp_path):
    from core.dataset_manager import combine_all
    for stem in ("a", "b"):
        (tmp_path / f"{stem}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (tmp_path / f"{stem}.nl").write_text(f"{stem} prose", encoding="utf-8")
        (tmp_path / f"{stem}.tags").write_text("1girl", encoding="utf-8")
    (tmp_path / "b.txt").write_text("DO NOT TOUCH", encoding="utf-8")

    written, errors = combine_all(str(tmp_path), only=[str(tmp_path / "a.png")])

    assert written == 1 and errors == 0
    assert "a prose" in (tmp_path / "a.txt").read_text(encoding="utf-8")
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "DO NOT TOUCH"


def test_duplicate_stem_names_flags_extension_twins():
    from core.dataset_manager import duplicate_stem_names
    dupes = duplicate_stem_names([
        "/d/hero_001.png", "/d/hero_001.jpg", "/d/Solo.png", "/d/HERO_001.webp",
    ])
    # case-insensitive stem match; each double lists its twins, singles omitted
    assert set(dupes) == {"hero_001.png", "hero_001.jpg", "HERO_001.webp"}
    assert set(dupes["hero_001.png"]) == {"hero_001.jpg", "HERO_001.webp"}
    assert "Solo.png" not in dupes
    assert duplicate_stem_names([]) == {}
