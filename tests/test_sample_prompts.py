import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.sample_prompts import (
    collect_keywords,
    build_messages,
    grab_caption_blocks,
    parse_prompts,
    read_dataset_captions,
)


def _make_captioned_dataset(tmp_path, captions: dict):
    """Write image + .txt pairs. captions maps stem -> .txt content ('' = uncaptioned)."""
    for stem, text in captions.items():
        (tmp_path / f"{stem}.png").write_bytes(b"x")
        (tmp_path / f"{stem}.txt").write_text(text, encoding="utf-8")


def test_grab_caption_blocks_returns_verbatim_txt_blocks(tmp_path):
    block_a = "1girl, sarah, red hair, forest. She stands among tall pines, smiling."
    block_b = "1boy, knight, plate armor, castle courtyard at dusk."
    _make_captioned_dataset(tmp_path, {"a": block_a, "b": block_b})
    blocks = grab_caption_blocks(str(tmp_path), 2, rng=random.Random(0))
    assert set(blocks) == {block_a, block_b}  # full NL + tags blocks, verbatim


def test_grab_caption_blocks_skips_empty_captions(tmp_path):
    _make_captioned_dataset(tmp_path, {"a": "a cat on a mat", "b": "   ", "c": ""})
    blocks = grab_caption_blocks(str(tmp_path), 5, rng=random.Random(0))
    assert blocks == ["a cat on a mat"]  # only the non-empty block


def test_grab_caption_blocks_returns_all_when_fewer_than_n(tmp_path):
    _make_captioned_dataset(tmp_path, {"a": "one", "b": "two"})
    blocks = grab_caption_blocks(str(tmp_path), 4, rng=random.Random(0))
    assert sorted(blocks) == ["one", "two"]


def test_grab_caption_blocks_samples_n_of_many(tmp_path):
    caps = {chr(ord("a") + i): f"caption {i}" for i in range(6)}
    _make_captioned_dataset(tmp_path, caps)
    blocks = grab_caption_blocks(str(tmp_path), 3, rng=random.Random(1))
    assert len(blocks) == 3
    assert len(set(blocks)) == 3  # no duplicates
    assert all(b in caps.values() for b in blocks)


def test_grab_caption_blocks_zero_or_missing(tmp_path):
    _make_captioned_dataset(tmp_path, {"a": "one"})
    assert grab_caption_blocks(str(tmp_path), 0) == []
    assert grab_caption_blocks("C:/definitely/not/here", 3) == []


def test_collect_keywords_counts_and_orders():
    caps = [
        "mychar, red hair, forest, smiling",
        "mychar, red hair, city, smiling",
        "mychar, red hair, beach",
    ]
    kw = collect_keywords(caps, top_n=5, trigger="mychar")
    assert kw[0] == "red hair"          # most frequent (3)
    assert "mychar" not in kw           # trigger excluded
    assert "smiling" in kw


def test_collect_keywords_drops_boilerplate():
    caps = ["masterpiece, best quality, a cat", "best quality, a cat"]
    kw = collect_keywords(caps, top_n=10)
    assert "masterpiece" not in kw
    assert "best quality" not in kw
    assert "a cat" in kw


def test_collect_keywords_dedups_within_caption():
    # the same token repeated in one caption counts once
    caps = ["sunset, sunset, sunset", "sunset"]
    kw = collect_keywords(caps, top_n=5)
    assert kw == ["sunset"]


def test_build_messages_excludes_trigger_instruction():
    msgs = build_messages(["red hair", "forest"], trigger="mychar", lora_type="character", n=3)
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert "red hair" in user
    assert "3" in user
    assert "character" in user
    assert "trigger" in user.lower()


def test_parse_prompts_strips_numbering_and_quotes():
    text = '1. "a girl in a forest"\n2) a girl at the beach\n- a girl in a city\n\n'
    out = parse_prompts(text, n=3)
    assert out == ["a girl in a forest", "a girl at the beach", "a girl in a city"]


def test_parse_prompts_limits_to_n():
    text = "one\ntwo\nthree\nfour"
    assert parse_prompts(text, n=2) == ["one", "two"]


def test_read_dataset_captions_prefers_tags(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "a.tags").write_text("red hair, forest", encoding="utf-8")
    (tmp_path / "a.txt").write_text("should be ignored", encoding="utf-8")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "b.txt").write_text("a cat", encoding="utf-8")
    caps = read_dataset_captions(str(tmp_path))
    assert "red hair, forest" in caps
    assert "a cat" in caps
    assert "should be ignored" not in caps


def test_read_dataset_captions_missing_folder():
    assert read_dataset_captions("C:/definitely/not/here") == []


def test_build_messages_features_characters_when_present():
    msgs = build_messages(["forest", "armor"], trigger="trig", n=5, characters=["sarah", "knight"])
    user = msgs[1]["content"]
    assert "sarah" in user and "knight" in user
    assert "exactly 5" in user
    assert "Do NOT include any trigger word" in user  # trigger still auto-added


def test_build_messages_without_characters_keeps_no_name_rule():
    msgs = build_messages(["forest"], trigger="trig", n=3)
    user = msgs[1]["content"]
    assert "or character name" in user  # the no-character-name fallback rule
