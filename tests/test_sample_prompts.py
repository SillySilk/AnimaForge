import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.sample_prompts import grab_caption_blocks


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
