import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import characters as ch
from core.workflow import dataset_state, naming_state, caption_state


def _img(folder: Path, name: str, caption: str = ""):
    """Create a dummy image + its .txt sidecar with the given caption text."""
    (folder / name).write_bytes(b"\x89PNG\r\n")  # enough to be a real file
    (folder / name).with_suffix(".txt").write_text(caption, encoding="utf-8")


# ---- dataset_state ----

def test_dataset_state_empty_folder(tmp_path):
    s = dataset_state(str(tmp_path))
    assert s["images"] == 0
    assert s["done"] is False


def test_dataset_state_nonexistent_folder():
    s = dataset_state("C:/no/such/folder/anywhere")
    assert s["images"] == 0
    assert s["done"] is False


def test_dataset_state_counts_images(tmp_path):
    _img(tmp_path, "a.png")
    _img(tmp_path, "b.png")
    s = dataset_state(str(tmp_path))
    assert s["images"] == 2
    assert s["done"] is True


# ---- naming_state ----

def test_naming_state_always_optional(tmp_path):
    assert naming_state(str(tmp_path))["optional"] is True


def test_naming_state_no_roster(tmp_path):
    s = naming_state(str(tmp_path))
    assert s["named"] == 0
    assert s["done"] is False


def test_naming_state_with_roster(tmp_path):
    data = ch.DatasetCharacters(roster=[ch.Character(token="kyrie"),
                                        ch.Character(token="billy")])
    ch.save(str(tmp_path), data)
    s = naming_state(str(tmp_path))
    assert s["named"] == 2
    assert s["done"] is True
    assert s["optional"] is True


# ---- caption_state ----

def test_caption_state_empty_folder(tmp_path):
    s = caption_state(str(tmp_path))
    assert s["images"] == 0
    assert s["captioned"] == 0
    assert s["done"] is False


def test_caption_state_no_captions(tmp_path):
    _img(tmp_path, "a.png", caption="")
    _img(tmp_path, "b.png", caption="   ")  # whitespace-only counts as empty
    s = caption_state(str(tmp_path))
    assert s["images"] == 2
    assert s["captioned"] == 0
    assert s["done"] is False


def test_caption_state_partial(tmp_path):
    _img(tmp_path, "a.png", caption="a cat")
    _img(tmp_path, "b.png", caption="")
    _img(tmp_path, "c.png", caption="")
    s = caption_state(str(tmp_path))
    assert s["images"] == 3
    assert s["captioned"] == 1
    assert s["done"] is False


def test_caption_state_fully_captioned(tmp_path):
    _img(tmp_path, "a.png", caption="a cat")
    _img(tmp_path, "b.png", caption="a dog")
    s = caption_state(str(tmp_path))
    assert s["images"] == 2
    assert s["captioned"] == 2
    assert s["done"] is True
