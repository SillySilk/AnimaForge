import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from scripts import gen_assets


def test_generate_all_writes_valid_images(tmp_path: Path):
    paths = gen_assets.generate_all(tmp_path)
    expected = {
        "emblem.png", "emblem.svg", "icon.png", "icon.ico",
        "hero_forge.png", "bg_embers.png", "panel_metal.png",
        "nav/home.png", "nav/setup.png", "nav/dataset.png",
        "nav/characters.png", "nav/train.png", "nav/batch.png",
    }
    rel = {p.relative_to(tmp_path).as_posix() for p in paths}
    assert expected <= rel, f"missing: {expected - rel}"
    for name in expected:
        f = tmp_path / name
        assert f.exists() and f.stat().st_size > 0, name
    for name in expected:
        if name.endswith(".png"):
            Image.open(tmp_path / name).verify()


def test_generate_all_is_idempotent(tmp_path: Path):
    gen_assets.generate_all(tmp_path)
    paths2 = gen_assets.generate_all(tmp_path)
    assert (tmp_path / "emblem.png").exists()
    assert len(paths2) == 13
