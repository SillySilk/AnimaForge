import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.model_locations import guess_model_scan_dir


def _make_models(base: Path):
    models = base / "models"
    (models / "text_encoder").mkdir(parents=True)
    (models / "VAE").mkdir(parents=True)
    (models / "Stable-diffusion").mkdir(parents=True)
    (models / "text_encoder" / "qwen_3_06b_base.safetensors").write_text("x")
    (models / "VAE" / "qwen_image_vae.safetensors").write_text("x")
    (models / "Stable-diffusion" / "anima-base-v1.0.safetensors").write_text("x")
    return models


def test_finds_dir_with_anima_files(tmp_path):
    models = _make_models(tmp_path / "Forge")
    got = guess_model_scan_dir(roots=[str(tmp_path)])
    assert Path(got) == models


def test_finds_one_level_deeper(tmp_path):
    # e.g. <root>/Forge_neo/forge-neo/models
    models = _make_models(tmp_path / "Forge_neo" / "forge-neo")
    got = guess_model_scan_dir(roots=[str(tmp_path)])
    assert Path(got) == models


def test_returns_empty_when_absent(tmp_path):
    (tmp_path / "Forge" / "models").mkdir(parents=True)  # exists but no anima files
    assert guess_model_scan_dir(roots=[str(tmp_path)]) == ""


def test_ignores_non_app_folders(tmp_path):
    # A 'models' dir under a non-SD folder name shouldn't be considered.
    _make_models(tmp_path / "RandomProject")
    assert guess_model_scan_dir(roots=[str(tmp_path)]) == ""
