import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.model_locations import detect_model_files, guess_model_scan_dir
from test_model_check import (
    FLUX2_VAE_KEYS, MOSS_DIT_KEYS, MOSS_TE_KEYS, MOSS_TE_W4A8_KEYS,
    DIT_KEYS, TE_KEYS, VAE_KEYS, WRAPPED_DIT_KEYS, write_safetensors,
)


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


# --- detect_model_files: layout- and name-independent detection -------------
#
# The old Setup auto-detect hardcoded the A1111/Forge tree (text_encoder/, VAE/,
# Stable-diffusion/) AND exact filenames. The official circlestone-labs/Anima
# download is the ComfyUI tree (split_files/text_encoders/, .../vae/,
# .../diffusion_models/), so it matched only the VAE — 'vae' vs 'VAE' happens to
# collide on case-insensitive Windows. These tests pin both layouts and renames.


def _comfy_tree(base: Path):
    """The layout you get by downloading circlestone-labs/Anima as published."""
    sf = base / "models"
    write_safetensors(sf / "diffusion_models" / "anima-base-v1.0.safetensors", DIT_KEYS)
    write_safetensors(sf / "text_encoders" / "qwen_3_06b_base.safetensors", TE_KEYS)
    write_safetensors(sf / "vae" / "qwen_image_vae.safetensors", VAE_KEYS)
    return sf


def _forge_tree(base: Path):
    sf = base / "models"
    write_safetensors(sf / "Stable-diffusion" / "anima-base-v1.0.safetensors", DIT_KEYS)
    write_safetensors(sf / "text_encoder" / "qwen_3_06b_base.safetensors", TE_KEYS)
    write_safetensors(sf / "VAE" / "qwen_image_vae.safetensors", VAE_KEYS)
    return sf


def test_detects_comfyui_layout(tmp_path):
    # The regression: this used to find the VAE and nothing else.
    root = _comfy_tree(tmp_path)
    got = detect_model_files(root)
    assert Path(got["dit"]).name == "anima-base-v1.0.safetensors"
    assert Path(got["text_encoder"]).name == "qwen_3_06b_base.safetensors"
    assert Path(got["vae"]).name == "qwen_image_vae.safetensors"


def test_detects_forge_layout(tmp_path):
    root = _forge_tree(tmp_path)
    got = detect_model_files(root)
    assert all(got[r] for r in ("dit", "text_encoder", "vae"))


def test_detects_raw_split_files_download(tmp_path):
    # Scan folder pointed straight at an unpacked HF download.
    write_safetensors(tmp_path / "split_files" / "diffusion_models" / "anima-base-v1.0.safetensors", DIT_KEYS)
    write_safetensors(tmp_path / "split_files" / "text_encoders" / "qwen_3_06b_base.safetensors", TE_KEYS)
    write_safetensors(tmp_path / "split_files" / "vae" / "qwen_image_vae.safetensors", VAE_KEYS)
    got = detect_model_files(tmp_path)
    assert all(got[r] for r in ("dit", "text_encoder", "vae"))


def test_detects_renamed_files(tmp_path):
    # Real filenames off the maintainer's own Forge install — none match the
    # canonical names, so filename-only matching found nothing.
    sf = tmp_path / "models"
    write_safetensors(sf / "Stable-diffusion" / "anima_baseV10.safetensors", DIT_KEYS)
    write_safetensors(sf / "text_encoder" / "TE Qwen3-0.6B base (Anima).safetensors", TE_KEYS)
    write_safetensors(sf / "VAE" / "VAE Qwen (Anima + PiD).safetensors", VAE_KEYS)
    got = detect_model_files(sf)
    assert Path(got["text_encoder"]).name.startswith("TE Qwen3")
    assert Path(got["vae"]).name.startswith("VAE Qwen")
    assert Path(got["dit"]).name == "anima_baseV10.safetensors"


def test_never_picks_a_wrapped_checkpoint_as_the_dit(tmp_path):
    # 'anima' in the name but ComfyUI-wrapped weights: picking it produced an
    # instant exit-code-1 run. Prefer the real one regardless of sort order.
    sf = tmp_path / "models" / "Stable-diffusion"
    write_safetensors(sf / "aaa_animaMerge_v1.safetensors", WRAPPED_DIT_KEYS)
    write_safetensors(sf / "zzz_anima_real.safetensors", DIT_KEYS)
    got = detect_model_files(tmp_path)
    assert Path(got["dit"]).name == "zzz_anima_real.safetensors"


def test_reports_nothing_rather_than_guessing(tmp_path):
    (tmp_path / "models" / "Lora").mkdir(parents=True)
    write_safetensors(tmp_path / "models" / "Lora" / "some_lora.safetensors", ["lora_unet_x.weight"])
    got = detect_model_files(tmp_path)
    assert got == {"dit": "", "text_encoder": "", "vae": ""}


def test_missing_scan_root_is_empty_not_an_error(tmp_path):
    assert detect_model_files(tmp_path / "nope") == {"dit": "", "text_encoder": "", "vae": ""}
    assert detect_model_files("") == {"dit": "", "text_encoder": "", "vae": ""}

def _moss_tree(base: Path):
    """Forge Neo folder names (diffusion_models / text_encoder / VAE), plus decoys."""
    m = base / "models"
    write_safetensors(m / "diffusion_models" / "moss_image_sft_bf16.safetensors", MOSS_DIT_KEYS)
    write_safetensors(m / "diffusion_models" / "moss_image_base_bf16.safetensors", MOSS_DIT_KEYS)
    write_safetensors(m / "text_encoder" / "qwen_3_06b_base.safetensors", TE_KEYS)
    write_safetensors(m / "text_encoder" / "TE (MOSS-Image).safetensors", MOSS_TE_W4A8_KEYS)
    write_safetensors(m / "text_encoder" / "qwen3.5_9b_int8_convrot.safetensors", MOSS_TE_KEYS)
    write_safetensors(m / "VAE" / "VAE (Klein + ERNIE + MOSS).safetensors", FLUX2_VAE_KEYS)
    write_safetensors(m / "VAE" / "flux2-vae.safetensors", FLUX2_VAE_KEYS)
    write_safetensors(m / "VAE" / "qwen_image_vae.safetensors", VAE_KEYS)
    return m


def test_moss_detection_prefers_base_and_the_supported_te(tmp_path):
    got = detect_model_files(_moss_tree(tmp_path), model="moss")
    assert Path(got["dit"]).name == "moss_image_base_bf16.safetensors"
    assert Path(got["text_encoder"]).name == "qwen3.5_9b_int8_convrot.safetensors"
    assert Path(got["vae"]).name == "flux2-vae.safetensors"


def test_moss_detection_never_picks_the_small_qwen_or_w4a8(tmp_path):
    m = tmp_path / "models"
    write_safetensors(m / "text_encoder" / "qwen_3_06b_base.safetensors", TE_KEYS)
    write_safetensors(m / "text_encoder" / "TE (MOSS-Image).safetensors", MOSS_TE_W4A8_KEYS)
    assert detect_model_files(m, model="moss")["text_encoder"] == ""


def test_moss_detection_accepts_comfyui_folder_names(tmp_path):
    m = tmp_path / "models"
    write_safetensors(m / "diffusion_models" / "moss_image_base_bf16.safetensors", MOSS_DIT_KEYS)
    write_safetensors(m / "text_encoders" / "qwen3.5_9b_int8_convrot.safetensors", MOSS_TE_KEYS)
    write_safetensors(m / "vae" / "flux2-vae.safetensors", FLUX2_VAE_KEYS)
    got = detect_model_files(m, model="moss")
    assert all(got[k] for k in ("dit", "text_encoder", "vae"))


def test_anima_detection_ignores_moss_files(tmp_path):
    got = detect_model_files(_moss_tree(tmp_path))
    assert got["dit"] == ""
    assert Path(got["text_encoder"]).name == "qwen_3_06b_base.safetensors"
    assert Path(got["vae"]).name == "qwen_image_vae.safetensors"
