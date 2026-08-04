import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.model_check import (
    ROLE_DIT, ROLE_TEXT_ENCODER, ROLE_VAE,
    check_model_file, identify, read_safetensors_keys,
)


def write_safetensors(path: Path, keys) -> Path:
    """Write a header-only safetensors file carrying `keys`. Enough for identify()."""
    header = {k: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for k in keys}
    blob = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00" * 4)
    return path


# Key shapes taken from real files on disk (anima_baseV10, the Qwen3 TE, the Qwen VAE)
# and from a ComfyUI-wrapped Anima merge — see core/model_check.py.
DIT_KEYS = ["net.blocks.0.adaln_modulation_cross_attn.1.weight", "net.x_embedder.weight"]
WRAPPED_DIT_KEYS = ["model.diffusion_model.blocks.0.adaln_modulation_cross_attn.1.weight"]
TE_KEYS = ["model.embed_tokens.weight", "model.layers.0.mlp.down_proj.weight"]
VAE_KEYS = ["conv1.weight", "conv2.weight",
            "encoder.downsamples.0.residual.0.gamma", "decoder.conv1.weight"]
# A Flux/SDXL VAE: same encoder./decoder. split, different architecture.
OTHER_VAE_KEYS = ["encoder.conv_in.weight", "encoder.conv_out.weight", "decoder.conv_in.weight"]


def test_reads_keys_from_header(tmp_path):
    p = write_safetensors(tmp_path / "m.safetensors", DIT_KEYS)
    assert read_safetensors_keys(p) == DIT_KEYS


def test_metadata_is_not_a_tensor_key(tmp_path):
    p = tmp_path / "m.safetensors"
    header = {"__metadata__": {"format": "pt"},
              "net.x_embedder.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    blob = json.dumps(header).encode("utf-8")
    p.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 4)
    assert read_safetensors_keys(p) == ["net.x_embedder.weight"]


def test_identifies_each_role(tmp_path):
    assert identify(write_safetensors(tmp_path / "a.safetensors", DIT_KEYS)) == ROLE_DIT
    assert identify(write_safetensors(tmp_path / "b.safetensors", TE_KEYS)) == ROLE_TEXT_ENCODER
    assert identify(write_safetensors(tmp_path / "c.safetensors", VAE_KEYS)) == ROLE_VAE


def test_wrapped_dit_is_not_accepted_as_dit(tmp_path):
    # A ComfyUI-wrapped Anima merge: right architecture, 'model.diffusion_model.' prefix.
    # anima_utils.load_anima_model strips only 'net.', so this raises
    # "Missing keys in checkpoint" and the run dies at exit code 1.
    p = write_safetensors(tmp_path / "waiANIMA_v10.safetensors", WRAPPED_DIT_KEYS)
    assert identify(p) != ROLE_DIT
    msg = check_model_file(p, ROLE_DIT)
    assert msg
    assert "model.diffusion_model." in msg


def test_flux_vae_is_not_accepted_as_the_qwen_vae(tmp_path):
    # Every autoencoder has encoder./decoder. keys, so "has both" wrongly accepted
    # a Flux VAE sitting in the same VAE/ folder.
    p = write_safetensors(tmp_path / "VAE Flux1 (Chroma).safetensors", OTHER_VAE_KEYS)
    assert identify(p) != ROLE_VAE
    msg = check_model_file(p, ROLE_VAE)
    assert msg and "qwen_image_vae.safetensors" in msg


def test_good_files_pass_their_own_role(tmp_path):
    assert check_model_file(write_safetensors(tmp_path / "a.safetensors", DIT_KEYS), ROLE_DIT) == ""
    assert check_model_file(write_safetensors(tmp_path / "b.safetensors", TE_KEYS), ROLE_TEXT_ENCODER) == ""
    assert check_model_file(write_safetensors(tmp_path / "c.safetensors", VAE_KEYS), ROLE_VAE) == ""


def test_swapped_files_are_named_in_the_message(tmp_path):
    # The classic mix-up: the VAE pasted into the DiT field.
    vae = write_safetensors(tmp_path / "qwen_image_vae.safetensors", VAE_KEYS)
    msg = check_model_file(vae, ROLE_DIT)
    assert "Qwen-Image VAE" in msg and "Anima DiT" in msg


def test_unreadable_file_never_blocks_the_run(tmp_path):
    # Best-effort: a file we can't parse must not be reported as wrong. Only a
    # positive identification of a DIFFERENT role is worth stopping Start for.
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a safetensors file at all")
    assert read_safetensors_keys(junk) is None
    assert identify(junk) == ""
    assert check_model_file(junk, ROLE_DIT) == ""


def test_missing_file_is_not_our_error_to_report(tmp_path):
    assert check_model_file(tmp_path / "nope.safetensors", ROLE_DIT) == ""


def test_absurd_header_length_is_rejected(tmp_path):
    p = tmp_path / "bad.safetensors"
    p.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    assert read_safetensors_keys(p) is None
