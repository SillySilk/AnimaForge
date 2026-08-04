"""Identify an Anima model file by reading its safetensors header.

Setup's three path fields were validated with `Path(p).exists()` alone, so any
readable file passed. Pointing one at the wrong checkpoint therefore got all the
way into the trainer, where `library/anima_utils.py` raises
"Missing keys in checkpoint" and the run dies at exit code 1 within seconds —
surfacing in the UI as nothing but "Preparing…" flipping to "Stopped".

Reading the header is cheap (an 8-byte length plus a small JSON blob; no tensor
data is touched), so the mistake can be caught before launch and named in plain
English instead.

Key shapes below were read off real files rather than recalled:

  anima_baseV10.safetensors        685 tensors, all `net.*`
  waiANIMA_v10Base10.safetensors   685 tensors, all `model.diffusion_model.*`
  Qwen3-0.6B text encoder          310 tensors, `model.embed_tokens.weight`, `model.layers.*`
  Qwen-Image VAE                   194 tensors, `encoder.downsamples.*`, top-level
                                   `conv1`/`conv2`, 5-D conv weights (a 3D/video VAE)
  Flux1 VAE (a near miss)          244 tensors, `encoder.conv_in.*`, 4-D conv weights

`anima_utils.load_anima_model` strips a `net.` prefix and nothing else, which is
why the second one — a genuine Anima merge, just saved in ComfyUI's wrapped
layout — fails despite being the right architecture. It is worth its own message.

The last two matter because every autoencoder has `encoder.*`/`decoder.*` keys, so
"has both" is not enough to call something the Qwen-Image VAE: a Flux or SDXL VAE
sitting in the same folder would satisfy it and then fail in training.
"""
import json
import struct
from pathlib import Path

ROLE_DIT = "dit"
ROLE_TEXT_ENCODER = "text_encoder"
ROLE_VAE = "vae"

# Wrapped Anima DiT: right architecture, wrong key prefix for this trainer.
ROLE_DIT_WRAPPED = "dit_wrapped"
# An autoencoder, but not Anima's — a Flux/SDXL/SD VAE.
ROLE_VAE_OTHER = "vae_other"

LABELS = {
    ROLE_DIT: "Anima DiT checkpoint",
    ROLE_TEXT_ENCODER: "Qwen3 text encoder",
    ROLE_VAE: "Qwen-Image VAE",
    ROLE_DIT_WRAPPED: "ComfyUI-wrapped Anima checkpoint",
    ROLE_VAE_OTHER: "VAE from a different model family (Flux/SDXL/SD)",
}

# A safetensors header is JSON metadata only. Real ones run to a few hundred KB;
# anything past this is a corrupt or non-safetensors file, not something to read.
MAX_HEADER_BYTES = 64 * 1024 * 1024


def read_safetensors_keys(path):
    """Tensor names from a safetensors header, or None if the file isn't one.

    None means "we could not tell" — never "the file is wrong". Callers must not
    turn an unreadable file into a blocking error.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return None
            length = struct.unpack("<Q", raw)[0]
            if not 0 < length <= MAX_HEADER_BYTES:
                return None
            header = json.loads(f.read(length))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(header, dict):
        return None
    return [k for k in header if k != "__metadata__"]


def identify(path) -> str:
    """The role this file actually holds, or "" when it can't be identified."""
    keys = read_safetensors_keys(path)
    if not keys:
        return ""
    if any(k.startswith("net.") for k in keys):
        return ROLE_DIT
    if any(k.startswith("model.diffusion_model.") for k in keys):
        return ROLE_DIT_WRAPPED
    if any(k.startswith("model.embed_tokens") or k.startswith("model.layers.") for k in keys):
        return ROLE_TEXT_ENCODER
    if any(k.startswith("encoder.") for k in keys) and any(k.startswith("decoder.") for k in keys):
        # Anima's VAE is the Qwen-Image one: a 3D/video-style autoencoder with
        # `encoder.downsamples.*` blocks and conv1/conv2 at the top level. A Flux
        # or SDXL VAE has the same encoder./decoder. split but conv_in/conv_out
        # blocks, so it must not pass as Anima's.
        if any(k.startswith("encoder.downsamples.") for k in keys) or (
            any(k.startswith("conv1.") for k in keys) and any(k.startswith("conv2.") for k in keys)
        ):
            return ROLE_VAE
        return ROLE_VAE_OTHER
    return ""


def check_model_file(path, expected_role: str) -> str:
    """"" when `path` is right for `expected_role`, else a plain-English problem.

    Deliberately one-sided: only a positive identification of a *different* role
    produces a message. An unreadable, missing or unrecognized file returns ""
    and the run proceeds, so this can never become the reason a valid setup
    won't start.
    """
    p = Path(path)
    if not p.is_file():
        return ""
    actual = identify(p)
    if not actual or actual == expected_role:
        return ""
    want = LABELS.get(expected_role, expected_role)
    if actual == ROLE_DIT_WRAPPED and expected_role == ROLE_DIT:
        return (
            f"“{p.name}” is an Anima checkpoint saved in ComfyUI's wrapped layout "
            f"(its weights are named “model.diffusion_model.…” instead of “net.…”). "
            f"AnimaForge's trainer only reads the “net.…” layout, so this would stop "
            f"with “Missing keys in checkpoint” before the first step.\n\n"
            f"Use the split_files/diffusion_models copy from "
            f"huggingface.co/circlestone-labs/Anima — e.g. anima-base-v1.0.safetensors."
        )
    if actual == ROLE_VAE_OTHER and expected_role == ROLE_VAE:
        return (
            f"“{p.name}” is a VAE, but not the Qwen-Image VAE that Anima needs "
            f"(it looks like a Flux/SDXL/SD one). Use qwen_image_vae.safetensors "
            f"from split_files/vae in huggingface.co/circlestone-labs/Anima."
        )
    return (
        f"“{p.name}” is set as the {want}, but its contents are a "
        f"{LABELS.get(actual, actual)}. Check the three model paths in Setup — "
        f"training would stop immediately with “Missing keys in checkpoint”."
    )
