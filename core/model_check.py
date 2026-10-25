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

# MOSS-Image files. Key shapes read off the real files (see the Task 1 findings):
#   moss_image_base_bf16.safetensors   379 tensors: `rope.bands`, `final_linear_video.*`,
#                                      `layers.N.proj.{qkv,o,ffn_up_gate,ffn_down}.weight`
#   qwen3.5_9b_int8_convrot            1409 tensors, everything under `model.language_model.*`
#                                      (hybrid linear-attention: `...linear_attn.A_log`) + `lm_head.*`
#   "TE (MOSS-Image)" W4A8 GPTQ        same layout, plus `*.weight_codebook` / `weight_s_channel` /
#                                      `weight_s_rel` quantisation tensors; sd-scripts cannot load it
#   flux2-vae.safetensors              251 tensors: encoder./decoder. plus a `bn.running_mean`
#                                      batch-norm that no Flux1/SDXL/Qwen-Image VAE carries
ROLE_MOSS_DIT = "moss_dit"
ROLE_MOSS_TE = "moss_text_encoder"
ROLE_MOSS_TE_W4A8 = "moss_text_encoder_w4a8"
ROLE_FLUX2_VAE = "flux2_vae"

LABELS = {
    ROLE_DIT: "Anima DiT checkpoint",
    ROLE_TEXT_ENCODER: "Qwen3 text encoder",
    ROLE_VAE: "Qwen-Image VAE",
    ROLE_DIT_WRAPPED: "ComfyUI-wrapped Anima checkpoint",
    ROLE_VAE_OTHER: "VAE from a different model family (Flux/SDXL/SD)",
    ROLE_MOSS_DIT: "MOSS-Image DiT",
    ROLE_MOSS_TE: "MOSS text encoder (Qwen3.5-9B)",
    ROLE_MOSS_TE_W4A8: "W4A8 (GPTQ) Qwen3.5 text encoder",
    ROLE_FLUX2_VAE: "FLUX.2 VAE",
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
    if any(k == "rope.bands" or k.startswith("final_linear_video.") for k in keys):
        return ROLE_MOSS_DIT
    if any(k.startswith("model.language_model.") for k in keys):
        # Qwen3.5: every weight sits under model.language_model.*. The W4A8 GPTQ build
        # adds codebook tensors; sd-scripts only reads the int8 / bf16 files.
        if any(k.endswith(".weight_codebook") for k in keys):
            return ROLE_MOSS_TE_W4A8
        return ROLE_MOSS_TE
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
        if "bn.running_mean" in keys:
            return ROLE_FLUX2_VAE
        if any(k.startswith("encoder.downsamples.") for k in keys) or (
            any(k.startswith("conv1.") for k in keys) and any(k.startswith("conv2.") for k in keys)
        ):
            return ROLE_VAE
        return ROLE_VAE_OTHER
    return ""


_W4A8_MESSAGE = (
    "“{name}” is the W4A8 (GPTQ) build of the text encoder, and W4A8 is not supported "
    "for training. Use the int8 file instead, e.g. qwen3.5_9b_int8_convrot.safetensors."
)


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
    if expected_role == ROLE_MOSS_TE:
        low = p.name.lower()
        if actual == ROLE_MOSS_TE_W4A8 or "w4a8" in low or "gptq" in low:
            return _W4A8_MESSAGE.format(name=p.name)
    if not actual or actual == expected_role:
        return ""
    want = LABELS.get(expected_role, expected_role)
    if expected_role == ROLE_MOSS_DIT:
        return (
            f"“{p.name}” is set as the MOSS-Image DiT, but its contents are a "
            f"{LABELS.get(actual, actual)}. Pick moss_image_base_bf16.safetensors "
            f"(from huggingface.co/OpenMOSS-Team/MOSS-Image-Base) in Setup."
        )
    if expected_role == ROLE_MOSS_TE:
        return (
            f"“{p.name}” is set as the MOSS text encoder, but its contents are a "
            f"{LABELS.get(actual, actual)}. MOSS-Image needs the Qwen3.5-9B text "
            f"encoder, e.g. qwen3.5_9b_int8_convrot.safetensors."
        )
    if expected_role == ROLE_FLUX2_VAE:
        return (
            f"“{p.name}” is set as the MOSS VAE, but its contents are a "
            f"{LABELS.get(actual, actual)}. MOSS-Image needs the FLUX.2 VAE, "
            f"e.g. flux2-vae.safetensors."
        )
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
