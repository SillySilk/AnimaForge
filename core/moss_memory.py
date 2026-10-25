"""MOSS-Image memory tiers and preflight warnings (pure; no Qt).

The tier is picked from total VRAM (MiB). Flag keys are the sd-scripts arg
names consumed by generate_configs(memory_flags=...).
"""
import importlib

TIER_24, TIER_16, TIER_12 = "24", "16", "12"

MIN_VRAM_MB = 11000
_TIER_24_MB = 23000
_TIER_16_MB = 15000
_RAM_RECOMMENDED_GB = 32

_BLOCK_MSG = "MOSS-Image needs a GPU with at least 12 GB of VRAM."
_TRITON_MSG = "INT8 will run slower without triton-windows — re-run install.bat."
_UNKNOWN_GPU_MSG = ("Couldn't read the GPU's memory; using the safest MOSS "
                    "memory setting (12 GB tier).")


class MossVramError(Exception):
    """Raised when the GPU is below the 12 GB floor; str(e) is the user message."""


_FLAGS_24 = {"convrot_int8_text_encoder": True}
_FLAGS_16 = {"convrot_int8": True, "convrot_int8_text_encoder": True}
_FLAGS_12 = {**_FLAGS_16, "blocks_to_swap": 28, "block_swap_h2d_only": True}


def tier_for(total_vram_mb):
    """Return (tier, memory_flags). None (unprobeable GPU) -> the 12 GB tier."""
    if total_vram_mb is None:
        return TIER_12, dict(_FLAGS_12)
    if total_vram_mb >= _TIER_24_MB:
        return TIER_24, dict(_FLAGS_24)
    if total_vram_mb >= _TIER_16_MB:
        return TIER_16, dict(_FLAGS_16)
    if total_vram_mb >= MIN_VRAM_MB:
        return TIER_12, dict(_FLAGS_12)
    raise MossVramError(_BLOCK_MSG)


def has_triton() -> bool:
    try:
        importlib.import_module("triton")
        return True
    except Exception:
        return False


def preflight_warnings(tier, total_ram_gb, has_triton, vram_known):
    """Non-blocking warnings to show before a MOSS run."""
    out = []
    if not vram_known:
        out.append(_UNKNOWN_GPU_MSG)
    if tier == TIER_12 and total_ram_gb is not None and total_ram_gb < _RAM_RECOMMENDED_GB:
        out.append(
            "12 GB MOSS training swaps model blocks into system RAM; "
            f"32 GB of RAM is recommended (found {total_ram_gb:g} GB)."
        )
    if not has_triton:
        out.append(_TRITON_MSG)
    return out
