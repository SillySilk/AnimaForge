"""Opt-in, quality-neutral low-VRAM training recipe + its in-memory session state.

Low-VRAM mode is a crutch for GPUs that cannot otherwise fit Anima LoRA training. Every
preset keeps quality identical (bf16, 1024 resolution, effective batch 4) and trades only
speed, via micro-batching + gradient accumulation and CPU block-swap. See
docs/superpowers/specs/2026-06-24-lowvram-and-settings-gear-design.md.

The current config lives in a process-global that starts None and is NEVER persisted — it
applies only after the user explicitly enables + acknowledges it, and only for that session.
"""

# target VRAM (GB) -> (micro_batch, grad_accum, blocks_to_swap). Effective batch == 4.
_PRESETS = {
    16: (4, 1, 0),
    12: (1, 4, 8),
    10: (1, 4, 16),
    8: (1, 4, 24),
}

# Anima DiT has 28 blocks; enable_block_swap asserts blocks_to_swap <= 26.
MAX_BLOCKS_TO_SWAP = 26


def recipe_for(target_gb: int) -> dict:
    """Quality-neutral recipe for a VRAM target. fp8 is always off here (opt-in elsewhere)."""
    micro, accum, swap = _PRESETS.get(int(target_gb), _PRESETS[16])
    return {"micro_batch": micro, "grad_accum": accum, "blocks_to_swap": swap, "fp8_base": False}


# ---- in-memory, non-persistent session state -----------------------------
_current = None  # type: dict | None


def set_current(cfg) -> None:
    """Activate low-VRAM mode for this session (or clear with None)."""
    global _current
    _current = dict(cfg) if cfg else None


def get_current():
    """The active low-VRAM config, or None when off (the default)."""
    return _current


def clear() -> None:
    global _current
    _current = None
