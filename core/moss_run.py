"""Launch-time checks for a MOSS-Image run (pure; no Qt).

Shared by the batch queue and the Train tab so both refuse the same way:
missing/wrong model files are named, a GPU under 12 GB is blocked, and the
VRAM tier's memory flags are picked from the real card.
"""
from pathlib import Path

from core import model_check

# (label, role) per slot, in (dit, text encoder, vae) order.
_SLOTS = (
    ("MOSS-Image DiT", model_check.ROLE_MOSS_DIT),
    ("MOSS text encoder (Qwen3.5-9B)", model_check.ROLE_MOSS_TE),
    ("MOSS VAE (FLUX.2)", model_check.ROLE_FLUX2_VAE),
)


def files_problem(dit_path: str, te_path: str, vae_path: str) -> str:
    """"" when all three MOSS files exist and are the right kind, else a plain
    message that names the label and the file path."""
    for (label, role), path in zip(_SLOTS, (dit_path, te_path, vae_path)):
        if not path:
            return f"{label} is not set. Go to the Setup tab."
        if not Path(path).is_file():
            return f"{label} file is missing: {path}"
        problem = model_check.check_model_file(path, role)
        if problem:
            return f"{label}: {problem}"
    return ""


def launch_plan() -> tuple:
    """(memory_flags, warnings) for the GPU in this machine.

    Raises moss_memory.MossVramError (str = the user message) below 12 GB.
    An unprobeable GPU uses the 12 GB tier and adds a warning; never blocks.
    """
    from core import gpu_check, moss_memory
    vram = gpu_check.total_vram_mb()
    tier, flags = moss_memory.tier_for(vram)
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3))
    except Exception:
        ram_gb = None
    warnings = moss_memory.preflight_warnings(
        tier, ram_gb, moss_memory.has_triton(), vram is not None)
    return flags, warnings
