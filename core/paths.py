"""Per-run output path helpers.

Every training run gets its own folder so the LoRA, its `sample/` previews, logs,
configs, and resumable state live together and never bleed across runs (sd-scripts
hardcodes samples to `{output_dir}/sample`, so isolating previews means isolating the
output dir). See docs/superpowers/specs/2026-06-24-train-workflow-and-name-cast-polish-design.md.
"""
import re
from pathlib import Path

# Characters illegal in Windows filenames (plus path separators), collapsed to "_".
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_name(lora_name: str) -> str:
    """Return a filesystem-safe folder name for a LoRA name (trims trailing dots/spaces)."""
    safe = _ILLEGAL.sub("_", (lora_name or "").strip())
    return safe.rstrip(". ")


def run_output_dir(base_dir: str, lora_name: str) -> str:
    """Per-run output folder `{base_dir}/{sanitized lora_name}`.

    Falls back to `base_dir` when either input is empty so callers stay safe before a
    name/output is set.
    """
    safe = sanitize_name(lora_name)
    if not base_dir or not safe:
        return base_dir or ""
    return str(Path(base_dir) / safe)
