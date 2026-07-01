from pathlib import Path

from core.paths import run_output_dir


def find_saved_state(output_dir: str, lora_name: str):
    """Return the path to the newest saved training-state folder for this LoRA, or None.

    sd-scripts names state folders '{name}-state' (last) and '{name}-{epoch:06d}-state'.

    Runs now write to a per-run folder ({output_dir}/{lora_name}); we look there first,
    then fall back to the legacy flat layout ({output_dir}) so older interrupted runs
    still resume.
    """
    run_dir = run_output_dir(output_dir, lora_name)
    state = _scan_for_state(run_dir, lora_name)
    if state is not None:
        return state
    if run_dir != output_dir:
        return _scan_for_state(output_dir, lora_name)
    return None


def _scan_for_state(output_dir: str, lora_name: str):
    out = Path(output_dir)
    if not out.is_dir() or not lora_name:
        return None
    candidates = []
    for p in out.iterdir():
        if not p.is_dir():
            continue
        n = p.name
        if n == f"{lora_name}-state" or (
            n.startswith(f"{lora_name}-") and n.endswith("-state")
        ):
            candidates.append(p)
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(newest)
