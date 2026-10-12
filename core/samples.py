"""Group training sample images into per-epoch rounds for the compare view.

sd-scripts names samples ``{name}_e{epoch:06d}_{i:02d}_{ts}{seed}.png`` for
epoch-scheduled renders and ``{name}_{steps:06d}_{i:02d}_...`` for step-scheduled
ones (``sample_at_first`` renders at step 0). Grouping by that tag gives the
side-by-side "which epoch looked best?" comparison. OneTrainer names them
``{ts}-training-sample-{global_step}-{epoch}-{epoch_step}.jpg`` inside one folder
per prompt (``{index} - {prompt start}``). — the user picks the earliest
strong checkpoint by eye (never auto-stopped).
"""

from __future__ import annotations

import re
from pathlib import Path

_EPOCH = re.compile(r"_e(\d{4,})_")
_STEP = re.compile(r"_(\d{4,})_")
_PROMPT_INDEX = re.compile(r"_e?\d{4,}_(\d{1,3})_")
_OT_SAMPLE = re.compile(r"training-sample-(\d+)-(\d+)-\d+")
_OT_PROMPT_DIR = re.compile(r"^(\d+) - ")


def round_key(filename: str):
    """(sort_key, label) for one sample file, or None when it carries no tag."""
    name = Path(filename).name
    m = _OT_SAMPLE.search(name)
    if m:
        step, epoch = int(m.group(1)), int(m.group(2))
        return (1, epoch), ("start" if step == 0 else f"epoch {epoch}")
    m = _EPOCH.search(name)
    if m:
        n = int(m.group(1))
        return (1, n), f"epoch {n}"
    m = _STEP.search(name)
    if m:
        n = int(m.group(1))
        return (0, n), ("start" if n == 0 else f"step {n}")
    return None


def prompt_index(filename: str) -> int | None:
    """The 0-based line index into the sample-prompts file this image was rendered
    from (the ``{i:02d}`` slot in sd-scripts' filename), or None if it can't be found."""
    if _OT_SAMPLE.search(Path(filename).name):
        m = _OT_PROMPT_DIR.match(Path(filename).parent.name)
        return int(m.group(1)) if m else None
    m = _PROMPT_INDEX.search(Path(filename).name)
    return int(m.group(1)) if m else None


def prompt_for_file(filename: str, prompts_file: str) -> str | None:
    """The prompt text that produced `filename`, read from the run's sample-prompts
    file, or None if the index or the file itself isn't available. A OneTrainer
    sample reads its run's samples.json (run/workspace/samples/<prompt>/<file>)
    instead, whatever `prompts_file` says."""
    idx = prompt_index(filename)
    if idx is None:
        return None
    if _OT_SAMPLE.search(Path(filename).name):
        import json
        try:
            run_dir = Path(filename).parents[3]
            entries = json.loads((run_dir / "samples.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, IndexError):
            return None
        return entries[idx].get("prompt") if 0 <= idx < len(entries) else None
    try:
        lines = [ln for ln in Path(prompts_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return None
    return lines[idx] if 0 <= idx < len(lines) else None


def group_by_round(files: list[str]) -> list[tuple[str, list[str]]]:
    """[(label, files)] newest round first; untagged files land in a trailing
    "other" group. Files within a round keep name order (prompt index)."""
    rounds: dict = {}
    other: list[str] = []
    for f in files:
        rk = round_key(f)
        if rk is None:
            other.append(f)
        else:
            rounds.setdefault(rk, []).append(f)
    out = []
    for (key, label) in sorted(rounds, reverse=True):
        out.append((label, sorted(rounds[(key, label)], key=lambda p: Path(p).name)))
    if other:
        out.append(("other", sorted(other, key=lambda p: Path(p).name)))
    return out
