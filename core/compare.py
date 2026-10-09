"""Caption-style comparison: two batch runs of one set, identical except caption style.

Both runs caption with the full tag -> describe chain, so the folder holds both the
.nl and .tags sidecars and each run builds its own .txt from the same prose and
tags. Both preview the same prompts with the same seeds. The pair is recorded in
sets/_comparisons.json so the side-by-side view can find it again.
"""
import copy
import json
import time
from pathlib import Path

from core import models, sets
from core.caption_policy import KEEP, OVERWRITE
from core.dataset_manager import NL_EXT, SUPPORTED_EXTENSIONS, TAGS_EXT

STYLE_SLUG = {"prose": "prose", "tags": "tags", "prose+tags": "prose-tags"}
STYLE_LABEL = {"prose": "Prose", "tags": "Tags", "prose+tags": "Prose + tags"}
RECORDS = "_comparisons"


def _sidecar(img: Path, ext: str) -> str:
    try:
        return img.with_suffix(ext).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def sidecars_complete(folder: str) -> bool:
    """Every image has a non-empty .nl and .tags sidecar."""
    p = Path(folder or "")
    if not p.is_dir():
        return False
    images = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    return bool(images) and all(_sidecar(i, NL_EXT) and _sidecar(i, TAGS_EXT) for i in images)


def plan(rd, other_style: str, prompts: list) -> tuple:
    """(run A in the current style, run B in `other_style`, the record)."""
    style = rd.caption_style
    if other_style not in models.CAPTION_STYLES:
        raise ValueError(f"unknown caption style {other_style!r}")
    if other_style == style:
        raise ValueError("Pick a caption style different from the current one.")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    group = f"{rd.lora_name}-{stamp}"
    # The first run writes both sidecars; when the folder already has them all,
    # keeping them means both runs share the prose that is already there.
    first_policy = KEEP if sidecars_complete(rd.dataset_folder) else OVERWRITE
    runs = []
    for s, policy in ((style, first_policy), (other_style, KEEP)):
        run = copy.deepcopy(rd)
        run.lora_name = f"{rd.lora_name}-{STYLE_SLUG[s]}"
        run.caption_style = s
        run.caption_policy = policy
        run.compare_group = group
        run.sample_prompts = list(prompts)
        run.sample_enabled = True
        runs.append(run)
    entry = {
        "id": group, "created": stamp, "set": rd.lora_name, "model": rd.model,
        "dataset_folder": rd.dataset_folder, "prompts": list(prompts),
        "runs": [{"name": r.lora_name, "style": r.caption_style, "output_dir": r.output_dir}
                 for r in runs],
    }
    return runs[0], runs[1], entry


def _path(root=None) -> Path:
    return sets.sets_dir(root) / f"{RECORDS}.json"


def load_all(root=None) -> list:
    """Every recorded comparison, newest first."""
    try:
        data = json.loads(_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return list(reversed(data)) if isinstance(data, list) else []


def record(entry: dict, root=None) -> None:
    data = list(reversed(load_all(root)))
    data = [e for e in data if e.get("id") != entry["id"]] + [entry]
    _path(root).write_text(json.dumps(data, indent=2), encoding="utf-8")


def find(group_id: str, root=None):
    return next((e for e in load_all(root) if e.get("id") == group_id), None)


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _kohya_finals(run_dir: Path, name: str) -> dict:
    """{prompt index: image} from the highest epoch sd-scripts sampled."""
    import re
    pattern = re.compile(rf"^{re.escape(name)}_e(\d+)_(\d+)_.*\.png$")
    by_epoch = {}
    folder = run_dir / "sample"
    if folder.is_dir():
        for f in folder.iterdir():
            m = pattern.match(f.name)
            if m:
                by_epoch.setdefault(int(m.group(1)), {})[int(m.group(2))] = str(f)
    return by_epoch[max(by_epoch)] if by_epoch else {}


def _onetrainer_finals(run_dir: Path) -> dict:
    """{prompt index: image} from the shipped file's final render (samples.json order)."""
    from core.engines.onetrainer import render
    dest = Path(render.destination(str(run_dir)))
    if not dest.is_dir():
        return {}
    images = sorted((f for f in dest.rglob("*") if f.suffix.lower() in _IMAGE_SUFFIXES),
                    key=lambda f: str(f))
    return {i: str(f) for i, f in enumerate(images)}


def final_previews(entry: dict, output_base: str, statuses: dict) -> list:
    """One row per prompt: {"prompt", "a", "b", "status": {"a", "b"}}. A side with no
    images is None; its status comes from the batch (done / failed), else pending."""
    from core import paths
    engine = models.get(entry.get("model") or models.DEFAULT_MODEL).engine
    sides = []
    for run in entry["runs"][:2]:
        base = paths.from_portable(run.get("output_dir") or output_base)
        run_dir = Path(paths.run_output_dir(base, run["name"]))
        finals = (_onetrainer_finals(run_dir) if engine == models.ONETRAINER
                  else _kohya_finals(run_dir, run["name"]))
        sides.append((finals, statuses.get(run["name"], "pending")))
    rows = []
    for i, prompt in enumerate(entry.get("prompts") or []):
        rows.append({"prompt": prompt,
                     "a": sides[0][0].get(i), "b": sides[1][0].get(i),
                     "status": {"a": sides[0][1], "b": sides[1][1]}})
    return rows
