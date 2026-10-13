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
from core.caption_policy import FILL

STYLE_SLUG = {"prose": "prose", "tags": "tags", "prose+tags": "prose-tags"}
STYLE_LABEL = {"prose": "Prose", "tags": "Tags", "prose+tags": "Prose + tags"}
RECORDS = "_comparisons"


def plan(rd, other_style: str, prompts: list) -> tuple:
    """(run A in the current style, run B in `other_style`, the record)."""
    style = rd.caption_style
    if other_style not in models.CAPTION_STYLES:
        raise ValueError(f"unknown caption style {other_style!r}")
    if other_style == style:
        raise ValueError("Pick a caption style different from the current one.")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    group = f"{rd.lora_name}-{stamp}"
    # Both runs fill only the sidecars an image lacks (existing prose is never
    # re-described) and rebuild every .txt in their own style. Decided here, never
    # by the batch's caption-conflict prompt (ui/batch_tab._resolve_conflicts).
    runs = []
    for s in (style, other_style):
        run = copy.deepcopy(rd)
        # The stamp keeps a later comparison of the same set out of these folders.
        run.lora_name = f"{rd.lora_name}-{STYLE_SLUG[s]}-{stamp[4:13]}"
        run.caption_style = s
        run.caption_policy = FILL
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


def _kohya_finals(run_dir: Path, name: str, epoch=None) -> tuple:
    """(epoch, {prompt index: image}): the shipped epoch's samples when run safety
    shipped an earlier snapshot and they exist, else the highest epoch sampled."""
    import re
    pattern = re.compile(rf"^{re.escape(name)}_e(\d+)_(\d+)_.*\.png$")
    by_epoch = {}
    folder = run_dir / "sample"
    if folder.is_dir():
        for f in folder.iterdir():
            m = pattern.match(f.name)
            if m:
                by_epoch.setdefault(int(m.group(1)), {})[int(m.group(2))] = str(f)
    if not by_epoch:
        return None, {}
    chosen = epoch if epoch in by_epoch else max(by_epoch)
    return chosen, by_epoch[chosen]


def _onetrainer_finals(run_dir: Path) -> dict:
    """{prompt index: image} from the shipped file's final render. The render writes
    prompt i loose into samples/ as "final-<i>"; a prompt whose render failed simply
    has no image. Runs rendered before that kept them in samples/final/"<i> - final"/."""
    import re
    from core.engines.onetrainer import render
    dest = Path(render.destination(str(run_dir)))
    out = render.final_images(str(dest))
    older = dest / "final"
    if out or not older.is_dir():
        return out
    for folder in older.iterdir():
        m = re.match(r"^(\d+) - final$", folder.name)
        if not (m and folder.is_dir()):
            continue
        images = sorted(f for f in folder.iterdir() if f.suffix.lower() in _IMAGE_SUFFIXES)
        if images:
            out[int(m.group(1))] = str(images[0])
    return out


def final_previews(entry: dict, output_base: str, statuses: dict, shipped=None) -> list:
    """One row per prompt: {"prompt", "a", "b", "status": {"a", "b"}, "epoch": {"a", "b"}}.
    A side with no images is None; its status comes from the batch (done / failed),
    else pending. `shipped` maps a run name to the epoch run safety shipped."""
    from core import paths
    engine = models.get(entry.get("model") or models.DEFAULT_MODEL).engine
    sides = []
    for run in entry["runs"][:2]:
        base = paths.from_portable(run.get("output_dir") or output_base)
        run_dir = Path(paths.run_output_dir(base, run["name"]))
        if engine == models.ONETRAINER:
            epoch, finals = None, _onetrainer_finals(run_dir)
        else:
            epoch, finals = _kohya_finals(run_dir, run["name"], (shipped or {}).get(run["name"]))
        sides.append((finals, statuses.get(run["name"], "pending"), epoch))
    rows = []
    for i, prompt in enumerate(entry.get("prompts") or []):
        rows.append({"prompt": prompt,
                     "a": sides[0][0].get(i), "b": sides[1][0].get(i),
                     "status": {"a": sides[0][1], "b": sides[1][1]},
                     "epoch": {"a": sides[0][2], "b": sides[1][2]}})
    return rows
