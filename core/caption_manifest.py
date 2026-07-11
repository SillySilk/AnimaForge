"""Caption progress, written into the dataset folder so the work travels with it.

`<dataset>/.animaforge/progress.json` records what the sidecars cannot:

  * Provenance -- a .txt with no manifest entry was not written by AnimaForge.
  * The settings the captions were built with (trigger, prefix, order, chain).

Written at STAGE boundaries, not per image: per-image truth for tag/describe
already lives on disk as .tags/.nl. `reconcile()` lets the disk win.

A falsy `folder` (None or "") is guarded everywhere below: `Path("")` silently
normalizes to `Path(".")`, so an unguarded `_path()` would resolve into the
process CWD instead of failing loudly. Readers return their empty value
(`{}` / `None`); writers become no-ops. Never let a write escape into the CWD.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from core.dataset_manager import NL_EXT, TAGS_EXT

MANIFEST_REL = Path(".animaforge") / "progress.json"
VERSION = 1

# stage -> the sidecar that proves it ran. `combine` writes .txt.
_STAGE_ARTIFACT = {"tag": TAGS_EXT, "describe": NL_EXT, "combine": ".txt"}


def _path(folder: str):
    """The manifest path for `folder`, or None when `folder` is falsy.

    None must be checked explicitly by every caller -- it means "no folder was
    given", not "use the current directory".
    """
    if not folder:
        return None
    return Path(folder) / MANIFEST_REL


def load(folder: str) -> dict:
    """The manifest dict, or {} when absent, corrupt, or `folder` is falsy. Never raises."""
    p = _path(folder)
    if p is None:
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def images_dict(folder: str):
    """The per-image stage dict, or None when there is no manifest at all.

    None and {} mean different things to caption_policy: None = no manifest, so every
    caption on disk is foreign; {} = a manifest exists but has no image entries yet.
    A falsy `folder` also reads as None -- there is nothing to be a manifest of.
    """
    p = _path(folder)
    if p is None or not p.is_file():
        return None
    return load(folder).get("images", {})


def save(folder: str, data: dict) -> None:
    """Write `data` to the manifest. No-op when `folder` is falsy."""
    p = _path(folder)
    if p is None:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    data["version"] = VERSION
    data["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def record_settings(folder: str, trigger: str, prefix: str, order: str, chain: list) -> None:
    """No-op when `folder` is falsy."""
    if not folder:
        return
    d = load(folder)
    d.update({"trigger": trigger, "prefix": prefix, "order": order, "chain": list(chain)})
    d.setdefault("images", {})
    save(folder, d)


def mark_stage(folder: str, stage: str, images: list) -> None:
    """Record `stage` as done for each image path in `images`. No-op when `folder` is falsy."""
    if not folder:
        return
    d = load(folder)
    imgs = d.setdefault("images", {})
    for path in images:
        imgs.setdefault(Path(path).name, {})[stage] = "done"
    save(folder, d)


def reconcile(folder: str) -> dict:
    """Correct the manifest against what is actually on disk.

    A stage whose proving sidecar is gone reverts to "pending". A falsy `folder`
    yields {} (load()'s empty value) with nothing to reconcile.
    """
    d = load(folder)
    if not folder:
        return d
    base = Path(folder)
    for name, stages in d.get("images", {}).items():
        img = base / name
        for stage in list(stages):
            ext = _STAGE_ARTIFACT.get(stage)
            if ext is None:
                continue
            proof = img.with_suffix(ext)
            if not proof.is_file():
                stages[stage] = "pending"
    return d
