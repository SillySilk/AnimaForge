"""Workflow readiness — the single source of truth for the Load → Name → Caption →
Train progress rail and the Train tab's readiness summary.

Pure, stdlib-only logic over a dataset folder so both the rail (ui/progress_rail.py)
and the Train tab read step-completion from one place. Naming is OPTIONAL by
contract (`naming_state` always reports `optional=True`); no caller may treat it as a
precondition for training. See
docs/superpowers/specs/2026-06-24-workflow-progress-rail-and-train-streamline-design.md.
"""
from core import characters as ch
from core.dataset_manager import scan_folder


def dataset_state(folder: str) -> dict:
    """Image-load step. `done` once the folder holds at least one supported image."""
    images = len(scan_folder(folder)) if folder else 0
    return {"images": images, "done": images > 0}


def naming_state(folder: str) -> dict:
    """Character-naming step — always optional. `named` = roster size; `done` when ≥1."""
    named = len(ch.load(folder).roster) if folder else 0
    return {"named": named, "done": named > 0, "optional": True}


def caption_state(folder: str) -> dict:
    """Captioning step. A caption counts only when its .txt sidecar is non-empty after
    strip (scan_folder auto-creates empty sidecars, so existence alone means nothing).
    `done` when every image is captioned."""
    items = scan_folder(folder) if folder else []
    images = len(items)
    captioned = sum(1 for d in items if d.get("caption", "").strip())
    return {"captioned": captioned, "images": images,
            "done": images > 0 and captioned == images}
