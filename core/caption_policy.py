"""What captioning should do when a dataset folder already holds caption files.

Pure, stdlib-only. `scan()` buckets a folder's images by how far captioning got;
`images_for()` turns a bucket set plus a policy into the exact list of images the
caption chain should process.

ASK is a UI-level policy only: it must be resolved to OVERWRITE or KEEP before it
reaches any runner, so `images_for()` rejects it rather than guessing.
"""
from dataclasses import dataclass
from pathlib import Path

from core.dataset_manager import NL_EXT, SUPPORTED_EXTENSIONS, TAGS_EXT

ASK = "ask"
OVERWRITE = "overwrite"
KEEP = "keep"

_UNSET = object()


@dataclass(frozen=True)
class FolderCaptionState:
    total: int
    captioned: list        # image paths whose .txt is non-empty after strip
    partial: list          # has .tags or .nl, but no usable .txt
    untouched: list        # nothing at all
    foreign: int           # captioned images with no manifest entry -> not ours


def _nonempty(p: Path) -> bool:
    try:
        return bool(p.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeDecodeError):
        return False


def scan(folder: str, manifest_images=_UNSET) -> FolderCaptionState:
    """Bucket every image in `folder`.

    `manifest_images` is the caption manifest's per-image dict (filename -> stage
    dict). Omit it to read the folder's own manifest; None means no manifest, so
    every existing caption is foreign.
    """
    if manifest_images is _UNSET:
        from core.caption_manifest import images_dict
        manifest_images = images_dict(folder)
    if not folder:
        return FolderCaptionState(0, [], [], [], 0)
    d = Path(folder)
    if not d.is_dir():
        return FolderCaptionState(0, [], [], [], 0)
    captioned, partial, untouched = [], [], []
    foreign = 0
    known = manifest_images or {}
    for img in sorted(p for p in d.iterdir()
                      if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS):
        if _nonempty(img.with_suffix(".txt")):
            captioned.append(str(img))
            if img.name not in known:
                foreign += 1
        elif (_nonempty(img.with_suffix(TAGS_EXT))
              or _nonempty(img.with_suffix(NL_EXT))):
            partial.append(str(img))
        else:
            untouched.append(str(img))
    return FolderCaptionState(
        total=len(captioned) + len(partial) + len(untouched),
        captioned=captioned, partial=partial, untouched=untouched, foreign=foreign)


def has_conflict(state: FolderCaptionState) -> bool:
    """True when running the chain would destroy work that is already on disk."""
    return bool(state.captioned)


def images_for(state: FolderCaptionState, policy: str) -> list:
    """The images the caption chain should process under `policy`."""
    if policy == OVERWRITE:
        return list(state.captioned) + list(state.partial) + list(state.untouched)
    if policy == KEEP:
        return list(state.partial) + list(state.untouched)
    raise ValueError(f"policy must be {OVERWRITE!r} or {KEEP!r}, not {policy!r} "
                     "— resolve ASK in the UI before calling")
