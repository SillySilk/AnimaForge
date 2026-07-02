import os
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Sidecar extensions for the non-destructive caption pipeline:
#   .tags  -> booru tags from the WD14 tagger
#   .nl    -> natural-language description from JoyCaption
#   .txt   -> the merged caption that training actually reads
TAGS_EXT = ".tags"
NL_EXT = ".nl"


def duplicate_stem_names(image_paths) -> dict:
    """Map each image filename to the OTHER images sharing its stem (case-insensitive).

    Two files like ``hero_001.png`` + ``hero_001.jpg`` are "doubles": the trainer sees
    both images but they silently share ONE ``.txt`` caption (sidecars key off the
    stem), doubling that picture's exposure. Names without a twin are omitted.
    """
    by_stem: dict = {}
    for p in image_paths:
        p = Path(p)
        by_stem.setdefault(p.stem.lower(), []).append(p.name)
    out = {}
    for names in by_stem.values():
        if len(names) > 1:
            for n in names:
                out[n] = [m for m in names if m != n]
    return out


def scan_folder(path: str) -> list:
    """
    Scan a folder for image files and their associated caption .txt files.

    Returns a list of dicts:
        [{"image_path": str, "txt_path": str, "caption": str}, ...]
    """
    results = []
    folder = Path(path)
    if not folder.is_dir():
        return results

    image_files = sorted(
        [
            f
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda f: f.name.lower(),
    )

    for img_path in image_files:
        txt_path = ensure_txt_file(str(img_path))
        caption = load_caption(txt_path)
        results.append(
            {
                "image_path": str(img_path),
                "txt_path": txt_path,
                "caption": caption,
            }
        )

    return results


def ensure_txt_file(image_path: str) -> str:
    """
    Given an image path, return the path to the corresponding .txt caption file.
    Creates an empty .txt file if it does not exist.
    """
    p = Path(image_path)
    txt_path = p.with_suffix(".txt")
    if not txt_path.exists():
        txt_path.write_text("", encoding="utf-8")
    return str(txt_path)


def load_caption(txt_path: str) -> str:
    """Read and return the content of a caption file."""
    try:
        return Path(txt_path).read_text(encoding="utf-8")
    except (OSError, IOError):
        return ""


def save_caption(txt_path: str, text: str) -> bool:
    """
    Write caption text to the given .txt file.
    Returns True on success, False on failure.
    """
    try:
        Path(txt_path).write_text(text, encoding="utf-8")
        return True
    except (OSError, IOError):
        return False


# ----------------------------------------------------------------------
# Sidecar helpers
# ----------------------------------------------------------------------

def read_sidecar(image_path: str, ext: str) -> str:
    """Read a sidecar file (e.g. '.tags' or '.nl') next to an image. '' if missing."""
    p = Path(image_path).with_suffix(ext)
    try:
        return p.read_text(encoding="utf-8").strip() if p.is_file() else ""
    except (OSError, IOError):
        return ""


def write_sidecar(image_path: str, ext: str, text: str) -> bool:
    """Write a sidecar file next to an image."""
    try:
        Path(image_path).with_suffix(ext).write_text(text, encoding="utf-8")
        return True
    except (OSError, IOError):
        return False


def combine_caption(nl: str, tags: str, prefix: str = "", order: str = "nl_first",
                    lead: str = "") -> str:
    """
    Merge a natural-language caption and booru tags into one caption string.

    Result is comma-joined: [prefix, ] [lead, ] [nl/tags in chosen order].
    Empty pieces are dropped. order is "nl_first" (default) or "tags_first".

    `lead` carries high-priority tokens (per-image character names, then the style anchor)
    that must lead the caption right after the trigger prefix. Lead tokens are de-duplicated
    and any copy already sitting in nl/tags is removed so the token is hoisted to the front
    rather than duplicated — keeping the result idempotent across re-combines.
    """
    nl = (nl or "").strip().strip(",").strip()
    tags = (tags or "").strip().strip(",").strip()
    prefix = (prefix or "").strip().strip(",").strip()
    lead = (lead or "").strip().strip(",").strip()

    lead_tokens, seen = [], set()
    for t in (x.strip() for x in lead.split(",")):
        if t and t.lower() not in seen:
            seen.add(t.lower())
            lead_tokens.append(t)

    def _strip_lead(seg: str) -> str:
        kept = [t for t in (x.strip() for x in seg.split(",")) if t and t.lower() not in seen]
        return ", ".join(kept)

    nl, tags = _strip_lead(nl), _strip_lead(tags)
    body_parts = [tags, nl] if order == "tags_first" else [nl, tags]

    parts = [p for p in ([prefix, ", ".join(lead_tokens)] + body_parts) if p]
    return ", ".join(parts)


def combine_all(folder_path: str, prefix: str = "", order: str = "nl_first",
                apply_anchors: bool = True) -> tuple:
    """
    For every image, merge its .nl and .tags sidecars into the training .txt file.

    When apply_anchors is True and the folder has an animaforge_characters.json, each image's
    explicitly-assigned character tokens and the dataset @-style anchor are forced into the tag
    portion (deterministic safety net, independent of the LLM pass). Harmless no-op otherwise.

    Returns (written_count, error_count).
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        return 0, 0

    chars = None
    if apply_anchors:
        from core import characters as _ch
        chars = _ch.load(folder_path)

    written = 0
    errors = 0
    images = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    for img in images:
        nl = read_sidecar(str(img), NL_EXT)
        tags = read_sidecar(str(img), TAGS_EXT)
        lead = ""
        if chars is not None:
            from core import characters as _ch
            toks = _ch.explicit_tokens_for_image(chars, img.name)
            anchor = (chars.style_anchor or "").strip()
            lead = ", ".join(list(toks) + ([anchor] if anchor else []))
        merged = combine_caption(nl, tags, prefix=prefix, order=order, lead=lead)
        if save_caption(str(img.with_suffix(".txt")), merged):
            written += 1
        else:
            errors += 1
    return written, errors


def apply_prefix(folder_path: str, prefix_text: str, trigger_word: str = "") -> tuple:
    """
    Prepend a prefix (and optional trigger word) to all .txt caption files.

    Skips files that already begin with the prefix to avoid duplication.
    Returns (modified_count, skipped_count, error_count).
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        return 0, 0, 0

    prefix_text = (prefix_text or "").strip().strip(",").strip()
    trigger_word = (trigger_word or "").strip()

    pieces = [p for p in [trigger_word, prefix_text] if p]
    if not pieces:
        return 0, 0, 0
    prefix = ", ".join(pieces) + ", "

    modified = 0
    skipped = 0
    errors = 0

    txt_files = [f for f in folder.iterdir() if f.is_file() and f.suffix == ".txt"]
    for txt_file in txt_files:
        try:
            current = txt_file.read_text(encoding="utf-8")
            if current.strip().startswith(pieces[0]):
                skipped += 1
                continue
            new_content = prefix + current if current.strip() else prefix.rstrip(", ")
            txt_file.write_text(new_content, encoding="utf-8")
            modified += 1
        except (OSError, IOError):
            errors += 1

    return modified, skipped, errors


def find_empty_captions(folder_path: str) -> list:
    """Image paths whose training .txt caption is missing or whitespace-only.

    Training fails on captionless images, so callers gate a run on this (offering a
    trigger-word fill for trigger-only datasets). Sorted by filename for stable output.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        return []
    empty = []
    for f in sorted(folder.iterdir(), key=lambda f: f.name.lower()):
        if not (f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS):
            continue
        txt = f.with_suffix(".txt")
        if not txt.is_file() or not load_caption(str(txt)).strip():
            empty.append(str(f))
    return empty


def fill_empty_captions(folder_path: str, trigger_word: str) -> int:
    """Write the trigger word into every missing/whitespace-only .txt caption.

    Never touches images that already have a caption. Returns files written.
    """
    trigger = (trigger_word or "").strip()
    if not trigger:
        return 0
    written = 0
    for img in find_empty_captions(folder_path):
        if save_caption(str(Path(img).with_suffix(".txt")), trigger):
            written += 1
    return written


def latest_files(folder_path: str, n: int = 6) -> list:
    """Return image paths in a folder, newest by mtime first. Up to n, or all if n is None.
    [] if folder missing."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return []
    imgs = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    imgs.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    if n is None:
        return [str(f) for f in imgs]
    return [str(f) for f in imgs[:n]]


def count_images_in_folder(folder_path: str) -> int:
    """Return the number of supported image files in a folder."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return 0
    return sum(
        1
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
