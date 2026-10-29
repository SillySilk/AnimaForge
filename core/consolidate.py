"""Merge several names for one character into one (Characters tab → Consolidate).

A set built from several sources often calls one character two things -- "Marge"
and "Marge Simpson" -- and a LoRA learns those as two names, each seen half as
often. Consolidating rewrites both places a name lives:

  file names   NAME_SERIAL_CATEGORY drives the captions, so every file whose NAME
               holds one of the old names is renamed to the new one (group shots
               keep their other subjects), re-serialized in its new group. Caption
               sidecars travel with it.
  caption text every .txt / .nl / .tags in the set has the old names replaced, whole
               words only, any case.

Trainer cache files beside the pictures (`<stem>_<...>.npz`) are handled too: the
image ones are renamed with their picture, and the text-encoder ones (`*_te.npz`)
of every picture whose text changed are deleted, because the trainer reuses a
cached caption without re-reading the text and would keep training the old name.
"""
import re
import uuid
from pathlib import Path

from core import naming
from core.dataset_manager import NL_EXT, SUPPORTED_EXTENSIONS, TAGS_EXT

_TEXT_EXTS = (".txt", NL_EXT, TAGS_EXT)


def _images(folder: Path) -> list:
    return sorted(f for f in folder.iterdir()
                  if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS)


def _caches(folder: Path, stem: str) -> list:
    return [f for f in folder.iterdir()
            if f.is_file() and f.suffix == ".npz" and f.name.startswith(stem + "_")]


def name_pattern(old_names, new_name: str):
    """Whole-word, any-case match of any old name. The new name is matched too
    (and so left as it is), longest first, so "Marge" -> "Marge Simpson" never
    turns an existing "Marge Simpson" into "Marge Simpson Simpson"."""
    names = {n.strip() for n in old_names if n.strip()} | {new_name.strip()}
    alts = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alts})(?!\w)", re.IGNORECASE)


def consolidate(folder: str, old_names, new_name: str) -> dict:
    """Rename `old_names` to `new_name` across the set in `folder`. Returns
    {"renamed": files renamed, "rewritten": text files changed,
     "caches_removed": text-encoder caches deleted}."""
    base = Path(folder)
    new_name = (new_name or "").strip()
    if not base.is_dir():
        raise ValueError(f"Dataset folder not found: {folder}")
    if not new_name or any(c in new_name for c in "_-"):
        raise ValueError("The new name can't be empty or contain '_' or '-' "
                         "(file names use them to separate fields and characters).")
    old = {n.strip().lower() for n in old_names if n.strip()}
    if not old:
        raise ValueError("Pick at least one character to consolidate.")

    # ---- 1. rename files: everything to a temp name first, so no final name can
    # collide with a file that has not moved yet.
    moves = []
    for img in _images(base):
        parsed = naming.parse_name(img.stem)
        if not parsed or not any(s.lower() in old for s in parsed["subjects"]):
            continue
        subjects = []
        for s in parsed["subjects"]:
            s = new_name if s.lower() in old else s
            if s.lower() not in (x.lower() for x in subjects):
                subjects.append(s)
        group = "-".join(subjects)
        if group != "-".join(parsed["subjects"]):      # already the new name: leave it
            moves.append((img, group, parsed["category"]))

    serials = {}
    for img in _images(base):
        parsed = naming.parse_name(img.stem)
        if parsed and img not in {m[0] for m in moves}:
            key = "-".join(parsed["subjects"]).lower()
            serials[key] = max(serials.get(key, 0), int(parsed["serial"]))

    staged = []
    for img, group, category in moves:
        tmp = f"__af_{uuid.uuid4().hex}"
        caches = _caches(base, img.stem)
        naming.rename_image(str(base), img.name, tmp + img.suffix)
        staged.append((tmp, img.suffix, img.stem, caches, group, category))
    for tmp, suffix, old_stem, caches, group, category in staged:
        key = group.lower()
        serials[key] = serials.get(key, 0) + 1
        stem = f"{group}_{serials[key]:03d}_{category}"
        naming.rename_image(str(base), tmp + suffix, stem + suffix)
        for c in caches:
            c.rename(base / (stem + c.name[len(old_stem):]))

    # ---- 2. rewrite caption text
    pattern = name_pattern(old_names, new_name)
    rewritten = 0
    removed = 0
    for img in _images(base):
        changed = False
        for ext in _TEXT_EXTS:
            f = img.with_suffix(ext)
            if not f.is_file():
                continue
            text = f.read_text(encoding="utf-8")
            new_text = pattern.sub(new_name, text)
            if new_text != text:
                f.write_text(new_text, encoding="utf-8")
                rewritten += 1
                changed = True
        if changed:
            for c in _caches(base, img.stem):
                if c.name.endswith("_te.npz"):
                    c.unlink()
                    removed += 1
    return {"renamed": len(moves), "rewritten": rewritten, "caches_removed": removed}
