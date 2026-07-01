"""Hard filename naming convention for AnimaForge datasets (v2).

A file stem is `NAME_SERIAL_CATEGORY`: three underscore-separated fields. The underscore is
structural and never appears inside a field.
  - NAME:     the subject(s). Spaces allowed inside one subject; multiple subjects are
              joined by a hyphen '-' (each hyphen starts a new subject). Case-insensitive.
  - SERIAL:   a zero-padded 3-digit number (001+), counting images within a group.
  - CATEGORY: Character / Object / Style (case-insensitive in, canonical out). One per project.

The captioner reads the NAME subjects (hyphen-split) as trigger tokens; the SERIAL is
organizational only. Training files are disposable, so auto_format renames freely.
Pure stdlib so the stdlib-only caption scripts can import it.
"""
from __future__ import annotations

import re
import uuid
from collections import Counter, OrderedDict
from pathlib import Path

CATEGORIES = ("Character", "Object", "Style")
_CANON = {c.lower(): c for c in CATEGORIES}
_SIDECAR_EXTS = (".tags", ".nl", ".txt")  # caption sidecars that travel with an image
_SERIAL_RE = re.compile(r"^\d{3,}$")       # zero-padded, 3 or more digits


def parse_name(stem: str):
    """Parse a stem into {'subjects': [...], 'serial': str, 'category': Canonical} or None.

    Requires exactly 3 underscore fields: NAME _ SERIAL _ CATEGORY. NAME splits on '-' into
    subjects (spaces kept). SERIAL must be 3+ digits. CATEGORY is matched case-insensitively
    and returned canonical.
    """
    parts = (stem or "").split("_")
    if len(parts) != 3:
        return None
    name, serial, category = parts[0].strip(), parts[1].strip(), parts[2].strip()
    if not name or not _SERIAL_RE.match(serial):
        return None
    canon = _CANON.get(category.lower())
    if canon is None:
        return None
    subjects = [s.strip() for s in name.split("-") if s.strip()]
    if not subjects:
        return None
    return {"subjects": subjects, "serial": serial, "category": canon}


def project_category(stems) -> str | None:
    """The dominant valid category across the set, or None if nothing parses."""
    counts = Counter()
    for stem in stems:
        parsed = parse_name(stem)
        if parsed:
            counts[parsed["category"]] += 1
    return counts.most_common(1)[0][0] if counts else None


def _reason(parsed, category) -> str:
    if parsed is None:
        return "does not match NAME_SERIAL_CATEGORY"
    return f"wrong category ({parsed['category']} in a {category} project)"


def validate_folder(image_names) -> dict:
    """Validate filenames against the convention.

    Returns {"category": project_category|None, "valid": [name,...],
             "invalid": [{"name","reason"}, ...]}. Fixing is bulk via auto_format (no
    per-file suggestion field).
    """
    stems = {n: Path(n).stem for n in image_names}
    category = project_category(stems.values())
    valid, invalid = [], []
    for name in image_names:
        parsed = parse_name(stems[name])
        if parsed and (category is None or parsed["category"] == category):
            valid.append(name)
        else:
            invalid.append({"name": name, "reason": _reason(parsed, category)})
    return {"category": category, "valid": valid, "invalid": invalid}


def assignments_from_names(image_names):
    """Derive (tokens, assignments) from conforming filenames for captioning anchors.

    tokens = ordered unique subjects across valid files; assignments[name] =
    {"present": [subjects], "oneoffs": []}. Non-conforming files are skipped.
    """
    tokens, assignments = [], {}
    for name in image_names:
        parsed = parse_name(Path(name).stem)
        if not parsed:
            continue
        for subj in parsed["subjects"]:
            if subj not in tokens:
                tokens.append(subj)
        assignments[name] = {"present": list(parsed["subjects"]), "oneoffs": []}
    return tokens, assignments


def bundles_from_names(image_names):
    """Group conforming images into read-only cast bundles for the Characters view.

    Returns {"category", "solo", "combined", "needs_naming"}:
      - solo:     [{"name": subject, "images": [name,...]}] for single-subject files
                  (grouped by subject, first-appearance order).
      - combined: [{"name": "A + B", "subjects": [A,B,...], "images": [name,...]}] for
                  multi-subject files (grouped by ordered subject tuple, first-appearance order).
      - needs_naming: names that don't conform / are off-category (validate_folder's invalid).
    Subjects that only ever appear in multi-subject files get no solo bundle.
    """
    res = validate_folder(image_names)
    solo, combined = OrderedDict(), OrderedDict()
    for name in res["valid"]:
        parsed = parse_name(Path(name).stem)
        subjects = parsed["subjects"]
        if len(subjects) == 1:
            solo.setdefault(subjects[0], []).append(name)
        else:
            combined.setdefault(tuple(subjects), []).append(name)
    return {
        "category": res["category"],
        "solo": [{"name": s, "images": imgs} for s, imgs in solo.items()],
        "combined": [{"name": " + ".join(k), "subjects": list(k), "images": v}
                     for k, v in combined.items()],
        "needs_naming": [d["name"] for d in res["invalid"]],
    }


def rename_image(folder, old_name: str, new_name: str) -> str:
    """Rename an image and its caption sidecars (.tags/.nl/.txt) on disk. No-op if equal;
    FileNotFoundError if source missing; FileExistsError if target exists."""
    if old_name == new_name:
        return new_name
    folder = Path(folder)
    src, dst = folder / old_name, folder / new_name
    if not src.is_file():
        raise FileNotFoundError(old_name)
    if dst.exists():
        raise FileExistsError(new_name)
    src.rename(dst)
    old_stem, new_stem = Path(old_name).stem, Path(new_name).stem
    for ext in _SIDECAR_EXTS:
        s, t = folder / (old_stem + ext), folder / (new_stem + ext)
        if s.is_file() and not t.exists():
            s.rename(t)
    return new_name


def write_characters_from_names(folder, image_names):
    """Write animaforge_characters.json roster+assignments from the filenames (the source of
    truth for captioning). Preserves the existing style_anchor."""
    from core import characters as ch
    tokens, assignments = assignments_from_names(image_names)
    data = ch.load(folder)
    data.roster = [ch.Character(t) for t in tokens]
    data.assignments = assignments
    ch.save(folder, data)
    return data


def _derive_name(stem: str):
    """Best-effort NAME (subject string) from a near-miss stem: strip a trailing category
    field and/or a trailing serial field if present. Returns the NAME, or None when the
    remainder is ambiguous (a stray underscore inside the name) or empty."""
    parts = [p.strip() for p in (stem or "").split("_")]
    if parts and parts[-1].lower() in _CANON:
        parts = parts[:-1]
    if parts and _SERIAL_RE.match(parts[-1] or ""):
        parts = parts[:-1]
    if len(parts) != 1:
        return None
    return parts[0] or None


def auto_format(folder, image_names, category: str | None = None):
    """Rename files to NAME_SERIAL_CATEGORY on disk: canonical project category, sequential
    001+ serials per subject-NAME group. Two-pass (temp names then finals) so re-serializing
    can never collide. Files whose NAME can't be derived are left alone. Returns the list of
    (old, new) renames actually applied.
    """
    folder = Path(folder)
    category = _CANON.get((category or "").lower()) or project_category(
        Path(n).stem for n in image_names)
    if category is None:
        return []

    groups = OrderedDict()
    suffixes = {}
    for name in image_names:
        derived = _derive_name(Path(name).stem)
        if derived is None:
            continue
        groups.setdefault(derived, []).append(name)
        suffixes[name] = Path(name).suffix

    desired = {}
    for group_name, members in groups.items():
        for i, old in enumerate(members, start=1):
            new = f"{group_name}_{i:03d}_{category}{suffixes[old]}"
            if new != old:
                desired[old] = new
    if not desired:
        return []

    temp = {}
    for old in desired:
        tmp = f"__af_{uuid.uuid4().hex}{Path(old).suffix}"
        rename_image(str(folder), old, tmp)
        temp[old] = tmp
    applied = []
    for old, new in desired.items():
        rename_image(str(folder), temp[old], new)
        applied.append((old, new))
    return applied
