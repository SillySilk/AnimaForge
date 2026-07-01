"""Per-dataset character roster + style anchor for caption refinement.

One JSON file per dataset folder (animaforge_characters.json) records the characters that may
appear in the set (token + recognition description), a dataset-wide @-style anchor, and per-image
assignments. Pure stdlib so scripts/llm_refine_run.py (stdlib + Pillow only) can import it.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

FILENAME = "animaforge_characters.json"


@dataclass
class Character:
    """One named character: a stable token, a recognition-only description, an optional
    detection-only find-hint (never used in the final caption prompt), and a role.

    role is "subject" (a person this LoRA is being trained on) or "label" (a side
    character who is only tagged in captions to keep the subject's likeness clean).
    """
    token: str
    description: str = ""
    hint: str = ""
    role: str = "subject"

    def to_dict(self) -> dict:
        return {"token": self.token, "description": self.description, "hint": self.hint,
                "role": self.role}

    @classmethod
    def from_dict(cls, d: dict) -> "Character":
        role = d.get("role", "subject")
        return cls(token=d.get("token", ""), description=d.get("description", ""),
                   hint=d.get("hint", ""), role=role if role in ("subject", "label") else "subject")


@dataclass
class DatasetCharacters:
    """The full per-folder record: roster, style anchor, and per-image assignments."""
    roster: list = field(default_factory=list)          # list[Character]
    style_anchor: str = ""
    assignments: dict = field(default_factory=dict)      # image_name -> {"present": [tok], "oneoffs": [dict]}

    def to_dict(self) -> dict:
        return {
            "style_anchor": self.style_anchor,
            "roster": [c.to_dict() for c in self.roster],
            "assignments": self.assignments,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetCharacters":
        return cls(
            roster=[Character.from_dict(c) for c in d.get("roster", [])],
            style_anchor=d.get("style_anchor", ""),
            assignments=d.get("assignments", {}) or {},
        )


def path_for(folder: str) -> str:
    return str(Path(folder) / FILENAME)


def load_file(path: str) -> DatasetCharacters:
    p = Path(path)
    if not p.is_file():
        return DatasetCharacters()
    try:
        return DatasetCharacters.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return DatasetCharacters()


def load(folder: str) -> DatasetCharacters:
    return load_file(path_for(folder))


def save_file(path: str, data: DatasetCharacters) -> None:
    Path(path).write_text(json.dumps(data.to_dict(), indent=2), encoding="utf-8")


def save(folder: str, data: DatasetCharacters) -> None:
    save_file(path_for(folder), data)


# ----------------------------------------------------------------------
# Cast resolution
# ----------------------------------------------------------------------

def _resolve_present(data: DatasetCharacters, tokens) -> list:
    """Return roster Characters for the given tokens, preserving order, skipping unknowns."""
    by_token = {c.token: c for c in data.roster}
    return [by_token[t] for t in tokens if t in by_token]


def present_for_image(data: DatasetCharacters, image_name: str) -> list:
    """Cast to send to the vision model for one image.

    No assignment -> the full roster (the model auto-matches whoever it sees).
    Assigned       -> exactly the ticked roster characters + this image's one-offs
                      (an explicitly empty assignment yields []).
    """
    entry = data.assignments.get(image_name)
    if entry is None:
        return list(data.roster)
    present = _resolve_present(data, entry.get("present", []))
    oneoffs = [Character.from_dict(o) for o in entry.get("oneoffs", [])]
    return present + oneoffs


def explicit_tokens_for_image(data: DatasetCharacters, image_name: str) -> list:
    """Tokens that are *confirmed* present (explicit assignment only) — for the deterministic net.

    Unassigned images return [] so the full-roster auto-match default is never forced into tags.
    """
    entry = data.assignments.get(image_name)
    if entry is None:
        return []
    toks = [c.token for c in _resolve_present(data, entry.get("present", []))]
    toks += [o.get("token", "") for o in entry.get("oneoffs", [])]
    return [t for t in toks if t.strip()]


def build_character_block(chars, style_anchor: str = "") -> str:
    """Assemble the <characters>/<style_anchor> prompt text; omit empty blocks."""
    parts = []
    if chars:
        lines = [(f"{c.token}: {c.description.strip()}" if c.description.strip() else c.token)
                 for c in chars]
        parts.append("<characters>\n" + "\n".join(lines) + "\n</characters>")
    if style_anchor.strip():
        parts.append(f"<style_anchor>{style_anchor.strip()}</style_anchor>")
    return "\n".join(parts)


def enforce_anchors_in_tags(tags: str, tokens, style_anchor: str = "") -> str:
    """Guarantee each token and the style anchor appear in a comma-separated tag string.

    Case-insensitive dedup, idempotent, existing order preserved (missing items appended).
    """
    existing = [t.strip() for t in (tags or "").split(",") if t.strip()]
    lower = {t.lower() for t in existing}
    wanted = list(tokens) + ([style_anchor] if style_anchor and style_anchor.strip() else [])
    for tok in wanted:
        tok = (tok or "").strip()
        if tok and tok.lower() not in lower:
            existing.append(tok)
            lower.add(tok.lower())
    return ", ".join(existing)


# ----------------------------------------------------------------------
# Filename-based detection (deterministic: "name_NN.ext" -> character)
# ----------------------------------------------------------------------

_NAME_NUM_RE = re.compile(r"^(?P<name>.*?)[ _\-]*(?P<num>\d+)$")


def parse_character_name(stem: str):
    """Extract a character name from a filename stem of the form 'name_NN'.

    Strips a trailing run of digits (with optional separators) and returns the
    leading name with underscores turned into spaces. Returns None when there is
    no trailing number or no alphabetic name (so generic files are ignored).
    """
    m = _NAME_NUM_RE.match((stem or "").strip())
    if not m:
        return None
    name = m.group("name").strip(" _-").replace("_", " ").strip()
    if not name or not any(ch.isalpha() for ch in name):
        return None
    return name


def characters_from_filenames(image_names):
    """Group images by the character name encoded in their filename.

    Returns (tokens, assignments) where tokens is the ordered unique list of
    detected names and assignments maps each matching image name to
    {"present": [token], "oneoffs": []}. Non-matching files are skipped.
    """
    tokens = []
    assignments = {}
    for name in image_names:
        token = parse_character_name(Path(name).stem)
        if not token:
            continue
        if token not in tokens:
            tokens.append(token)
        assignments[name] = {"present": [token], "oneoffs": []}
    return tokens, assignments


def merge_detected(data: "DatasetCharacters", tokens, assignments) -> "DatasetCharacters":
    """Add any missing roster tokens (keep existing descriptions) and overlay
    the given per-image assignments. Idempotent."""
    existing = {c.token for c in data.roster}
    for t in tokens:
        t = (t or "").strip()
        if t and t not in existing:
            data.roster.append(Character(t))
            existing.add(t)
    for name, entry in (assignments or {}).items():
        data.assignments[name] = {
            "present": [t for t in entry.get("present", []) if str(t).strip()],
            "oneoffs": entry.get("oneoffs", []),
        }
    return data


# ----------------------------------------------------------------------
# Natural-language correction ops (applied to a DatasetCharacters)
# ----------------------------------------------------------------------

def replace_token(data: "DatasetCharacters", frm: str, to: str, image_names=None) -> int:
    """Character Doctor: swap token `frm` -> `to` in per-image assignments.

    Scope: all images when image_names is None, else only those filenames. The target
    token is auto-added to the roster if missing (so the assignment resolves). Present
    lists are de-duplicated while preserving order; one-off tokens are swapped too.
    Pure (mutates `data`); returns the number of images actually changed.
    """
    frm = (frm or "").strip()
    to = (to or "").strip()
    if not frm or not to or frm == to:
        return 0
    if to not in {c.token for c in data.roster}:
        data.roster.append(Character(to))
    scope = set(image_names) if image_names is not None else None
    changed = 0
    for name, entry in data.assignments.items():
        if scope is not None and name not in scope:
            continue
        present = entry.get("present", [])
        swapped = [to if t == frm else t for t in present]
        seen, deduped = set(), []
        for t in swapped:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        oneoffs = entry.get("oneoffs", [])
        new_oneoffs = []
        for o in oneoffs:
            if o.get("token") == frm:
                o = {**o, "token": to}
            new_oneoffs.append(o)
        if deduped != present or new_oneoffs != oneoffs:
            entry["present"] = deduped
            entry["oneoffs"] = new_oneoffs
            changed += 1
    return changed


def example_images_for_token(data: "DatasetCharacters", image_names, token, limit: int = 4):
    """Up to `limit` image names whose assignment lists `token` as present, in given order."""
    token = (token or "").strip()
    out = []
    for name in image_names:
        entry = data.assignments.get(name)
        if entry and token in entry.get("present", []):
            out.append(name)
            if len(out) >= limit:
                break
    return out


def together_combinations(data: "DatasetCharacters", image_names) -> list:
    """Multi-character combinations that occur, for the 'Together' bucket.

    One dict per distinct set of 2+ explicitly-present people:
    {"tokens": [tok,...] (sorted), "count": n, "examples": [name,...] (<=3)}.
    """
    combos = {}
    order = []
    for name in image_names:
        toks = explicit_tokens_for_image(data, name)
        if len(toks) >= 2:
            key = tuple(sorted(toks))
            if key not in combos:
                combos[key] = []
                order.append(key)
            combos[key].append(name)
    return [{"tokens": list(k), "count": len(combos[k]), "examples": combos[k][:3]}
            for k in order]


def unrecognized_images(data: "DatasetCharacters", image_names) -> list:
    """Images with a person present but unnamed: present == [] AND flagged unidentified.
    (Phase-2 detector sentinel: distinguishes 'a person we couldn't name' from scenery.)"""
    out = []
    for name in image_names:
        entry = data.assignments.get(name)
        if entry is not None and not entry.get("present") and entry.get("unidentified"):
            out.append(name)
    return out


def set_role(data: "DatasetCharacters", token: str, role: str) -> bool:
    """Set a roster character's role ('subject' | 'label'). Returns True if found."""
    role = role if role in ("subject", "label") else "subject"
    for c in data.roster:
        if c.token == token:
            c.role = role
            return True
    return False


def scenery_images(data: "DatasetCharacters", image_names) -> list:
    """Images explicitly marked as having no person (present == [], no one-offs, not
    flagged unidentified). An unassigned image (never reviewed) is NOT scenery."""
    out = []
    for name in image_names:
        entry = data.assignments.get(name)
        if (entry is not None and not entry.get("present")
                and not entry.get("oneoffs") and not entry.get("unidentified")):
            out.append(name)
    return out


def stable_color_for(token: str) -> str:
    """Deterministic mid-bright hex colour for a token (used as a group colour dot)."""
    import hashlib
    h = int(hashlib.md5((token or "").encode("utf-8")).hexdigest()[:6], 16)
    r = 80 + (h & 0x7F)
    g = 80 + ((h >> 7) & 0x7F)
    b = 80 + ((h >> 14) & 0x7F)
    return f"#{r:02x}{g:02x}{b:02x}"


def _ensure_roster(data: "DatasetCharacters", token: str) -> None:
    if token and token not in {c.token for c in data.roster}:
        data.roster.append(Character(token))


def add_token_to_images(data: "DatasetCharacters", image_names, token: str) -> int:
    """Add `token` to each image's present list (auto-adding it to the roster). Returns #changed."""
    token = (token or "").strip()
    if not token:
        return 0
    _ensure_roster(data, token)
    changed = 0
    for name in image_names:
        entry = data.assignments.setdefault(name, {"present": [], "oneoffs": []})
        present = entry.get("present", [])
        if token not in present:
            entry["present"] = present + [token]
            entry.setdefault("oneoffs", [])
            entry["unidentified"] = False  # a named person clears the unrecognized sentinel
            changed += 1
    return changed


def remove_token_from_images(data: "DatasetCharacters", image_names, token: str) -> int:
    """Remove `token` from each named image's present list. Returns #changed."""
    token = (token or "").strip()
    changed = 0
    for name in image_names:
        entry = data.assignments.get(name)
        if not entry:
            continue
        present = entry.get("present", [])
        if token in present:
            entry["present"] = [t for t in present if t != token]
            changed += 1
    return changed


def split_off_new_character(data: "DatasetCharacters", image_names, new_token: str,
                            old_token: str = "") -> int:
    """Give the named images a `new_token` (auto-rostered) and drop `old_token` from them.
    For fixing an AI mis-merge. Returns #images changed."""
    new_token = (new_token or "").strip()
    if not new_token:
        return 0
    _ensure_roster(data, new_token)
    old_token = (old_token or "").strip()
    changed = 0
    for name in image_names:
        entry = data.assignments.setdefault(name, {"present": [], "oneoffs": []})
        present = [t for t in entry.get("present", []) if t != old_token]
        if new_token not in present:
            present.append(new_token)
        if present != entry.get("present", []):
            entry["present"] = present
            entry.setdefault("oneoffs", [])
            changed += 1
    return changed


def apply_ops(data: "DatasetCharacters", ops) -> "DatasetCharacters":
    """Apply a list of correction ops (e.g. produced by the chat assistant).

    Each op is a dict with an 'op' key. Supported:
      {"op":"rename","from":tok,"to":tok}
      {"op":"set_description","token":tok,"description":str}
      {"op":"add","token":tok,"description":str}
      {"op":"remove","token":tok}
      {"op":"set_present","image":name,"present":[tok,...]}
    Unknown / malformed ops are ignored. Pure; mutates and returns `data`.
    """
    for op in ops or []:
        if not isinstance(op, dict):
            continue
        kind = (op.get("op") or op.get("type") or "").strip().lower()

        if kind == "rename":
            src = (op.get("from") or "").strip()
            dst = (op.get("to") or "").strip()
            if not src or not dst:
                continue
            for c in data.roster:
                if c.token == src:
                    c.token = dst
            for entry in data.assignments.values():
                entry["present"] = [dst if t == src else t for t in entry.get("present", [])]
                for o in entry.get("oneoffs", []):
                    if o.get("token") == src:
                        o["token"] = dst

        elif kind in ("set_description", "describe"):
            tok = (op.get("token") or "").strip()
            desc = (op.get("description") or "").strip()
            for c in data.roster:
                if c.token == tok:
                    c.description = desc

        elif kind == "add":
            tok = (op.get("token") or "").strip()
            if tok and tok not in {c.token for c in data.roster}:
                data.roster.append(Character(tok, (op.get("description") or "").strip()))

        elif kind == "remove":
            tok = (op.get("token") or "").strip()
            if not tok:
                continue
            data.roster = [c for c in data.roster if c.token != tok]
            for entry in data.assignments.values():
                entry["present"] = [t for t in entry.get("present", []) if t != tok]

        elif kind == "set_present":
            image = (op.get("image") or "").strip()
            if not image:
                continue
            present = [str(t).strip() for t in (op.get("present") or []) if str(t).strip()]
            e = data.assignments.get(image, {"present": [], "oneoffs": []})
            e["present"] = present
            e.setdefault("oneoffs", [])
            data.assignments[image] = e

    return data
