"""User-defined find/replace rules applied when a caption is built.

Banning a tag is find/replace with an empty replacement, so there is one rule type:
    ("1boy", "")        -> drops the tag
    ("man", "woman")    -> fixes a misgendered description

Matching is whole-word and case-insensitive, so `man` never matches inside `woman`,
`human`, `romantic` or `command`. Deleting a term leaves stray separators behind
("1girl, , solo"), so every application ends with a tidy pass.
"""
import json
import re


def parse_rules(raw):
    """[(find, replace)] from the stored JSON. Never raises; bad input yields []."""
    try:
        data = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict):
            find, repl = item.get("find", ""), item.get("replace", "")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            find, repl = item
        else:
            continue
        find = str(find or "").strip()
        if find:
            out.append((find, str(repl or "")))
    return out


def dump_rules(rules):
    """The inverse of parse_rules, for the settings store."""
    return json.dumps([{"find": f, "replace": r} for f, r in rules], indent=2)


def _match_case(matched: str, replacement: str) -> str:
    """Carry the matched term's capitalization onto the replacement."""
    if not replacement or not matched:
        return replacement
    if matched.isupper() and len(matched) > 1:
        return replacement.upper()
    if matched[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def tidy(text: str) -> str:
    """Repair the separators a deletion leaves behind."""
    text = re.sub(r"[ \t]+", " ", text)          # squeeze runs of spaces
    text = re.sub(r"\s*,(?:\s*,)+", ",", text)   # ", , ," -> ","
    text = re.sub(r",(?=\S)", ", ", text)        # ensure one space after a comma
    text = re.sub(r"\s+,", ",", text)            # no space before a comma
    return text.strip().strip(",").strip()


def apply_caption_rules(text: str, rules) -> str:
    """Apply every rule to `text` in order, then tidy. Pure."""
    if not text or not rules:
        return text
    for find, replacement in rules:
        # Lookarounds on \w rather than \b: \b misbehaves when a term starts or ends
        # with a non-word char (e.g. "(medium)"), while \w boundaries hold for
        # "1boy", "score_7" and multi-word phrases alike.
        pattern = r"(?<!\w)" + re.escape(find) + r"(?!\w)"
        text = re.sub(pattern,
                      lambda m, r=replacement: _match_case(m.group(0), r),
                      text, flags=re.IGNORECASE)
    return tidy(text)
