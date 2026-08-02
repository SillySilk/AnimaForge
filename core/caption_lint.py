"""Deterministic caption lint: flag mutually exclusive attribute phrases.

The customer case: "brown hair" and "black hair" living in the same .txt after a
hand-edit. No model, no image check — a caption can only be linted against itself,
so this stays a soft warning: with several characters in frame two hair colors are
perfectly legitimate (the multi_person flag softens the message, never hides it).

Runs on the combined caption (prose + tag tail), so a conflict between the NL
sentence and a booru tag is caught too. Word-boundary regex, not comma-splitting —
"Her black hair shines" must match just like the "black hair" tag.
"""
import re
from dataclasses import dataclass, field

_COLORS = (
    "black", "brown", "blonde", "blond", "red", "orange", "pink", "purple",
    "blue", "green", "grey", "gray", "silver", "white", "aqua", "auburn",
)

# family -> (value-phrase regex, canonical value, suppressor phrases)
# A suppressor declares the mix on purpose (multicolored hair, heterochromia),
# which makes two values in one family expected rather than contradictory.
_FAMILIES = {
    "hair color": {
        "pattern": re.compile(r"\b(" + "|".join(_COLORS) + r")\s+hair\b", re.I),
        "canon": {"blond": "blonde", "gray": "grey"},
        "suppressors": re.compile(
            r"\b(multicolored hair|two-tone hair|gradient hair|streaked hair|"
            r"rainbow hair|split-color hair|colored inner hair)\b", re.I),
    },
    "eye color": {
        "pattern": re.compile(r"\b(" + "|".join(_COLORS) + r")\s+eyes\b", re.I),
        "canon": {"gray": "grey"},
        "suppressors": re.compile(r"\bheterochromia\b", re.I),
    },
    "hair length": {
        "pattern": re.compile(r"\b(very long|long|medium|short)\s+hair\b", re.I),
        "canon": {"very long": "long"},
        "suppressors": None,
    },
}

_MULTI_PERSON = re.compile(
    r"\b(\d+\s*(girls|boys|others)|2girls|3girls|4girls|5girls|6\+girls|"
    r"2boys|3boys|4boys|5boys|6\+boys|multiple girls|multiple boys|"
    r"multiple views|group|crowd|duo|trio)\b", re.I)


@dataclass
class CaptionLint:
    conflicts: list = field(default_factory=list)  # [(family, sorted values)]
    multi_person: bool = False


def lint_caption(text: str) -> CaptionLint:
    """Scan one caption for same-family attribute conflicts. Pure, order-stable."""
    text = text or ""
    lint = CaptionLint(multi_person=bool(_MULTI_PERSON.search(text)))
    for family, spec in _FAMILIES.items():
        if spec["suppressors"] is not None and spec["suppressors"].search(text):
            continue
        values = {spec["canon"].get(m.group(1).lower(), m.group(1).lower())
                  for m in spec["pattern"].finditer(text)}
        if len(values) > 1:
            lint.conflicts.append((family, sorted(values)))
    return lint


def format_warning(lint: CaptionLint) -> str:
    """One warning line for the editor; empty string when the caption is clean."""
    if not lint.conflicts:
        return ""
    parts = [f"{family}: {' / '.join(values)}" for family, values in lint.conflicts]
    line = "⚠ Possible conflict — " + ";  ".join(parts)
    if lint.multi_person:
        line += "   (multiple characters in frame — may be intentional)"
    return line
