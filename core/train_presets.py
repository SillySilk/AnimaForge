"""Named training presets — "intent profiles" for the front-page Preset picker.

A preset bundles the per-run training intent (subject type, optimizer, network
size, step target) under one name, so the front page carries a single labeled
control instead of a knob per setting (the anti-Kohya rule: only vital controls
up front). Three built-ins cover the classic intents; users add their own in
Settings — Anima is diverse enough that bigger configurations (larger dim/alpha,
longer runs) earn a saved name.

Storage is a JSON array under one settings key (machine-global, so custom
presets survive reinstalls and are shared across side-by-side installs). All
functions here are pure JSON-string-in/JSON-string-out; the UI owns the
QSettings round-trip.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from core import models

SETTINGS_KEY = "train_presets_json"

SUBJECT_TYPES = ("character", "concept", "style")


@dataclass
class TrainPreset:
    name: str
    subject_type: str = "character"   # character | concept | style
    optimizer: str = "prodigy"        # prodigy | adamw8bit
    learning_rate: float = 0.0001     # honored only by adamw8bit (prodigy is LR-free)
    network_dim: int = 16
    network_alpha: int = 8
    target_steps: int = 0             # 0 = auto-suggest from the dataset
    uncap_steps: bool = False
    # The model this preset was saved for ("" = any model: the built-ins and presets
    # saved before v3) and its advanced-setting overrides, which only mean something
    # to that model's engine.
    model: str = ""
    overrides: dict = field(default_factory=dict)
    builtin: bool = False


# The three classic intents. "Person" is the default selection.
BUILTINS = (
    TrainPreset("Person", subject_type="character", builtin=True),
    TrainPreset("Object / Concept", subject_type="concept", builtin=True),
    TrainPreset("Style", subject_type="style", builtin=True),
)
DEFAULT_NAME = "Person"


def builtin_for_subject(subject_type: str) -> TrainPreset:
    """The built-in preset matching a subject-type key (default: Person)."""
    key = (subject_type or "").strip().lower()
    for p in BUILTINS:
        if p.subject_type == key:
            return p
    return BUILTINS[0]


def parse_customs(store_json: str) -> list[TrainPreset]:
    """Custom presets from the stored JSON. Malformed input returns [] (never raises)."""
    try:
        raw = json.loads(store_json or "[]")
    except (TypeError, ValueError):
        return []
    out = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict) or not str(item.get("name", "")).strip():
            continue
        base = TrainPreset(name=str(item["name"]).strip())
        for f in ("subject_type", "optimizer"):
            if isinstance(item.get(f), str):
                setattr(base, f, item[f])
        if base.subject_type not in SUBJECT_TYPES:
            base.subject_type = "character"
        try:
            base.learning_rate = float(item.get("learning_rate", base.learning_rate))
            base.network_dim = int(item.get("network_dim", base.network_dim))
            base.network_alpha = int(item.get("network_alpha", base.network_alpha))
            base.target_steps = int(item.get("target_steps", base.target_steps))
        except (TypeError, ValueError):
            pass
        base.uncap_steps = bool(item.get("uncap_steps", False))
        if item.get("model") in models.MODELS:
            base.model = item["model"]
        if isinstance(item.get("overrides"), dict) and base.model:
            base.overrides = dict(item["overrides"])
        base.builtin = False  # stored presets are custom by definition
        out.append(base)
    return out


def serialize_customs(customs: list[TrainPreset]) -> str:
    return json.dumps([{k: v for k, v in asdict(p).items() if k != "builtin"}
                       for p in customs])


def all_presets(store_json: str) -> list[TrainPreset]:
    """Built-ins first, then customs (sorted by name)."""
    return list(BUILTINS) + sorted(parse_customs(store_json), key=lambda p: p.name.lower())


def presets_for(store_json: str, model_id: str) -> list[TrainPreset]:
    """The presets offered while `model_id` is selected: built-ins, model-less
    customs, and customs saved for that model."""
    return [p for p in all_presets(store_json) if p.model in ("", model_id)]


def find(store_json: str, name: str) -> TrainPreset | None:
    want = (name or "").strip().lower()
    for p in all_presets(store_json):
        if p.name.lower() == want:
            return p
    return None


def add_custom(store_json: str, preset: TrainPreset) -> str:
    """Add (or replace, by case-insensitive name) a custom preset. Built-in names
    are reserved — raises ValueError so the UI can tell the user."""
    name = (preset.name or "").strip()
    if not name:
        raise ValueError("Preset needs a name.")
    if any(b.name.lower() == name.lower() for b in BUILTINS):
        raise ValueError(f"“{name}” is a built-in preset — pick another name.")
    preset.name = name
    customs = [p for p in parse_customs(store_json) if p.name.lower() != name.lower()]
    customs.append(preset)
    return serialize_customs(customs)


def remove_custom(store_json: str, name: str) -> str:
    want = (name or "").strip().lower()
    return serialize_customs(
        [p for p in parse_customs(store_json) if p.name.lower() != want])


# Why the built-ins differ even though optimizer/network match: the step budget.
_FORMULA_NOTES = {
    "character": "identities need the most exposure",
    "concept": "objects converge faster",
    "style": "style takes fastest",
}


def formula_line(p: TrainPreset, model: str = "anima") -> str:
    """The step math behind a preset, for the picker's small print.

    The built-ins look identical at a glance (same optimizer/network) — the real
    difference is the exposures-per-image target driving the auto step count, so
    show that formula. Fixed-step presets state the fixed number instead.
    """
    if p.target_steps:
        return f"steps fixed at {p.target_steps:,}" + (" · uncapped" if p.uncap_steps else "")
    from core.step_calculator import target_exposures
    profile = models.get(model).steps
    if profile is None:
        return "auto steps sized by the training engine"
    exp = target_exposures(p.subject_type, model)
    note = _FORMULA_NOTES.get(p.subject_type, "")
    line = f"auto steps = {exp} × images ÷ {profile.batch_size}"
    return f"{line}  ({note})" if note else line


def summary_line(p: TrainPreset) -> str:
    """One-line description for pickers/lists."""
    opt = "Prodigy (auto LR)" if p.optimizer == "prodigy" else f"AdamW8bit @ {p.learning_rate:g}"
    steps = "auto steps" if not p.target_steps else f"{p.target_steps} steps"
    if p.uncap_steps:
        steps += " (uncapped)"
    subject = {"character": "Character", "concept": "Object / Concept",
               "style": "Style"}.get(p.subject_type, p.subject_type)
    return f"{subject} · {opt} · dim {p.network_dim}/α{p.network_alpha} · {steps}"
