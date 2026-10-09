"""Model presets: which base model a LoRA is for, and how that model is trained.

Each preset names the engine that trains it and the caption style it starts with.
Adding a model later means adding an entry here plus one proven real run — no new
code paths.
"""
from dataclasses import dataclass

KOHYA = "kohya"            # bundled sd-scripts (core/trainer.py)
ONETRAINER = "onetrainer"  # external OneTrainer install (core/engines/onetrainer)

CAPTION_STYLES = ("prose", "tags", "prose+tags")


@dataclass(frozen=True)
class ModelPreset:
    id: str
    name: str
    engine: str
    caption_style: str
    # What a fresh run's "target steps" starts at. 0 = Auto (engine-computed).
    default_target_steps: int


MODELS = {
    m.id: m for m in (
        ModelPreset("anima", "Anima", KOHYA, "prose+tags", 500),
        ModelPreset("krea2", "Krea 2", ONETRAINER, "prose+tags", 0),
    )
}

DEFAULT_MODEL = "anima"


def get(model_id: str) -> ModelPreset:
    try:
        return MODELS[model_id]
    except KeyError:
        raise KeyError(f"Unknown model preset: {model_id!r}") from None
