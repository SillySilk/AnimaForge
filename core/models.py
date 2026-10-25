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
class StepProfile:
    """How a model's auto step target is derived (see core/step_calculator.py)."""
    batch_size: int
    exposures: dict        # {"character": int, "concept": int, "style": int}
    floor_steps: dict      # same keys
    soft_cap_steps: int


ANIMA_STEPS = StepProfile(4, {"character": 56, "concept": 34, "style": 26},
                          {"character": 800, "concept": 800, "style": 800}, 3000)
MOSS_STEPS = StepProfile(1, {"character": 40, "concept": 25, "style": 12},
                         {"character": 2000, "concept": 1000, "style": 600}, 5000)


@dataclass(frozen=True)
class ModelPreset:
    id: str
    name: str
    engine: str
    caption_style: str
    # What a fresh run's "target steps" starts at. 0 = Auto (engine-computed).
    default_target_steps: int
    steps: "StepProfile | None" = None   # None = the engine computes steps (OneTrainer)
    min_vram_gb: int = 0
    forge_test: bool = True              # can the Forge test-render client use this model
    cast_lora: bool = True               # can it train Cast (multi-character) LoRAs


MODELS = {
    m.id: m for m in (
        ModelPreset("anima", "Anima", KOHYA, "prose+tags", 500, steps=ANIMA_STEPS),
        ModelPreset("krea2", "Krea 2", ONETRAINER, "prose+tags", 0),
        ModelPreset("moss", "MOSS-Image", KOHYA, "prose+tags", 2500,
                    steps=MOSS_STEPS, min_vram_gb=12, forge_test=False, cast_lora=False),
    )
}

DEFAULT_MODEL = "anima"


def get(model_id: str) -> ModelPreset:
    try:
        return MODELS[model_id]
    except KeyError:
        raise KeyError(f"Unknown model preset: {model_id!r}") from None
