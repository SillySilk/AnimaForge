"""Step/epoch recommendation for a Krea 2 LoRA run.

The proven reference recipe is 1000 steps for a 25-image character set, so that is the
anchor and every other size scales off it — sublinearly, as the square root of the set
size. Steps grow with the set but steps-PER-IMAGE shrink, because each epoch of a big
set already shows the model more data; a flat per-image rate calibrated at 25 images
asks for 4000 steps at 100 and badly overtrains. A style rides a multiplier on top
(broader signal, slower to converge). The band is deliberately narrow — outside
500..3000 you are either wasting hours of GPU time or overtraining. Setup's "Target
steps" overrides the curve when set; left on Auto, the curve decides.

OneTrainer's config counts epochs, not steps, so `recommend_epochs` converts through
the locked preset's batch size. An epoch is therefore SMALL — a 47-image set at batch
2 is 24 steps — which is why the epoch count lands near 80 whatever the set size
(40·n steps over n/2 steps-per-epoch cancels the image count out). Total steps is the
number that governs training, so `plan` reports it and the UI shows it.
"""
import math

MIN_STEPS = 500
MAX_STEPS = 3000

# Matches the locked preset (assets/krea2_preset.json "batch_size").
BATCH_SIZE = 2

# The locked preset warms the learning rate up over 200 steps, ~10% of the ~1880-step
# reference run. Warmup must scale with the run or a short one spends most of itself
# ramping the LR and barely trains; it never exceeds the preset's proven value.
WARMUP_FRACTION = 0.1
MIN_WARMUP = 10
MAX_WARMUP = 200

# The proven point on the curve: a 25-image character set trains in 1000 steps.
ANCHOR_STEPS = 1000
ANCHOR_IMAGES = 25

# Prodigy re-estimates its step size until told to stop (`prodigy_steps`). Frozen
# at three quarters of the run so the closing stretch trains at a settled rate.
# Sized per run like the warmup; see preset.py for what is and is not proven.
PRODIGY_FREEZE_FRACTION = 0.75

# How much more work a concept needs than a character, at the same set size.
_MULTIPLIERS = {
    "character": 1.0,
    "face": 1.0,
    "object": 1.125,
    "concept": 1.125,   # Home's subject-type key for object/concept sets
    "style": 1.5,
}
_DEFAULT_MULTIPLIER = _MULTIPLIERS["character"]


def _target(target_steps) -> int:
    """A usable explicit target, or 0 for Auto. Junk degrades to Auto, never crashes."""
    try:
        return max(0, int(target_steps or 0))
    except (TypeError, ValueError):
        return 0


def recommend_steps(image_count: int, concept: str, target_steps=0) -> int:
    """A sane target step count for this set, clamped to MIN_STEPS..MAX_STEPS.

    An explicit `target_steps` (Home's "Training steps") wins outright; 0 means Auto,
    which scales the proven anchor by the square root of the set size.
    """
    explicit = _target(target_steps)
    if explicit:
        raw = explicit
    else:
        mult = _MULTIPLIERS.get((concept or "").strip().lower(), _DEFAULT_MULTIPLIER)
        images = max(0, int(image_count or 0))
        raw = round(ANCHOR_STEPS * mult * math.sqrt(images / ANCHOR_IMAGES))
    return max(MIN_STEPS, min(MAX_STEPS, raw))


def steps_per_epoch(image_count: int, batch_size: int = BATCH_SIZE) -> int:
    """How many optimizer steps one pass over the set takes. Always >= 1."""
    return max(1, math.ceil(max(1, int(image_count or 0)) / max(1, batch_size)))


def recommend_epochs(image_count: int, concept: str, batch_size: int = BATCH_SIZE,
                     target_steps=0) -> int:
    """The `epochs` value to patch into the run config: recommended steps converted
    through the dataset's steps-per-epoch at the preset's batch size. Always >= 1."""
    steps = recommend_steps(image_count, concept, target_steps)
    return max(1, round(steps / steps_per_epoch(image_count, batch_size)))


def warmup_steps(total_steps: int) -> int:
    """LR warmup scaled to the run — WARMUP_FRACTION of it, capped at the preset's
    value so a shortened run is not almost entirely warmup."""
    scaled = round(max(0, int(total_steps or 0)) * WARMUP_FRACTION)
    return max(MIN_WARMUP, min(MAX_WARMUP, scaled))


def prodigy_steps(total_steps: int) -> int:
    """The step after which Prodigy stops re-estimating its step size: three
    quarters of the run, never below 1 so a tiny run still freezes at the end."""
    return max(1, round(max(0, int(total_steps or 0)) * PRODIGY_FREEZE_FRACTION))


def plan(image_count: int, concept: str, target_steps=0,
         batch_size: int = BATCH_SIZE) -> dict:
    """The whole run plan in one place, so the config and the progress bar agree.

    `total_steps` is what the trainer will ACTUALLY run (epochs × steps_per_epoch),
    not the pre-rounding target — the bar is sized from it, so it has to be the
    real number.
    """
    per_epoch = steps_per_epoch(image_count, batch_size)
    epochs = recommend_epochs(image_count, concept, batch_size, target_steps)
    total = epochs * per_epoch
    return {
        "epochs": epochs,
        "steps_per_epoch": per_epoch,
        "total_steps": total,
        "warmup_steps": warmup_steps(total),
        "prodigy_steps": prodigy_steps(total),
    }
