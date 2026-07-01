import math

# ── Exposure-based step targeting ────────────────────────────────────────────
# The quality knob is EXPOSURES PER IMAGE — how many times each image is shown to
# the optimizer over the whole run — NOT the raw total step count:
#
#   exposures_per_image = repeats × epochs              (batch-independent)
#   total_steps         = image_count × exposures_per_image / effective_batch
#
# The empirical sweet spot is ~30–66 exposures/image (character needs the most, style
# the least). This matches the original "a style LoRA looked good by ~500 steps" anchor
# — for a ~30-image set that is 500*4/30 ≈ 66 exposures — and a 152-image character set
# lands at ~2,500 steps, not ~25,000.
#
# Dataset size enters ONLY through image_count: a bigger cast means more images and
# therefore more total steps at the SAME per-image exposure. So there is deliberately
# no roster-size step bump — a thin cast is fixed by adding images (see
# images_per_character_warning), never by over-exposing the few images you have.
EXPOSURE_TARGETS = {
    "character": 66,   # identities need the most exposure to lock in
    "concept": 40,     # objects / concepts converge faster on Anima
    "style": 30,       # style is the most forgiving / fastest to take
}
DEFAULT_EXPOSURES = 66

# Anima's effective training batch (train_batch_size; gradient accumulation in
# low-VRAM mode keeps this effective value constant, so 4 is always correct here).
BATCH_SIZE = 4

# Bounds on the AUTO-SUGGESTION (not on manual overrides). The floor keeps a tiny set
# from being starved; the soft cap keeps a huge cast from running overnight by default.
# A power-user "uncap" toggle drops the cap (the floor still applies).
FLOOR_STEPS = 800       # ~45 min on a 4060 Ti — a real run even for a small set
SOFT_CAP_STEPS = 3500   # ~3.5 hr — default ceiling; removable via the uncap toggle

# Back-compatible default total-step target, used only before a dataset is loaded.
TARGET_STEPS = 500

_TYPE_ALIASES = {
    "character": "character", "person": "character", "face": "character",
    "concept": "concept", "object": "concept",
    "style": "style",
}


def canonical_subject_type(subject_type: str) -> str:
    """Map any subject-type label to one of: character / concept / style.

    Unknown / empty types fall back to 'style' — the most forgiving band.
    """
    return _TYPE_ALIASES.get((subject_type or "").strip().lower(), "style")


def target_exposures(subject_type: str) -> int:
    """The fine-tuned exposures-per-image target for a subject type."""
    return EXPOSURE_TARGETS.get(canonical_subject_type(subject_type), DEFAULT_EXPOSURES)


def exposures_for_steps(image_count: int, total_steps: int,
                        batch_size: int = BATCH_SIZE) -> float:
    """Invert the identity: how many exposures/image a given total-step count yields."""
    if image_count <= 0:
        return 0.0
    return total_steps * batch_size / image_count


def calculate_training_params(image_count: int, target_steps: int = TARGET_STEPS,
                              batch_size: int = BATCH_SIZE) -> dict:
    """
    Calculate epochs and repeats to hit ~target_steps total steps.
    Total steps = (image_count * repeats * epochs) / batch_size

    Strategy:
    - Keep epochs between 10-30
    - Adjust repeats to hit target
    - Round up to nearest whole number

    The returned dict also reports `exposures_per_image` (= repeats * epochs), the
    quantity that actually governs LoRA quality.
    """
    if image_count <= 0:
        return {
            "epochs": 0,
            "repeats": 0,
            "total_steps": 0,
            "steps_per_epoch": 0,
            "image_count": image_count,
            "exposures_per_image": 0,
            "batch_size": batch_size,
        }

    best_result = None
    best_distance = float("inf")

    for epochs in range(10, 31):
        # total_steps = (image_count * repeats * epochs) / batch_size
        # repeats = (target_steps * batch_size) / (image_count * epochs)
        repeats_float = (target_steps * batch_size) / (image_count * epochs)
        repeats = max(1, math.ceil(repeats_float))

        total_steps = (image_count * repeats * epochs) // batch_size
        distance = abs(total_steps - target_steps)

        if best_result is None or distance < best_distance:
            best_distance = distance
            steps_per_epoch = (image_count * repeats) // batch_size
            best_result = {
                "epochs": epochs,
                "repeats": repeats,
                "total_steps": total_steps,
                "steps_per_epoch": steps_per_epoch,
                "image_count": image_count,
                "exposures_per_image": repeats * epochs,
                "batch_size": batch_size,
            }

    return best_result


def _raw_target_steps(subject_type: str, image_count: int,
                      batch_size: int = BATCH_SIZE) -> int:
    """Unbounded exposures-per-image formula: exposures * image_count / batch."""
    count = max(int(image_count or 0), 1)
    return round(target_exposures(subject_type) * count / batch_size)


def suggest_target_steps(subject_type: str, image_count: int, n_characters: int = 1,
                         batch_size: int = BATCH_SIZE, uncapped: bool = False) -> int:
    """Recommend a total-step target that lands the dataset on its per-type
    exposures-per-image goal, bounded by the floor and (unless `uncapped`) the soft cap.

    total_steps = clamp(target_exposures(subject_type) * image_count / batch_size,
                        FLOOR_STEPS, SOFT_CAP_STEPS)

    `n_characters` is accepted for signature/back-compat but no longer changes the
    suggestion: exposures/image are held constant and total steps scale with
    `image_count` (more cast → more images → more steps). A thin cast is surfaced by
    `images_per_character_warning`, i.e. fixed with more images, not more exposure.

    `uncapped=True` removes the soft cap (the floor still applies) — the power-user
    escape hatch behind the Train-tab toggle.
    """
    raw = _raw_target_steps(subject_type, image_count, batch_size)
    if not uncapped:
        raw = min(raw, SOFT_CAP_STEPS)
    return max(FLOOR_STEPS, raw)


def is_capped(subject_type: str, image_count: int,
              batch_size: int = BATCH_SIZE) -> bool:
    """True when the raw formula exceeds the soft cap — i.e. the default suggestion is
    being clamped down and the uncap toggle would let it train longer. Drives the UI hint.
    """
    return _raw_target_steps(subject_type, image_count, batch_size) > SOFT_CAP_STEPS


def images_per_character_warning(image_count: int, n_characters: int,
                                 floor: int = 15) -> str:
    """Return a warning string when a group set is thin on images-per-character, else ''."""
    n = int(n_characters or 0)
    if n > 1 and image_count > 0:
        per = image_count / n
        if per < floor:
            return (f"~{per:.0f} images/character across {n} characters "
                    f"(below ~{floor}) — consider more images per character.")
    return ""


def format_calculation_string(params: dict) -> str:
    """Returns a human-readable string describing the calculation."""
    if params["image_count"] <= 0:
        return "No images loaded"
    exp = params.get("exposures_per_image", 0)
    batch = params.get("batch_size", BATCH_SIZE)
    return (
        f"{params['image_count']} images × {params['repeats']} repeats × "
        f"{params['epochs']} epochs ÷ {batch} batch = "
        f"{params['total_steps']} steps  (~{exp} exposures/image)"
    )
