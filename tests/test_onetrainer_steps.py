import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.engines.onetrainer.steps import plan, recommend_steps, recommend_epochs, warmup_steps


def test_recommended_steps_scale_with_image_count():
    assert recommend_steps(10, "character") < recommend_steps(40, "character")


def test_a_style_needs_more_steps_than_a_character_for_the_same_set():
    assert recommend_steps(20, "style") > recommend_steps(20, "character")


def test_steps_are_clamped_to_the_supported_band():
    assert recommend_steps(1, "character") == 500
    assert recommend_steps(10000, "style") == 3000


def test_an_unknown_concept_falls_back_to_the_character_rate():
    assert recommend_steps(20, "banana") == recommend_steps(20, "character")


def test_a_zero_image_set_still_returns_the_floor():
    assert recommend_steps(0, "character") == 500


def test_recommend_epochs_typical_character_set():
    # 25 images @40 = 1000 steps; 13 steps/epoch at batch 2 -> 77 epochs
    assert recommend_epochs(25, "character") == 77


def test_recommend_epochs_never_below_one():
    assert recommend_epochs(0, "character") >= 1
    assert recommend_epochs(1, "style") >= 1


def test_recommend_epochs_concept_alias_matches_object():
    assert recommend_epochs(20, "concept") == recommend_epochs(20, "object")


# ---------------------------------------------------------------------------
# An explicit target overrides the per-image rate. Without one the auto
# recommendation stands, so an untouched Setup keeps today's behaviour.
# ---------------------------------------------------------------------------

def test_an_explicit_target_overrides_the_per_image_rate():
    assert recommend_steps(47, "character") == 1371       # auto, sublinear
    assert recommend_steps(47, "character", target_steps=800) == 800
    assert recommend_steps(47, "style", target_steps=800) == 800


def test_a_zero_target_means_auto():
    for target in (0, None, ""):
        assert recommend_steps(47, "character", target_steps=target) == 1371


def test_an_explicit_target_is_still_clamped_to_the_band():
    assert recommend_steps(47, "character", target_steps=50) == 500
    assert recommend_steps(47, "character", target_steps=99999) == 3000


def test_target_steps_reaches_the_epoch_count():
    # 47 images, batch 2 -> 24 steps/epoch. 800 steps is ~33 epochs.
    assert recommend_epochs(47, "character", target_steps=800) == 33
    assert recommend_epochs(47, "character") == 57          # auto: 1371 / 24


# ---------------------------------------------------------------------------
# Auto is sublinear: steps grow with the set, steps-PER-IMAGE shrink. A linear
# rate calibrated on a 25-image set overshoots badly by 100 images, because each
# epoch of a big set already shows the model more data.
# ---------------------------------------------------------------------------

def test_the_proven_anchor_is_preserved_exactly():
    # 1000 steps for a 25-image character set is the recipe that worked. Every
    # other size is scaled off this point, so it must land on it dead on.
    assert recommend_steps(25, "character") == 1000


def test_steps_grow_but_steps_per_image_shrink_with_set_size():
    sizes = [25, 47, 100, 200]
    totals = [recommend_steps(n, "character") for n in sizes]
    per_image = [t / n for t, n in zip(totals, sizes)]
    assert totals == sorted(totals), totals
    assert per_image == sorted(per_image, reverse=True), per_image


def test_auto_curve_hits_the_agreed_numbers():
    assert recommend_steps(15, "character") == 775
    assert recommend_steps(25, "character") == 1000
    assert recommend_steps(47, "character") == 1371
    assert recommend_steps(100, "character") == 2000


def test_concept_multipliers_ride_on_top_of_the_curve():
    # The multiplier scales the curve and the result is rounded ONCE, so allow the
    # single step of slack that separates that from rounding a rounded number.
    base = recommend_steps(47, "character")
    assert abs(recommend_steps(47, "object") - base * 1.125) <= 1
    assert abs(recommend_steps(47, "style") - base * 1.5) <= 1
    assert recommend_steps(47, "face") == base


# ---------------------------------------------------------------------------
# Warmup has to scale with the run. The locked preset warms up over 200 steps,
# which is ~10% of the ~1880-step reference run; leaving it fixed would spend a
# 240-step run almost entirely ramping the learning rate.
# ---------------------------------------------------------------------------

def test_warmup_scales_with_the_run_length():
    assert warmup_steps(1880) == 188
    assert warmup_steps(800) == 80
    assert warmup_steps(500) == 50


def test_warmup_never_swallows_a_short_run():
    # Whatever the run length, warmup stays a small slice of it.
    for total in (240, 500, 800, 1880, 3000):
        assert warmup_steps(total) <= total // 5


def test_warmup_is_capped_at_the_presets_value():
    assert warmup_steps(100000) == 200


def test_plan_is_internally_consistent():
    p = plan(47, "character")
    assert p["steps_per_epoch"] == 24
    assert p["epochs"] == 57
    # total_steps is what the trainer will ACTUALLY run: epochs * steps_per_epoch,
    # not the pre-rounding target. The progress bar is sized from this.
    assert p["total_steps"] == 57 * 24
    assert p["warmup_steps"] == warmup_steps(p["total_steps"])


def test_plan_honours_an_explicit_target():
    p = plan(47, "character", target_steps=800)
    assert p["epochs"] == 33
    assert p["total_steps"] == 33 * 24
    assert p["warmup_steps"] == 79


def test_the_plan_freezes_prodigy_for_the_last_quarter_of_the_run():
    """Prodigy keeps re-estimating its step size for the whole run unless told to
    stop; the plan stops it at three quarters, so the closing stretch trains at
    a settled rate. Sized from the real step count, like the warmup."""
    p = plan(36, "character")
    assert p["prodigy_steps"] == round(p["total_steps"] * 0.75)
    assert 0 < p["prodigy_steps"] < p["total_steps"]
