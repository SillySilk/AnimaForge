import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.step_calculator import (
    suggest_target_steps,
    is_capped,
    images_per_character_warning,
    target_exposures,
    exposures_for_steps,
    calculate_training_params,
    canonical_subject_type,
    BATCH_SIZE,
    FLOOR_STEPS,
    SOFT_CAP_STEPS,
)


# ── recalibrated per-type exposures ──────────────────────────────────────────
def test_per_type_exposure_values():
    assert target_exposures("character") == 56
    assert target_exposures("concept") == 34
    assert target_exposures("style") == 26


def test_type_ordering_character_highest_style_lowest():
    # For the same mid-size set (all in the [floor, cap] band): char > concept > style.
    c = suggest_target_steps("character", 150)
    o = suggest_target_steps("concept", 150)
    s = suggest_target_steps("style", 150)
    assert c > o > s
    assert FLOOR_STEPS < s and c < SOFT_CAP_STEPS


# ── 152 character images lands inside the band ───────────────────────────────
def test_character_152_lands_inside_band():
    # 56 * 152 / 4 = 2128, comfortably inside [800, 3000].
    assert suggest_target_steps("character", 152) == 2128


# ── floor ────────────────────────────────────────────────────────────────────
def test_small_set_is_floored():
    # 66 * 30 / 4 = 495 -> floored to 800.
    assert suggest_target_steps("character", 30) == FLOOR_STEPS


def test_style_tiny_set_is_floored():
    # 30 * 10 / 4 = 75 -> floored to 800.
    assert suggest_target_steps("style", 10) == FLOOR_STEPS


# ── soft cap ──────────────────────────────────────────────────────────────────
def test_large_set_is_capped():
    # 56 * 500 / 4 = 7000 -> capped to 3000.
    assert suggest_target_steps("character", 500) == SOFT_CAP_STEPS


def test_uncapped_removes_the_cap():
    # With the cap removed the raw formula stands (floor still applies, but n/a here).
    assert suggest_target_steps("character", 500, uncapped=True) == 7000


def test_uncapped_still_respects_floor():
    # Removing the cap does not remove the floor.
    assert suggest_target_steps("character", 30, uncapped=True) == FLOOR_STEPS


# ── is_capped helper drives the UI hint ──────────────────────────────────────
def test_is_capped_true_only_when_formula_exceeds_cap():
    assert is_capped("character", 500) is True      # raw 7000 > 3000
    assert is_capped("character", 152) is False     # raw 2128 < 3000
    assert is_capped("character", 30) is False       # raw 420 (floored, not capped)


# ── linear scaling holds inside the band ─────────────────────────────────────
def test_scales_with_image_count_inside_band():
    # 100 and 200 imgs both land in (floor, cap): 1400 and 2800.
    assert suggest_target_steps("character", 200) == 2 * suggest_target_steps("character", 100)


def test_roster_size_does_not_bump_suggestion():
    assert suggest_target_steps("character", 150, 3) == suggest_target_steps("character", 150, 1)
    assert suggest_target_steps("character", 150, 20) == suggest_target_steps("character", 150, 1)


# ── aliases / unknown types ──────────────────────────────────────────────────
def test_aliases_and_unknown_type():
    assert canonical_subject_type("person") == "character"
    assert canonical_subject_type("object") == "concept"
    assert canonical_subject_type("") == "style"   # unknown -> most forgiving band
    assert suggest_target_steps("person", 150) == suggest_target_steps("character", 150)
    assert suggest_target_steps("object", 150) == suggest_target_steps("concept", 150)
    assert suggest_target_steps("", 150) == suggest_target_steps("style", 150)


# ── calculate_training_params unchanged math ─────────────────────────────────
def test_calculate_params_reports_consistent_exposures():
    target = suggest_target_steps("character", 152)
    p = calculate_training_params(152, target_steps=target)
    assert p["exposures_per_image"] == p["repeats"] * p["epochs"]
    assert p["total_steps"] > 0
    assert isinstance(p["epochs"], int) and isinstance(p["repeats"], int)


def test_exposures_for_steps_inverts_the_identity():
    assert round(exposures_for_steps(152, 2128)) == 56


# ── thin-roster warning (unchanged) ──────────────────────────────────────────
def test_images_per_character_warning():
    assert images_per_character_warning(60, 1) == ""        # single -> no warning
    assert images_per_character_warning(60, 3) == ""        # 20 each -> ok
    assert "below" in images_per_character_warning(20, 3)   # ~6 each -> warn
    assert images_per_character_warning(0, 3) == ""         # no images -> no warning


# ── per-model profiles ───────────────────────────────────────────────────────
# Expected values were computed on the pre-profile code (before this change).
_ANIMA_SWEEP = {
    "character": {1: 800, 10: 800, 20: 800, 30: 800, 60: 840, 100: 1400, 152: 2128,
                  300: 3000, 500: 3000},
    "concept": {1: 800, 10: 800, 20: 800, 30: 800, 60: 800, 100: 850, 152: 1292,
                300: 2550, 500: 3000},
    "style": {1: 800, 10: 800, 20: 800, 30: 800, 60: 800, 100: 800, 152: 988,
              300: 1950, 500: 3000},
}
_ANIMA_UNCAPPED = {("character", 300): 4200, ("character", 500): 7000,
                   ("concept", 500): 4250, ("style", 500): 3250, ("style", 300): 1950}


def test_anima_suggestions_unchanged():
    for t, rows in _ANIMA_SWEEP.items():
        for n, want in rows.items():
            assert suggest_target_steps(t, n) == want
            assert suggest_target_steps(t, n, model="anima") == want
    for (t, n), want in _ANIMA_UNCAPPED.items():
        assert suggest_target_steps(t, n, uncapped=True) == want
    assert is_capped("character", 500) and not is_capped("character", 152)
    assert calculate_training_params(60, 840)["batch_size"] == 4


def test_moss_character_60_images():
    assert suggest_target_steps("character", 60, model="moss") == 2400


def test_moss_character_small_set_floor():
    assert suggest_target_steps("character", 20, model="moss") == 2000


def test_moss_style_floor():
    assert suggest_target_steps("style", 30, model="moss") == 600


def test_moss_cap():
    assert suggest_target_steps("character", 200, model="moss") == 5000
    assert suggest_target_steps("character", 200, model="moss", uncapped=True) == 8000
    assert is_capped("character", 200, model="moss")
    assert not is_capped("character", 60, model="moss")


def test_moss_params_batch_1():
    p = calculate_training_params(60, 2400, model="moss")
    assert p["batch_size"] == 1 and p["total_steps"] == 2400


def test_engine_sized_model_has_no_step_profile():
    import pytest
    with pytest.raises(ValueError):
        suggest_target_steps("character", 60, model="krea2")
