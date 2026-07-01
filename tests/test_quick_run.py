import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.quick_run import plan_phases, DETECT, CAPTION, TRAIN


def test_character_uncaptioned_no_roster_runs_all():
    assert plan_phases("character", has_roster=False, is_captioned=False) == [DETECT, CAPTION, TRAIN]


def test_character_with_roster_skips_detect():
    assert plan_phases("character", has_roster=True, is_captioned=False) == [CAPTION, TRAIN]


def test_captioned_skips_caption():
    assert plan_phases("character", has_roster=True, is_captioned=True) == [TRAIN]


def test_object_never_detects():
    assert plan_phases("concept", has_roster=False, is_captioned=False) == [CAPTION, TRAIN]
    assert plan_phases("object", has_roster=False, is_captioned=True) == [TRAIN]


def test_style_never_detects():
    assert plan_phases("style", has_roster=False, is_captioned=False) == [CAPTION, TRAIN]


def test_train_always_last_and_present():
    for st in ("character", "concept", "style", "", "weird"):
        phases = plan_phases(st, has_roster=False, is_captioned=False)
        assert phases[-1] == TRAIN
        assert phases.count(TRAIN) == 1
