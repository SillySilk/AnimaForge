import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.lowvram import recipe_for, set_current, get_current, clear, MAX_BLOCKS_TO_SWAP


def test_recipes_keep_effective_batch_4():
    for gb in (16, 12, 10, 8):
        r = recipe_for(gb)
        assert r["micro_batch"] * r["grad_accum"] == 4
        assert r["fp8_base"] is False
        assert 0 <= r["blocks_to_swap"] <= MAX_BLOCKS_TO_SWAP


def test_recipe_values():
    assert recipe_for(12) == {"micro_batch": 1, "grad_accum": 4, "blocks_to_swap": 8, "fp8_base": False}
    assert recipe_for(10)["blocks_to_swap"] == 16
    assert recipe_for(8)["blocks_to_swap"] == 24
    assert recipe_for(16) == {"micro_batch": 4, "grad_accum": 1, "blocks_to_swap": 0, "fp8_base": False}


def test_unknown_target_falls_back_to_16():
    assert recipe_for(9999)["micro_batch"] == 4


def test_holder_default_none_and_roundtrip():
    clear()
    assert get_current() is None
    set_current({"micro_batch": 1, "grad_accum": 4})
    assert get_current() == {"micro_batch": 1, "grad_accum": 4}
    set_current(None)
    assert get_current() is None
    set_current({"x": 1})
    clear()
    assert get_current() is None
