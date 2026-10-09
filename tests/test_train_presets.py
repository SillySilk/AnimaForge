import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import train_presets as tp


def test_builtins_person_default():
    names = [p.name for p in tp.BUILTINS]
    assert names == ["Person", "Object / Concept", "Style"]
    assert tp.DEFAULT_NAME == "Person"
    assert all(p.builtin for p in tp.BUILTINS)
    assert tp.builtin_for_subject("style").name == "Style"
    assert tp.builtin_for_subject("nonsense").name == "Person"


def test_parse_customs_malformed_is_empty():
    assert tp.parse_customs("") == []
    assert tp.parse_customs("not json") == []
    assert tp.parse_customs('{"a": 1}') == []
    assert tp.parse_customs('[{"no_name": true}, 42]') == []


def test_add_find_remove_round_trip():
    p = tp.TrainPreset("Big Style", subject_type="style", optimizer="adamw8bit",
                       learning_rate=2e-4, network_dim=64, network_alpha=32,
                       target_steps=2000, uncap_steps=True)
    store = tp.add_custom("", p)
    got = tp.find(store, "big style")   # case-insensitive
    assert got is not None and got.network_dim == 64 and got.uncap_steps
    assert not got.builtin
    # all_presets: builtins first, custom present
    names = [x.name for x in tp.all_presets(store)]
    assert names[:3] == ["Person", "Object / Concept", "Style"]
    assert "Big Style" in names
    # replace by same name
    p2 = tp.TrainPreset("big style", network_dim=96)
    store = tp.add_custom(store, p2)
    assert tp.find(store, "Big Style").network_dim == 96
    assert len(tp.parse_customs(store)) == 1
    # remove
    store = tp.remove_custom(store, "BIG STYLE")
    assert tp.parse_customs(store) == []


def test_builtin_names_reserved():
    with pytest.raises(ValueError):
        tp.add_custom("", tp.TrainPreset("Person"))
    with pytest.raises(ValueError):
        tp.add_custom("", tp.TrainPreset("  style  "))
    with pytest.raises(ValueError):
        tp.add_custom("", tp.TrainPreset("   "))


def test_parse_sanitizes_subject_and_builtin_flag():
    store = '[{"name": "X", "subject_type": "weird", "builtin": true}]'
    got = tp.parse_customs(store)[0]
    assert got.subject_type == "character"
    assert got.builtin is False


def test_summary_line():
    assert "auto steps" in tp.summary_line(tp.BUILTINS[0])
    p = tp.TrainPreset("X", optimizer="adamw8bit", learning_rate=1e-4,
                       target_steps=900, uncap_steps=True)
    line = tp.summary_line(p)
    assert "AdamW8bit" in line and "900 steps" in line and "uncapped" in line


def test_formula_line_differs_per_builtin():
    lines = [tp.formula_line(p) for p in tp.BUILTINS]
    # the small print is the visible difference between the three intents
    assert len(set(lines)) == 3
    assert "56" in lines[0] and "34" in lines[1] and "26" in lines[2]
    assert all("÷ 4" in l for l in lines)


def test_formula_line_fixed_steps():
    p = tp.TrainPreset("X", target_steps=1500)
    assert tp.formula_line(p) == "steps fixed at 1,500"
    p2 = tp.TrainPreset("Y", target_steps=5000, uncap_steps=True)
    assert "uncapped" in tp.formula_line(p2)


def test_presets_are_filtered_by_model():
    store = tp.add_custom("", tp.TrainPreset("Krea look", model="krea2", overrides={"learning_rate": 1}))
    store = tp.add_custom(store, tp.TrainPreset("Any model"))
    krea = [p.name for p in tp.presets_for(store, "krea2")]
    anima = [p.name for p in tp.presets_for(store, "anima")]
    assert "Krea look" in krea and "Krea look" not in anima
    assert "Any model" in krea and "Any model" in anima
    for b in tp.BUILTINS:
        assert b.name in krea and b.name in anima


def test_old_presets_load_as_any_model():
    [p] = tp.parse_customs('[{"name": "Old", "network_dim": 32}]')
    assert p.model == "" and p.overrides == {}


def test_preset_overrides_round_trip():
    store = tp.add_custom("", tp.TrainPreset("K", model="krea2",
                                             overrides={"optimizer.weight_decay": 0.01}))
    p = tp.find(store, "K")
    assert p.model == "krea2" and p.overrides == {"optimizer.weight_decay": 0.01}


def test_bad_model_or_overrides_fall_back_to_defaults():
    [p] = tp.parse_customs('[{"name": "X", "model": "sdxl", "overrides": [1, 2]}]')
    assert p.model == "" and p.overrides == {}
