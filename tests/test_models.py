import pytest

from core import models


def test_anima_preset_is_kohya_with_prose_and_tags():
    m = models.get("anima")
    assert (m.name, m.engine, m.caption_style) == ("Anima", models.KOHYA, "prose+tags")
    assert m.default_target_steps == 500


def test_krea2_preset_is_onetrainer_with_prose_and_tags_and_auto_steps():
    m = models.get("krea2")
    assert (m.name, m.engine, m.caption_style) == ("Krea 2", models.ONETRAINER, "prose+tags")
    assert m.default_target_steps == 0  # 0 = Auto (Krea's per-concept steps rate)


def test_every_preset_uses_a_known_engine_and_caption_style():
    for m in models.MODELS.values():
        assert m.engine in (models.KOHYA, models.ONETRAINER)
        assert m.caption_style in models.CAPTION_STYLES
        assert models.MODELS[m.id] is m


def test_unknown_model_is_an_error_naming_it():
    with pytest.raises(KeyError, match="flux9"):
        models.get("flux9")


def test_default_model_is_anima():
    assert models.DEFAULT_MODEL == "anima"


def test_moss_preset_fields():
    p = models.get("moss")
    assert p.engine == models.KOHYA and p.caption_style == "prose+tags"
    assert p.min_vram_gb == 12 and p.forge_test is False and p.auto_character_steps
    assert p.default_target_steps == 2500 and p.steps is models.MOSS_STEPS


def test_existing_presets_keep_their_meaning():
    assert models.get("anima").steps is models.ANIMA_STEPS
    assert models.get("anima").forge_test and not models.get("anima").auto_character_steps
    assert models.get("krea2").steps is None


def test_old_rundefinition_dict_loads_as_anima():
    from core.batch import RunDefinition
    d = RunDefinition("n", "C:/ds", 12).to_dict()
    del d["model"]
    d["target_steps"], d["caption_style"] = 1234, "prose"
    r = RunDefinition.from_dict(d)
    assert r.model == "anima" and r.target_steps == 1234 and r.caption_style == "prose"


def test_only_moss_picks_its_own_memory():
    assert [m.id for m in models.MODELS.values() if m.auto_memory] == ["moss"]
