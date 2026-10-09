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
