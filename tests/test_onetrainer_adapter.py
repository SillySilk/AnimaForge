import pytest

from core.batch import RunDefinition
from core.engines.onetrainer.adapter import to_onetrainer_params


def run(**kw):
    base = dict(lora_name="Amber", dataset_folder="C:/sets/amber", image_count=25,
                trigger_word="amberh", subject_type="character", quality_prefix="masterpiece",
                output_dir="C:/out", model="krea2", onetrainer_root="C:/OT", target_steps=0)
    base.update(kw)
    return RunDefinition(**base)


def test_maps_every_field():
    p = to_onetrainer_params(run())
    assert (p.lora_name, p.dataset_dir, p.image_count, p.concept, p.trigger_word,
            p.output_base, p.onetrainer_root, p.quality_prefix, p.target_steps) == (
        "Amber", "C:/sets/amber", 25, "character", "amberh", "C:/out", "C:/OT",
        "masterpiece", 0)


def test_blank_subject_means_character():
    assert to_onetrainer_params(run(subject_type="")).concept == "character"


def test_krea2_run_uses_auto_steps_unless_set():
    # 500 is Anima's default; carried into a Krea run it would silently halve a
    # 25-image character set's proven 1000 steps. The adapter honours what the run
    # explicitly holds; new runs are seeded from models.get(id).default_target_steps.
    assert to_onetrainer_params(run(target_steps=500)).target_steps == 500
    assert to_onetrainer_params(run(target_steps=0)).target_steps == 0


def test_non_onetrainer_model_is_refused():
    with pytest.raises(ValueError, match="anima"):
        to_onetrainer_params(run(model="anima"))


def test_advanced_overrides_travel_to_the_session():
    p = to_onetrainer_params(run(config_overrides={"learning_rate": 1}))
    assert p.config_overrides == {"learning_rate": 1}
