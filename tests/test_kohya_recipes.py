import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.kohya_recipes import KohyaRecipe, recipe


def test_recipe_unknown_raises():
    with pytest.raises(KeyError):
        recipe("krea2")


def test_anima_recipe_values():
    r = recipe("anima")
    assert isinstance(r, KohyaRecipe)
    assert (r.train_script, r.network_module, r.text_encoder_arg) == (
        "anima_train_network.py", "networks.lora_anima", "qwen3")
    assert (r.batch_size, r.max_blocks_to_swap, r.vae_chunk_size) == (4, 26, 64)
    assert r.timestep_args == {"timestep_sampling": "sigmoid"}
    assert r.allow_text_encoder_training is True and r.sample_flags == ""


def test_moss_recipe_values():
    r = recipe("moss")
    assert (r.train_script, r.network_module, r.text_encoder_arg) == (
        "moss_image_train_network.py", "networks.lora_generic", "text_encoder")
    assert (r.batch_size, r.max_blocks_to_swap, r.vae_chunk_size) == (1, 40, None)
    assert list(r.timestep_args) == [
        "timestep_sampling", "weighting_scheme", "logit_mean", "logit_std", "discrete_flow_shift"]
    assert r.sample_flags == "--s 24 --l 3.0 --fs 5.0"
    assert r.allow_text_encoder_training is False


def test_preview_cadence_per_model():
    # MOSS previews are slow (~90 s each), so they render every third epoch.
    assert recipe("anima").sample_every_n_epochs == 1
    assert recipe("moss").sample_every_n_epochs == 3
