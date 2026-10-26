"""Per-model sd-scripts (Kohya) training recipes.

One frozen record per model that trains through sd-scripts. `config_generator`
reads it, so adding a model means adding an entry here rather than editing the
TOML writer. Models that don't train through sd-scripts have no recipe.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class KohyaRecipe:
    train_script: str           # "anima_train_network.py" | "moss_image_train_network.py"
    network_module: str         # "networks.lora_anima" | "networks.lora_generic"
    text_encoder_arg: str       # model_arguments key: "qwen3" | "text_encoder"
    batch_size: int             # 4 | 1
    max_blocks_to_swap: int     # 26 | 40
    timestep_args: dict         # training-block keys, in order
    vae_chunk_size: int | None  # 64 | None (omitted for MOSS)
    max_bucket_reso: int        # 1536 | 1536
    sample_flags: str           # "" | "--s 24 --l 3.0 --fs 5.0"
    sample_every_n_epochs: int  # 1 | 3 (MOSS previews take ~90 s each)
    allow_text_encoder_training: bool  # True | False


_RECIPES = {
    "anima": KohyaRecipe(
        train_script="anima_train_network.py",
        network_module="networks.lora_anima",
        text_encoder_arg="qwen3",
        batch_size=4,
        max_blocks_to_swap=26,
        timestep_args={"timestep_sampling": "sigmoid"},
        vae_chunk_size=64,
        max_bucket_reso=1536,
        sample_flags="",
        sample_every_n_epochs=1,
        allow_text_encoder_training=True,
    ),
    "moss": KohyaRecipe(
        train_script="moss_image_train_network.py",
        network_module="networks.lora_generic",
        text_encoder_arg="text_encoder",
        batch_size=1,
        max_blocks_to_swap=40,
        timestep_args={
            "timestep_sampling": "sigma",
            "weighting_scheme": "logit_normal",
            "logit_mean": -1.09,
            "logit_std": 0.95,
            "discrete_flow_shift": 1.0,
        },
        vae_chunk_size=None,
        max_bucket_reso=1536,
        sample_flags="--s 24 --l 3.0 --fs 5.0",
        sample_every_n_epochs=3,
        allow_text_encoder_training=False,
    ),
}


def recipe(model_id: str) -> KohyaRecipe:
    """Recipe for a Kohya-trained model. KeyError for unknown / non-Kohya ids."""
    try:
        return _RECIPES[model_id]
    except KeyError:
        raise KeyError(f"No Kohya recipe for model {model_id!r}") from None
