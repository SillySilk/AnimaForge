import os
import toml

from core import config_overrides
from core.kohya_recipes import recipe as kohya_recipe
from pathlib import Path

# Default training settings for the Anima architecture.
# Anima is a ~2B DiT model (NVIDIA Cosmos base) using a Qwen3-0.6B text encoder
# and the Qwen-Image VAE. It is NOT SDXL — none of the SDXL flags apply
# (v2, sdxl, clip_skip, xformers, no_half_vae, min_snr_gamma are all omitted).
# Override paths under this prefix address the dataset TOML instead of the main config.
DATASET_PREFIX = "dataset_toml."
# Settings the run itself sets — model files from Setup, the run folder and name,
# the dataset folder, and the step math (epochs, repeats). An override of one would
# freeze it across runs and desync the app's own bookkeeping, so they stay locked.
LOCKED_PATHS = (
    "model_arguments.*",
    "training_arguments.output_dir",
    "training_arguments.output_name",
    "training_arguments.logging_dir",
    "training_arguments.max_train_epochs",
    "training_arguments.sample_prompts",
    "training_arguments.resume",
    "network_arguments.network_weights",
    "dataset_arguments.dataset_config",
    DATASET_PREFIX + "datasets.*.subsets.*.image_dir",
    DATASET_PREFIX + "datasets.*.subsets.*.num_repeats",
)

HARDCODED_CONFIG = {
    # Output
    "save_model_as": "safetensors",
    "save_precision": "bf16",
    # Training (model-specific values — batch size, timestep sampling, VAE chunking,
    # network module — live in core/kohya_recipes.py)
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    # Network (LoRA)
    "network_dim": 16,
    "network_alpha": 8,
    # Caching
    "cache_latents": True,
    "cache_latents_to_disk": True,  # reuse latent cache across re-runs of the same dataset
    # Data loading — keep workers alive across epochs so they aren't torn down and
    # respawned every epoch (that respawn is the startup flood LogDenoiser hides).
    # Real per-epoch speedup at essentially no cost.
    "persistent_data_loader_workers": True,
    "max_data_loader_n_workers": 2,
    # Logging
    "log_with": "tensorboard",
}

# ProdigyPlusScheduleFree (LoganBooker/prodigy-plus-schedule-free): Prodigy's
# automatic, learning-rate-free adaptation combined with Schedule-Free training.
# It keeps the "set lr=1.0 and forget it" convenience that anchors concepts so
# reliably, but adds: (1) no LR schedule to configure — robust to stopping/resuming
# mid-run since it doesn't depend on a known total step count; (2) lower VRAM via
# an Adafactor-style factored second moment + StableAdamW + stochastic rounding
# (all on by default). sd-scripts has first-class support: the class name ends in
# "ScheduleFree" so sd-scripts auto-detects it, swaps in a dummy LR scheduler, and
# calls optimizer.train()/eval() around sampling/saving.
#
# `use_speed=True` is the one community-recommended extra (more stable across tasks,
# better when training multiple network groups). All other knobs are left at the
# author's defaults, which "just work" (factored, use_stableadamw, stochastic
# rounding on; experimental use_cautious/use_adopt/use_orthograd off).
PRODIGY_OPTIMIZER_TYPE = "prodigyplus.ProdigyPlusScheduleFree"
PRODIGY_OPTIMIZER_ARGS = [
    "use_speed=True",
]


def _saving_block(epochs: int, save_every_n_epochs, save_last_n_epochs) -> dict:
    """Epoch-save cadence. every: None/0 -> auto, a tenth of the run (so run safety
    has a snapshot from near any point to ship). Keep-last is never handed to
    sd-scripts: it keeps every epoch save until the run ends, and run safety
    (core/engines/kohya/safety.prune) applies the user's keep-last afterwards, so
    the best epoch is still on disk when an overshoot is found."""
    return {
        "save_every_n_epochs": int(save_every_n_epochs) if save_every_n_epochs
        else max(1, round(epochs / 10)),
    }


def generate_configs(
    output_dir: str,
    lora_name: str,
    dit_path: str,
    qwen3_path: str,
    vae_path: str,
    dataset_folder: str,
    epochs: int,
    repeats: int,
    optimizer: str = "prodigy",
    learning_rate: float = None,
    network_dim: int = None,
    network_alpha: int = None,
    train_text_encoder: bool = False,
    resolution: int = None,
    enable_bucket: bool = True,
    save_state: bool = True,
    save_every_n_steps: int = None,
    save_every_n_epochs: int = None,
    save_last_n_epochs: int = None,
    resume_state_path: str = None,
    network_weights: str = None,
    training_comment: str = None,
    extra_args: dict = None,
    micro_batch: int = None,
    grad_accum: int = None,
    blocks_to_swap: int = None,
    fp8_base: bool = False,
    overrides: dict = None,
    model: str = "anima",
    memory_flags: dict = None,
) -> tuple:
    """
    Generate the main Anima training TOML config and the dataset TOML config.

    Args:
        dit_path:   Anima DiT checkpoint (.safetensors) -> --pretrained_model_name_or_path
        qwen3_path: Qwen3-0.6B text encoder -> --qwen3
        vae_path:   Qwen-Image VAE -> --vae
        optimizer:  "prodigy" (default → ProdigyPlusScheduleFree, learning-rate-free
                    and schedule-free) or "adamw8bit"
        learning_rate: only used for adamw8bit (defaults to 1e-4)
        train_text_encoder: if True, also train the Qwen3 encoder (no TE-output caching)
        resolution: base training resolution (square). None/0 → 1024. The bucket window
                    stays 512–1536 (Anima's supported span) regardless.
        save_every_n_epochs: LoRA epoch-save cadence. None/0 → auto (max(1, epochs//5)).
        save_last_n_epochs:  how many epoch saves to keep. None → 3 (legacy default);
                    0 → keep all (key omitted).

        model:      Kohya model id; its recipe (core/kohya_recipes.py) supplies the
                    batch size, network module, text-encoder key and timestep args.
        memory_flags: extra training-block keys (e.g. convrot_int8, block-swap flags),
                    written after the low-VRAM knobs.

    Returns (main_config_path, dataset_config_path). Raises on error.
    """
    rec = kohya_recipe(model)
    if fp8_base and not rec.allow_text_encoder_training:
        raise ValueError(f"fp8_base is not supported for model {model!r}")
    if not rec.allow_text_encoder_training:
        train_text_encoder = False  # frozen text encoder, always
    # Advanced-settings overrides: "dataset_toml.*" paths address the dataset TOML,
    # every other path the main config.
    main_overrides, dataset_overrides = {}, {}
    for path, value in (overrides or {}).items():
        if path.startswith(DATASET_PREFIX):
            dataset_overrides[path[len(DATASET_PREFIX):]] = value
        else:
            main_overrides[path] = value
    configs_dir = Path(output_dir) / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    dim = network_dim if network_dim is not None else HARDCODED_CONFIG["network_dim"]
    alpha = network_alpha if network_alpha is not None else HARDCODED_CONFIG["network_alpha"]
    # Low-VRAM: micro-batch shrinks GPU activation memory; gradient accumulation keeps the
    # effective batch (and thus training dynamics / quality / step count) unchanged.
    batch = micro_batch if micro_batch else rec.batch_size
    res = int(resolution) if resolution else 1024

    # ---- Dataset config ----
    # shuffle_caption is OFF: combined captions contain natural-language sentences
    # (with commas) that must not be shuffled. keep_tokens protects the trigger word.
    dataset_toml_path = configs_dir / f"{lora_name}_dataset.toml"
    dataset_config = {
        "general": {
            "shuffle_caption": False,
            "caption_extension": ".txt",
            "keep_tokens": 1,
        },
        "datasets": [
            {
                "resolution": res,
                "batch_size": batch,
                "subsets": [
                    {
                        "image_dir": str(Path(dataset_folder).as_posix()),
                        "num_repeats": repeats,
                    }
                ],
            }
        ],
    }
    # Aspect-ratio bucketing: train on non-square images without cropping.
    if enable_bucket:
        dataset_config["general"].update({
            "enable_bucket": True,
            "min_bucket_reso": 512,
            "max_bucket_reso": rec.max_bucket_reso,  # Anima: 512–1536
            "bucket_reso_steps": 64,
            "bucket_no_upscale": True,
        })
    if dataset_overrides:
        dataset_config = config_overrides.apply(dataset_config, dataset_overrides)
    with open(dataset_toml_path, "w", encoding="utf-8") as f:
        toml.dump(dataset_config, f)

    # ---- Optimizer block ----
    if optimizer.lower() == "prodigy":
        # Schedule-free: a constant scheduler is required. sd-scripts also detects
        # the "ScheduleFree" suffix and substitutes its own dummy scheduler, but we
        # set constant explicitly so the intent is clear and correct either way.
        optimizer_block = {
            "optimizer_type": PRODIGY_OPTIMIZER_TYPE,
            "optimizer_args": PRODIGY_OPTIMIZER_ARGS,
            "lr_scheduler": "constant",
            "learning_rate": 1.0,
        }
    else:  # adamw8bit
        lr = learning_rate if learning_rate else 1e-4
        optimizer_block = {
            "optimizer_type": "AdamW8bit",
            "lr_scheduler": "constant",
            "learning_rate": lr,
        }

    # ---- Network block ----
    network_block = {
        "network_module": rec.network_module,
        "network_dim": dim,
        "network_alpha": alpha,
    }
    if not train_text_encoder:
        network_block["network_train_unet_only"] = True
    if network_weights:
        network_block["network_weights"] = str(Path(network_weights).as_posix())

    # ---- Training block ----
    training_block = {
        "output_dir": str(Path(output_dir).as_posix()),
        "output_name": lora_name,
        "save_model_as": HARDCODED_CONFIG["save_model_as"],
        "save_precision": HARDCODED_CONFIG["save_precision"],
        "resolution": f"{res},{res}",
        "train_batch_size": batch,
        "mixed_precision": HARDCODED_CONFIG["mixed_precision"],
        "gradient_checkpointing": HARDCODED_CONFIG["gradient_checkpointing"],
        **rec.timestep_args,
        "max_train_epochs": epochs,
        "cache_latents": HARDCODED_CONFIG["cache_latents"],
        "cache_latents_to_disk": HARDCODED_CONFIG["cache_latents_to_disk"],
        **({"vae_chunk_size": rec.vae_chunk_size} if rec.vae_chunk_size is not None else {}),
        "persistent_data_loader_workers": HARDCODED_CONFIG["persistent_data_loader_workers"],
        "max_data_loader_n_workers": HARDCODED_CONFIG["max_data_loader_n_workers"],
        "log_with": HARDCODED_CONFIG["log_with"],
        "logging_dir": str((Path(output_dir) / "logs").as_posix()),
    }
    # Cache text-encoder outputs only when we are NOT training the encoder.
    # Cache to disk as well (like latents) so Qwen3 outputs are reused across re-runs
    # instead of recomputed each time — faster start on repeated trainings.
    if not train_text_encoder:
        training_block["cache_text_encoder_outputs"] = True
        training_block["cache_text_encoder_outputs_to_disk"] = True
    # Resumable state, resume path, and Civitai-style trigger metadata.
    if save_state:
        training_block["save_state"] = True
        training_block["save_last_n_epochs_state"] = 2  # bound disk; prune old states
        # Step-interval state lets the user Stop mid-epoch and resume losing at most
        # save_every_n_steps of progress (accelerate can't flush on a hard kill, so a
        # Stop resumes from the last saved checkpoint, not the exact step clicked).
        if save_every_n_steps and save_every_n_steps > 0:
            training_block["save_every_n_steps"] = int(save_every_n_steps)
            training_block["save_last_n_steps_state"] = 2
    if resume_state_path:
        training_block["resume"] = str(Path(resume_state_path).as_posix())
    if training_comment:
        training_block["training_comment"] = training_comment
    # Low-VRAM knobs (only present when low-VRAM mode is active — otherwise omitted entirely
    # so a normal run's config is byte-for-byte unchanged).
    if grad_accum and grad_accum > 1:
        training_block["gradient_accumulation_steps"] = int(grad_accum)
    if blocks_to_swap and blocks_to_swap > 0:
        training_block["blocks_to_swap"] = int(blocks_to_swap)
    if fp8_base:
        training_block["fp8_base"] = True
    # Model memory flags (INT8 / block-swap tuning) sit after the low-VRAM knobs.
    if memory_flags:
        training_block.update(memory_flags)
    # Advanced/sample knobs from Settings (only the enabled ones are present).
    if extra_args:
        training_block.update(extra_args)

    # ---- Main training config ----
    main_config_path = configs_dir / f"{lora_name}_config.toml"
    config = {
        "model_arguments": {
            "pretrained_model_name_or_path": str(Path(dit_path).as_posix()),
            rec.text_encoder_arg: str(Path(qwen3_path).as_posix()),
            "vae": str(Path(vae_path).as_posix()),
        },
        "training_arguments": training_block,
        "optimizer_arguments": optimizer_block,
        "network_arguments": network_block,
        "dataset_arguments": {
            "dataset_config": str(dataset_toml_path.as_posix()),
        },
        "saving_arguments": _saving_block(epochs, save_every_n_epochs, save_last_n_epochs),
    }

    if main_overrides:
        config = config_overrides.apply(config, main_overrides)
    with open(main_config_path, "w", encoding="utf-8") as f:
        toml.dump(config, f)

    return str(main_config_path), str(dataset_toml_path)


def get_config_summary(optimizer: str = "prodigy", network_dim: int = None,
                       network_alpha: int = None, train_text_encoder: bool = False,
                       resolution: int = None, model: str = "anima") -> str:
    """Return a human-readable summary of the training settings for `model`."""
    rec = kohya_recipe(model)
    name = "Anima" if model == "anima" else "MOSS-Image"
    arch = ("Anima (DiT + Qwen3 + Qwen VAE)" if model == "anima"
            else "MOSS-Image (DiT + Qwen3.5 + FLUX.2 VAE)")
    dim = network_dim if network_dim is not None else HARDCODED_CONFIG["network_dim"]
    alpha = network_alpha if network_alpha is not None else HARDCODED_CONFIG["network_alpha"]
    res = int(resolution) if resolution else 1024
    opt_label = "Prodigy+ ScheduleFree (auto LR)" if optimizer.lower() == "prodigy" else "AdamW8bit"
    te_label = "DiT + Text Encoder" if train_text_encoder else "DiT only"
    lines = [
        f"── {name} Training Settings ──",
        f"  Architecture : {arch}",
        f"  Resolution   : {res},{res}",
        f"  Batch Size   : {rec.batch_size}",
        f"  Precision    : {HARDCODED_CONFIG['mixed_precision']}",
        f"  Optimizer    : {opt_label}",
        f"  Trains       : {te_label}",
        f"  Timestep     : {rec.timestep_args['timestep_sampling']}",
        f"  Network Dim  : {dim}",
        f"  Network Alpha: {alpha}",
        f"  Grad Ckpt    : {HARDCODED_CONFIG['gradient_checkpointing']}",
        f"  Cache Latents: {HARDCODED_CONFIG['cache_latents']}",
        f"  VAE Chunk    : {rec.vae_chunk_size if rec.vae_chunk_size is not None else 'n/a'}",
        f"  Save As      : {HARDCODED_CONFIG['save_model_as']}",
        "─────────────────────────────",
    ]
    return "\n".join(lines)
