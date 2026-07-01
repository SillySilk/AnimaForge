import os
import toml
from pathlib import Path

# Default training settings for the Anima architecture.
# Anima is a ~2B DiT model (NVIDIA Cosmos base) using a Qwen3-0.6B text encoder
# and the Qwen-Image VAE. It is NOT SDXL — none of the SDXL flags apply
# (v2, sdxl, clip_skip, xformers, no_half_vae, min_snr_gamma are all omitted).
HARDCODED_CONFIG = {
    # Output
    "save_model_as": "safetensors",
    "save_precision": "bf16",
    # Training
    "resolution": "1024,1024",
    "batch_size": 4,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "timestep_sampling": "sigmoid",
    "vae_chunk_size": 64,
    # Network (LoRA) — Anima module
    "network_module": "networks.lora_anima",
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
    enable_bucket: bool = True,
    save_state: bool = True,
    save_every_n_steps: int = None,
    resume_state_path: str = None,
    network_weights: str = None,
    training_comment: str = None,
    extra_args: dict = None,
    micro_batch: int = None,
    grad_accum: int = None,
    blocks_to_swap: int = None,
    fp8_base: bool = False,
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

    Returns (main_config_path, dataset_config_path). Raises on error.
    """
    configs_dir = Path(output_dir) / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    dim = network_dim if network_dim is not None else HARDCODED_CONFIG["network_dim"]
    alpha = network_alpha if network_alpha is not None else HARDCODED_CONFIG["network_alpha"]
    # Low-VRAM: micro-batch shrinks GPU activation memory; gradient accumulation keeps the
    # effective batch (and thus training dynamics / quality / step count) unchanged.
    batch = micro_batch if micro_batch else HARDCODED_CONFIG["batch_size"]

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
                "resolution": 1024,
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
            "max_bucket_reso": 1536,  # Anima's supported range is 512–1536
            "bucket_reso_steps": 64,
            "bucket_no_upscale": True,
        })
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
        "network_module": HARDCODED_CONFIG["network_module"],
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
        "resolution": HARDCODED_CONFIG["resolution"],
        "train_batch_size": batch,
        "mixed_precision": HARDCODED_CONFIG["mixed_precision"],
        "gradient_checkpointing": HARDCODED_CONFIG["gradient_checkpointing"],
        "timestep_sampling": HARDCODED_CONFIG["timestep_sampling"],
        "max_train_epochs": epochs,
        "cache_latents": HARDCODED_CONFIG["cache_latents"],
        "cache_latents_to_disk": HARDCODED_CONFIG["cache_latents_to_disk"],
        "vae_chunk_size": HARDCODED_CONFIG["vae_chunk_size"],
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
    # Advanced/sample knobs from Settings (only the enabled ones are present).
    if extra_args:
        training_block.update(extra_args)

    # ---- Main training config ----
    main_config_path = configs_dir / f"{lora_name}_config.toml"
    config = {
        "model_arguments": {
            "pretrained_model_name_or_path": str(Path(dit_path).as_posix()),
            "qwen3": str(Path(qwen3_path).as_posix()),
            "vae": str(Path(vae_path).as_posix()),
        },
        "training_arguments": training_block,
        "optimizer_arguments": optimizer_block,
        "network_arguments": network_block,
        "dataset_arguments": {
            "dataset_config": str(dataset_toml_path.as_posix()),
        },
        "saving_arguments": {
            "save_every_n_epochs": max(1, epochs // 5),
            "save_last_n_epochs": 3,
        },
    }

    with open(main_config_path, "w", encoding="utf-8") as f:
        toml.dump(config, f)

    return str(main_config_path), str(dataset_toml_path)


def get_config_summary(optimizer: str = "prodigy", network_dim: int = None,
                       network_alpha: int = None, train_text_encoder: bool = False) -> str:
    """Return a human-readable summary of the Anima training settings."""
    dim = network_dim if network_dim is not None else HARDCODED_CONFIG["network_dim"]
    alpha = network_alpha if network_alpha is not None else HARDCODED_CONFIG["network_alpha"]
    opt_label = "Prodigy+ ScheduleFree (auto LR)" if optimizer.lower() == "prodigy" else "AdamW8bit"
    te_label = "DiT + Text Encoder" if train_text_encoder else "DiT only"
    lines = [
        "── Anima Training Settings ──",
        f"  Architecture : Anima (DiT + Qwen3 + Qwen VAE)",
        f"  Resolution   : {HARDCODED_CONFIG['resolution']}",
        f"  Batch Size   : {HARDCODED_CONFIG['batch_size']}",
        f"  Precision    : {HARDCODED_CONFIG['mixed_precision']}",
        f"  Optimizer    : {opt_label}",
        f"  Trains       : {te_label}",
        f"  Timestep     : {HARDCODED_CONFIG['timestep_sampling']}",
        f"  Network Dim  : {dim}",
        f"  Network Alpha: {alpha}",
        f"  Grad Ckpt    : {HARDCODED_CONFIG['gradient_checkpointing']}",
        f"  Cache Latents: {HARDCODED_CONFIG['cache_latents']}",
        f"  VAE Chunk    : {HARDCODED_CONFIG['vae_chunk_size']}",
        f"  Save As      : {HARDCODED_CONFIG['save_model_as']}",
        "─────────────────────────────",
    ]
    return "\n".join(lines)
