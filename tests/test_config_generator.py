import sys
from pathlib import Path

import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config_generator import generate_configs, get_config_summary


def _gen(tmp_path, **kwargs):
    defaults = dict(
        output_dir=str(tmp_path),
        lora_name="test_lora",
        dit_path="C:/models/anima_baseV10.safetensors",
        qwen3_path="C:/models/qwen_3_06b_base.safetensors",
        vae_path="C:/models/qwen_image_vae.safetensors",
        dataset_folder=str(tmp_path / "imgs"),
        epochs=10,
        repeats=5,
    )
    defaults.update(kwargs)
    main_cfg, ds_cfg = generate_configs(**defaults)
    return toml.load(main_cfg), toml.load(ds_cfg)


def test_emits_anima_keys(tmp_path):
    main, _ = _gen(tmp_path)
    assert main["network_arguments"]["network_module"] == "networks.lora_anima"
    assert main["model_arguments"]["qwen3"].endswith("qwen_3_06b_base.safetensors")
    assert main["model_arguments"]["vae"].endswith("qwen_image_vae.safetensors")
    assert main["training_arguments"]["timestep_sampling"] == "sigmoid"
    assert main["training_arguments"]["vae_chunk_size"] == 64
    assert main["training_arguments"]["cache_latents_to_disk"] is True


def test_no_sdxl_keys(tmp_path):
    # Check KEY names only — pytest's tmp_path embeds this test's own name
    # ("test_no_sdxl_keys0"), so path VALUES always contain "sdxl" and a dump-wide
    # substring check fails on itself (the old "known flaky" failure).
    main, _ = _gen(tmp_path)

    def all_keys(d):
        for k, v in d.items():
            yield k
            if isinstance(v, dict):
                yield from all_keys(v)

    keys = " ".join(all_keys(main))
    for forbidden in ["sdxl", "clip_skip", "min_snr_gamma", "no_half_vae", "xformers"]:
        assert forbidden not in keys, f"{forbidden} should not appear in Anima config"


def test_prodigy_default(tmp_path):
    main, _ = _gen(tmp_path)
    opt = main["optimizer_arguments"]
    # Default "prodigy" now emits ProdigyPlusScheduleFree (learning-rate-free AND
    # schedule-free). sd-scripts loads it by module.Class path.
    assert opt["optimizer_type"] == "prodigyplus.ProdigyPlusScheduleFree"
    assert opt["learning_rate"] == 1.0
    # Schedule-free requires a constant scheduler.
    assert opt["lr_scheduler"] == "constant"
    assert "use_speed=True" in opt["optimizer_args"]
    # Classic-Prodigy-only args must not leak in (they're a different API).
    flat = " ".join(opt["optimizer_args"])
    for forbidden in ["decouple", "safeguard_warmup", "d_coef"]:
        assert forbidden not in flat


def test_adamw8bit_option(tmp_path):
    main, _ = _gen(tmp_path, optimizer="adamw8bit", learning_rate=2e-5)
    opt = main["optimizer_arguments"]
    assert opt["optimizer_type"] == "AdamW8bit"
    assert opt["learning_rate"] == 2e-5
    assert "optimizer_args" not in opt


def test_dit_only_by_default(tmp_path):
    main, _ = _gen(tmp_path)
    assert main["network_arguments"]["network_train_unet_only"] is True
    assert main["training_arguments"]["cache_text_encoder_outputs"] is True


def test_train_text_encoder_disables_caching(tmp_path):
    main, _ = _gen(tmp_path, train_text_encoder=True)
    assert "network_train_unet_only" not in main["network_arguments"]
    assert "cache_text_encoder_outputs" not in main["training_arguments"]


def test_dataset_no_shuffle(tmp_path):
    _, ds = _gen(tmp_path)
    assert ds["general"]["shuffle_caption"] is False
    assert ds["general"]["keep_tokens"] == 1
    assert ds["datasets"][0]["subsets"][0]["num_repeats"] == 5


def test_custom_dim_alpha(tmp_path):
    main, _ = _gen(tmp_path, network_dim=32, network_alpha=16)
    assert main["network_arguments"]["network_dim"] == 32
    assert main["network_arguments"]["network_alpha"] == 16


def test_extra_args_merged(tmp_path):
    main, _ = _gen(tmp_path, extra_args={"fp8_scaled": True, "weighting_scheme": "logit_normal"})
    t = main["training_arguments"]
    assert t["fp8_scaled"] is True
    assert t["weighting_scheme"] == "logit_normal"


def test_summary_mentions_anima():
    s = get_config_summary()
    assert "Anima" in s
    assert "Prodigy" in s


def test_bucketing_default_on(tmp_path):
    _, ds = _gen(tmp_path)
    g = ds["general"]
    assert g["enable_bucket"] is True
    assert g["min_bucket_reso"] == 512 and g["max_bucket_reso"] == 1536
    assert g["bucket_no_upscale"] is True


def test_bucketing_off(tmp_path):
    _, ds = _gen(tmp_path, enable_bucket=False)
    assert "enable_bucket" not in ds["general"]


def test_save_state_default(tmp_path):
    main, _ = _gen(tmp_path)
    t = main["training_arguments"]
    assert t["save_state"] is True
    assert t["save_last_n_epochs_state"] == 2


def test_resume_only_when_given(tmp_path):
    main, _ = _gen(tmp_path)
    assert "resume" not in main["training_arguments"]
    main2, _ = _gen(tmp_path, resume_state_path="C:/out/mylora-000004-state")
    assert main2["training_arguments"]["resume"].endswith("mylora-000004-state")


def test_network_weights_only_when_given(tmp_path):
    main, _ = _gen(tmp_path)
    assert "network_weights" not in main["network_arguments"]
    main2, _ = _gen(tmp_path, network_weights="C:/out/base.safetensors")
    assert main2["network_arguments"]["network_weights"].endswith("base.safetensors")


def test_speed_flags_present(tmp_path):
    main, _ = _gen(tmp_path)
    t = main["training_arguments"]
    assert t["persistent_data_loader_workers"] is True
    assert t["max_data_loader_n_workers"] == 2
    # TE outputs cached to disk (DiT-only default) for faster re-runs
    assert t["cache_text_encoder_outputs_to_disk"] is True


def test_te_disk_cache_off_when_training_te(tmp_path):
    main, _ = _gen(tmp_path, train_text_encoder=True)
    assert "cache_text_encoder_outputs_to_disk" not in main["training_arguments"]


def test_step_interval_state(tmp_path):
    main, _ = _gen(tmp_path, save_every_n_steps=250)
    t = main["training_arguments"]
    assert t["save_every_n_steps"] == 250
    assert t["save_last_n_steps_state"] == 2


def test_step_interval_off_when_zero(tmp_path):
    main, _ = _gen(tmp_path, save_every_n_steps=0)
    assert "save_every_n_steps" not in main["training_arguments"]


def test_step_interval_requires_save_state(tmp_path):
    main, _ = _gen(tmp_path, save_state=False, save_every_n_steps=250)
    t = main["training_arguments"]
    assert "save_every_n_steps" not in t
    assert "save_state" not in t


def test_lowvram_params_applied(tmp_path):
    main, ds = _gen(tmp_path, micro_batch=1, grad_accum=4, blocks_to_swap=8)
    t = main["training_arguments"]
    assert t["train_batch_size"] == 1
    assert t["gradient_accumulation_steps"] == 4
    assert t["blocks_to_swap"] == 8
    assert ds["datasets"][0]["batch_size"] == 1   # micro-batch flows to the dataset too
    assert "fp8_base" not in t                     # off unless opted in


def test_lowvram_fp8_opt_in(tmp_path):
    main, _ = _gen(tmp_path, micro_batch=1, grad_accum=4, blocks_to_swap=24, fp8_base=True)
    assert main["training_arguments"]["fp8_base"] is True


def test_no_lowvram_keys_by_default(tmp_path):
    # The whole safety guarantee: a normal run is unchanged — batch 4, no low-VRAM keys.
    main, ds = _gen(tmp_path)
    t = main["training_arguments"]
    assert t["train_batch_size"] == 4
    assert ds["datasets"][0]["batch_size"] == 4
    assert "gradient_accumulation_steps" not in t
    assert "blocks_to_swap" not in t
    assert "fp8_base" not in t


def test_training_comment(tmp_path):
    main, _ = _gen(tmp_path, training_comment="mychar")
    assert main["training_arguments"]["training_comment"] == "mychar"
    main2, _ = _gen(tmp_path)
    assert "training_comment" not in main2["training_arguments"]
