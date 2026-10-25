import pytest
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


def test_resolution_default_1024(tmp_path):
    main, ds = _gen(tmp_path)
    assert main["training_arguments"]["resolution"] == "1024,1024"
    assert ds["datasets"][0]["resolution"] == 1024


def test_resolution_choice_flows_to_both_tomls(tmp_path):
    main, ds = _gen(tmp_path, resolution=1536)
    assert main["training_arguments"]["resolution"] == "1536,1536"
    assert ds["datasets"][0]["resolution"] == 1536


def test_resolution_leaves_bucket_range_alone(tmp_path):
    # The bucket window is Anima's supported span, independent of the base resolution.
    _, ds = _gen(tmp_path, resolution=512)
    g = ds["general"]
    assert g["min_bucket_reso"] == 512 and g["max_bucket_reso"] == 1536


def test_epoch_save_cadence_defaults_to_auto_formula(tmp_path):
    main, _ = _gen(tmp_path)  # epochs=10 -> every tenth of the run = every epoch
    s = main["saving_arguments"]
    assert s["save_every_n_epochs"] == 1
    # keep-last is applied by run safety after the run, never handed to sd-scripts
    assert "save_last_n_epochs" not in s


def test_epoch_save_cadence_override(tmp_path):
    main, _ = _gen(tmp_path, save_every_n_epochs=1, save_last_n_epochs=8)
    s = main["saving_arguments"]
    assert s["save_every_n_epochs"] == 1
    assert "save_last_n_epochs" not in s


def test_epoch_save_zero_means_auto(tmp_path):
    # 0 is the UI spinner's "auto" sentinel — same as not passing it.
    main, _ = _gen(tmp_path, save_every_n_epochs=0)
    assert main["saving_arguments"]["save_every_n_epochs"] == 1


def test_keep_all_epoch_saves_when_last_n_zero(tmp_path):
    # 0 = keep every epoch save (omit the pruning key entirely).
    main, _ = _gen(tmp_path, save_every_n_epochs=1, save_last_n_epochs=0)
    assert "save_last_n_epochs" not in main["saving_arguments"]


def test_summary_shows_chosen_resolution():
    s = get_config_summary(resolution=1536)
    assert "1536,1536" in s


def test_training_comment(tmp_path):
    main, _ = _gen(tmp_path, training_comment="mychar")
    assert main["training_arguments"]["training_comment"] == "mychar"
    main2, _ = _gen(tmp_path)
    assert "training_comment" not in main2["training_arguments"]


@pytest.mark.parametrize("epochs,every", [(10, 1), (14, 1), (20, 2), (25, 2), (30, 3)])
def test_auto_epoch_saves_land_every_tenth(tmp_path, epochs, every):
    main, _ = _gen(tmp_path, epochs=epochs)
    assert main["saving_arguments"]["save_every_n_epochs"] == every


def test_keep_last_is_not_handed_to_sd_scripts(tmp_path):
    main, _ = _gen(tmp_path, save_last_n_epochs=3)
    assert "save_last_n_epochs" not in main["saving_arguments"]


# ---- Anima golden lock + MOSS recipe ----

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "anima_golden"
# Golden files were captured from the pre-refactor generator with a temp output
# root; every occurrence of that root is stored as "<ROOT>". The tests swap the
# run's tmp_path (posix form) for "<ROOT>" and compare byte-for-byte otherwise.
GOLDEN_ARGS = {
    "a": {},
    "b": dict(micro_batch=1, grad_accum=4, blocks_to_swap=8,
              resume_state_path="D:/state/epoch-3-state",
              network_weights="D:/loras/prev.safetensors", save_every_n_steps=250,
              training_comment="trigger: xyz", optimizer="adamw8bit", learning_rate=2e-4,
              network_dim=32, network_alpha=16, resolution=768,
              extra_args={"sample_every_n_epochs": 1, "sample_prompts": "D:/p.txt", "seed": 42}),
}


@pytest.mark.parametrize("key", ["a", "b"])
def test_anima_configs_match_golden(tmp_path, key):
    kw = dict(output_dir=str(tmp_path), lora_name="golden",
              dit_path="C:/models/anima.safetensors",
              qwen3_path="C:/models/qwen3.safetensors",
              vae_path="C:/models/vae.safetensors",
              dataset_folder=str(tmp_path / "imgs"), epochs=10, repeats=5,
              model="anima")
    kw.update(GOLDEN_ARGS[key])
    main_cfg, ds_cfg = generate_configs(**kw)
    for path, suffix in ((main_cfg, "config"), (ds_cfg, "dataset")):
        got = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(tmp_path.as_posix().encode(), b"<ROOT>")
        want = (GOLDEN_DIR / f"{key}_{suffix}.toml").read_bytes().replace(b"\r\n", b"\n")
        assert got == want


def test_moss_config_core_keys(tmp_path):
    main, _ = _gen(tmp_path, model="moss")
    assert main["network_arguments"]["network_module"] == "networks.lora_generic"
    mdl = main["model_arguments"]
    assert "text_encoder" in mdl and "qwen3" not in mdl
    t = main["training_arguments"]
    assert t["train_batch_size"] == 1
    assert t["timestep_sampling"] == "sigma"
    assert t["weighting_scheme"] == "logit_normal"
    assert t["logit_mean"] == -1.09
    assert t["logit_std"] == 0.95
    assert t["discrete_flow_shift"] == 1.0
    assert "vae_chunk_size" not in t
    assert main["network_arguments"]["network_train_unet_only"] is True
    assert t["cache_text_encoder_outputs"] is True


def test_moss_forces_frozen_te(tmp_path):
    main, _ = _gen(tmp_path, model="moss", train_text_encoder=True)
    assert main["network_arguments"]["network_train_unet_only"] is True
    assert main["training_arguments"]["cache_text_encoder_outputs"] is True


def test_moss_memory_flags_written(tmp_path):
    flags = {"convrot_int8": True, "blocks_to_swap": 28, "block_swap_h2d_only": True}
    main, _ = _gen(tmp_path, model="moss", memory_flags=flags)
    for k, v in flags.items():
        assert main["training_arguments"][k] == v


def test_moss_rejects_fp8(tmp_path):
    with pytest.raises(ValueError, match="fp8"):
        _gen(tmp_path, model="moss", fp8_base=True)
