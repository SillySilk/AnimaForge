import json
from core.engines.onetrainer import preset

def test_template_loads_and_is_version_pinned():
    t = preset.load_template()
    assert t["__version"] == preset.EXPECTED_TEMPLATE_VERSION

def test_recipe_guard_locks_the_proven_knobs():
    """The recipe that trains a 12B model on 16 GB. A re-exported preset that loses
    any of these is a different recipe and must fail here, loudly."""
    t = preset.load_template()
    assert t["model_type"] == "KREA_2"
    assert t["base_model_name"] == "krea/Krea-2-Raw"
    assert t["training_method"] == "LORA"
    # Compressed since 2026-09-04: same int8 weights, nvCOMP-packed on the card.
    assert t["transformer"]["weight_dtype"] == "INT_W8A8_COMPRESSED"
    assert t["transformer"]["offload_fraction"] == 0.3
    assert t["transformer"]["gradient_checkpointing"] is True
    assert t["batch_size"] == 2
    assert t["resolution"] == "512"
    assert t["lora_rank"] == 32
    assert t["output_model_format"] == "COMFY_LORA"
    assert t["optimizer"]["optimizer"] == "ADAMW_8BIT"
    assert t["cloud"]["enabled"] is False


def test_adamw_runs_at_a_fixed_rate_the_krea2_guides_use():
    """Prodigy blew up three Krea 2 runs (run_H, run_S, ModernFam 2026-10-12), the
    last one while capped. AdamW8bit at a fixed 1e-4 is what the Krea 2 character
    guides train at, and an AdamW-sized rate is a real step size here, not a
    multiplier: 1.0 would wreck the run in a step."""
    t = preset.load_template()
    assert t["optimizer"]["optimizer"] == "ADAMW_8BIT"
    assert t["learning_rate"] == 1e-4
    assert t["optimizer"]["weight_decay"] == 0.01
    assert t["learning_rate_scheduler"] == "CONSTANT"
    assert t["learning_rate_warmup_steps"] > 0


def test_output_format_matches_the_naming_the_inference_side_reads():
    """Every Krea 2 LoRA that actually works in the image tool is `diffusion_model.`
    -prefixed with the native module names (blocks.N.attn.wq, blocks.N.mlp.up).
    That is OneTrainer's COMFY_LORA. DIFFUSERS_LORA writes the diffusers namespace
    instead (transformer.transformer_blocks.N.attn.to_q, .ff.up) -- not one key in
    common, so the loader matches nothing and silently applies nothing."""
    t = preset.load_template()
    assert t["output_model_format"] == "COMFY_LORA"


def test_lora_alpha_equals_rank_so_the_applied_scale_is_one():
    """A LoRA module scales by alpha/rank. alpha=1 at rank 16 trains a module that
    contributes at 1/16 strength, and whether that survives export depends on the
    loader honouring the alpha key -- a loader that assumes alpha==rank would apply
    it 16x too strong instead. alpha == rank is scale 1.0 under either behaviour,
    and matches the reference LoRAs."""
    t = preset.load_template()
    assert t["lora_alpha"] == float(t["lora_rank"])

def _put(cfg: dict, dotted: str, value):
    *parents, leaf = dotted.split(".")
    for k in parents:
        cfg = cfg[k]
    cfg[leaf] = value


def _get(cfg: dict, dotted: str):
    for k in dotted.split("."):
        cfg = cfg[k]
    return cfg


def test_patch_touches_only_the_declared_fields():
    t = preset.load_template()
    cfg = preset.build_run_config(t, r"C:\out\MyLora", "MyLora", 12, 50, prodigy_steps=900)
    for key in preset.PATCHED_FIELDS:
        _put(cfg, key, _get(t, key))
    assert cfg == t

def test_warmup_is_patched_so_it_scales_with_the_run():
    """A shortened run must not spend most of itself ramping the LR: the preset's
    200-step warmup is ~10% of the reference run, not a constant."""
    t = preset.load_template()
    assert t["learning_rate_warmup_steps"] == 200      # the proven anchor
    cfg = preset.build_run_config(t, r"C:\out\MyLora", "MyLora", 21, 50, 300)
    assert cfg["learning_rate_warmup_steps"] == 50
    assert isinstance(cfg["learning_rate_warmup_steps"], float)   # OneTrainer types it float

def test_patched_paths_are_absolute_and_epochs_int():
    t = preset.load_template()
    cfg = preset.build_run_config(t, r"C:\out\MyLora", "MyLora", 12, 50, 300)
    assert cfg["epochs"] == 12
    assert cfg["concept_file_name"] == r"C:\out\MyLora\concepts.json"
    assert cfg["sample_definition_file_name"] == r"C:\out\MyLora\samples.json"
    assert cfg["workspace_dir"] == r"C:\out\MyLora\workspace"
    assert cfg["cache_dir"] == r"C:\out\MyLora\cache"
    assert cfg["output_model_destination"] == r"C:\out\MyLora\MyLora.safetensors"

def test_previews_land_every_tenth_of_the_run_rounded_up_to_an_epoch():
    t = preset.load_template()
    for epochs, every in ((80, 8), (67, 7), (12, 2), (10, 1), (5, 1), (1, 1)):
        cfg = preset.build_run_config(t, "R", "X", epochs, 10, 8)
        assert cfg["sample_after"] == every
        assert cfg["sample_after_unit"] == "EPOCH"


def test_template_is_not_mutated_by_patching():
    t = preset.load_template()
    before = json.dumps(t, sort_keys=True)
    preset.build_run_config(t, r"C:\x", "X", 1, 10, 8)
    assert json.dumps(t, sort_keys=True) == before


def test_the_preset_does_not_launch_a_web_server_it_cannot_reap():
    """`tensorboard: true` makes OneTrainer Popen a tensorboard server on port
    6006 for every run (BaseTrainer._start_tensorboard). Nothing reaps it: the
    stop call lives on the clean shutdown path, so a trainer killed by Windows --
    exactly what an access violation does -- leaves it running forever, and even
    the clean path only kills the launcher shim, orphaning the server it spawned.
    The run_A crash on 2026-08-17 left two of them behind, still holding
    port 6006 against the next run's.

    Krea Forge never opens it: the app has no reference to tensorboard anywhere,
    and shows loss in the progress label itself. Nothing is lost by leaving the
    server off -- GenericTrainer builds its SummaryWriter unconditionally, so the
    event files (scalars and sample images) are still written to
    workspace/tensorboard and can be read any time with a tensorboard of one's
    own. A re-export from the OneTrainer UI will bring the default back; that is
    what this guard is for."""
    t = preset.load_template()
    assert t["tensorboard"] is False
    # Left on, this starts the same server by another door.
    assert t["tensorboard_always_on"] is False


def test_interruption_cadence_is_the_measured_one():
    """Sampling and backups are the only recurring work that stops the stepping.

    The familydrawn run (2026-08-22) took 2:19:25 for 66 epochs whose 17 steps
    each cost ~3.6s -- ~61s of stepping against a 93.4s/epoch average, so roughly
    a quarter of the wall clock went to something other than training. Previews
    every 10 minutes (14 passes, ~44s to warm up each) and a full backup every 30
    minutes (5 of them) were the two measurable sources. Widened to 30/60: still
    frequent enough to watch a run and to survive a crash, without spending a
    quarter of the run on it. Since 2026-10-07 previews are counted in epochs --
    ten passes per run, patched per run (preset.sample_every_epochs) -- so the
    template only fixes the unit. A re-exported preset that silently restores
    OneTrainer's defaults must fail here."""
    t = preset.load_template()
    assert t["sample_after_unit"] == "EPOCH"
    assert t["backup_after"] == 60
    assert t["backup_after_unit"] == "MINUTE"


def test_the_latent_cache_stays_inside_the_run_folder():
    """Why the cache is rebuilt every run, and why that is not a knob.

    `clear_cache_before_training` looks like the cause but is not: preset.build
    points cache_dir at <run>/cache, a fresh empty directory per run, so the flag
    clears nothing and turning it off would save nothing. Sharing one cache
    across runs is the only thing that would help, and it would serve stale
    latents the moment the dataset or its captions changed."""
    from pathlib import Path
    assert "cache_dir" in preset.PATCHED_FIELDS
    cfg = preset.build_run_config(preset.load_template(), run_dir="R",
                                  lora_name="x", epochs=1, warmup_steps=10,
                                  prodigy_steps=8)
    assert Path(cfg["cache_dir"]) == Path("R") / "cache"


def test_the_run_keeps_snapshots_in_the_shipped_format_every_five_epochs():
    """run_H (2026-09-04) overshot in its last 15 epochs and the only files left
    were the overshot end and two backups at arbitrary minutes. A snapshot every
    five epochs, written in the same format as the final file, is what the session
    falls back to when verify says the end of the run went bad (~114 MB, ~2 s each,
    deleted once the run has settled which file ships)."""
    t = preset.load_template()
    assert t["save_every"] == 5
    assert t["save_every_unit"] == "EPOCH"
    assert t["save_skip_first"] == 0
    assert t["output_model_format"] == "COMFY_LORA"     # snapshots share this format


def test_prodigy_step_size_is_frozen_per_run_not_in_the_template():
    """run_H's Prodigy step size settled at 5.8e-04 against 2.9e-04..3.4e-04 on
    every healthy run. `prodigy_steps` freezes the estimate after N steps; it is
    sized from the run (steps.plan) and patched in, so the template carries no
    fixed number. UNPROVEN as a recipe until a full run has shipped with it."""
    t = preset.load_template()
    assert t["optimizer"]["prodigy_steps"] is None
    cfg = preset.build_run_config(t, r"C:\out\MyLora", "MyLora", 12, 50, prodigy_steps=900)
    assert cfg["optimizer"]["prodigy_steps"] == 900
    assert t["optimizer"]["prodigy_steps"] is None      # template untouched


def test_the_prodigy_cap_sits_just_above_the_healthy_band():
    """Healthy runs settled at 2.9e-04..3.4e-04; the two that broke at 5.8e-04
    and 7.3e-04."""
    assert 3.4e-4 < preset.PRODIGY_D_CAP < 5.8e-4


def test_snapshots_are_saved_on_the_same_tenth_of_the_run_as_the_previews():
    """Every preview has a file behind it: the epoch whose preview looked right
    is one that can ship (2026-10-08). Overrides the template's fixed five."""
    t = preset.load_template()
    for epochs in (80, 67, 33, 12, 10, 1):
        cfg = preset.build_run_config(t, "R", "X", epochs, 10, 8)
        assert cfg["save_every"] == cfg["sample_after"] == preset.sample_every_epochs(epochs)
        assert cfg["save_every_unit"] == cfg["sample_after_unit"] == "EPOCH"
