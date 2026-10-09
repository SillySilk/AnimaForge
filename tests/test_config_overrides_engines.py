import toml

from core.config_generator import generate_configs


def _gen(tmp_path, name, **kw):
    ds = tmp_path / "ds"
    ds.mkdir(exist_ok=True)
    (ds / "a.png").write_bytes(b"x")
    return generate_configs(output_dir=str(tmp_path / name), lora_name="L", dit_path="d.st",
                            qwen3_path="q.st", vae_path="v.st", dataset_folder=str(ds),
                            epochs=10, repeats=1, **kw)


def _text(path, root):
    return open(path, encoding="utf-8").read().replace(str(root).replace("\\", "/"), "X")


def test_no_overrides_writes_identical_toml(tmp_path):
    a_main, a_ds = _gen(tmp_path, "a")
    b_main, b_ds = _gen(tmp_path, "b", overrides={})
    assert _text(a_main, tmp_path / "a") == _text(b_main, tmp_path / "b")
    assert _text(a_ds, tmp_path / "a") == _text(b_ds, tmp_path / "b")


def test_kohya_overrides_land_in_the_toml(tmp_path):
    main, _ = _gen(tmp_path, "c", overrides={"training_arguments.max_train_epochs": 3,
                                             "training_arguments.noise_offset": 0.05})
    cfg = toml.load(main)
    assert cfg["training_arguments"]["max_train_epochs"] == 3
    assert cfg["training_arguments"]["noise_offset"] == 0.05


def test_dataset_toml_settings_are_overridable_too(tmp_path):
    main, ds = _gen(tmp_path, "d", overrides={"dataset_toml.general.keep_tokens": 2})
    assert toml.load(ds)["general"]["keep_tokens"] == 2
    assert "dataset_toml" not in toml.load(main)


def test_one_dataset_setting_never_freezes_the_image_folder(tmp_path):
    _, ds = _gen(tmp_path, "e", overrides={"dataset_toml.datasets.0.subsets.0.caption_dropout_rate": 0.1})
    sub = toml.load(ds)["datasets"][0]["subsets"][0]
    assert sub["caption_dropout_rate"] == 0.1
    assert sub["image_dir"].endswith("/ds") and sub["num_repeats"] == 1


def test_run_managed_settings_are_locked():
    from core import config_overrides as co
    from core.config_generator import LOCKED_PATHS
    from core.engines.onetrainer.preset import LOCKED_PATHS as OT_LOCKED
    for p in ("training_arguments.output_dir", "training_arguments.output_name",
              "training_arguments.max_train_epochs", "dataset_arguments.dataset_config",
              "dataset_toml.datasets.0.subsets.0.image_dir",
              "dataset_toml.datasets.0.subsets.0.num_repeats", "model_arguments.vae"):
        assert co.is_locked(p, LOCKED_PATHS), p
    for p in ("workspace_dir", "output_model_destination", "epochs", "optimizer.prodigy_steps"):
        assert co.is_locked(p, OT_LOCKED), p
    assert not co.is_locked("learning_rate", OT_LOCKED)
