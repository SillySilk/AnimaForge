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
