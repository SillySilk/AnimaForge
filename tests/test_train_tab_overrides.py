import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import toml
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])


def _tab(tmp_path, model="anima"):
    from core.engines.onetrainer import engine
    from ui.train_tab import TrainTab
    t = TrainTab()
    ds = tmp_path / "ds"
    ds.mkdir()
    for i in range(4):
        (ds / f"a{i}.png").write_bytes(b"x")
        (ds / f"a{i}.txt").write_text("mrx, a man", encoding="utf-8")
    t._output_dir = str(tmp_path / "out")
    t.set_dataset(str(ds), 4)
    t.set_lora_name("Mr X")
    t.set_trigger_word("mrx")
    t._dit_path, t._qwen3_path, t._vae_path = "d.st", "q.st", "v.st"
    t._validate_for_config = lambda: (True, "")
    t.set_model(model)
    return t


def test_overrides_follow_the_model_and_the_set(tmp_path, monkeypatch):
    from core.engines.onetrainer import engine
    monkeypatch.setattr(engine, "validate", lambda root: (True, ""))
    t = _tab(tmp_path, "krea2")
    counts = []
    t.config_overrides_changed.connect(counts.append)
    t.set_config_overrides({"learning_rate": 1})
    assert t.config_overrides() == {"learning_rate": 1} and counts[-1] == 1
    t.set_model("anima")
    assert t.config_overrides() == {} and counts[-1] == 0

    t.set_config_overrides({"training_arguments.noise_offset": 0.05})
    t._validate_for_model = lambda: (True, "")
    rd, _ = t.build_run_definition()
    assert rd.config_overrides == {"training_arguments.noise_offset": 0.05}

    rd.model, rd.config_overrides = "krea2", {"learning_rate": 1}
    t.apply_run_definition(rd)
    assert t.model() == "krea2" and t.config_overrides() == {"learning_rate": 1}


def test_overrides_are_a_copy(tmp_path):
    t = _tab(tmp_path)
    d = {"a": 1}
    t.set_config_overrides(d)
    d["b"] = 2
    assert t.config_overrides() == {"a": 1}


def test_krea_base_config_lists_every_template_setting(tmp_path, monkeypatch):
    from core.engines.onetrainer import engine
    monkeypatch.setattr(engine, "validate", lambda root: (True, ""))
    t = _tab(tmp_path, "krea2")
    t.set_onetrainer_root("D:/OT")
    flat, msg = t.base_config()
    assert flat is not None, msg
    assert len(flat) >= 150
    assert flat["model_type"] == "KREA_2"
    assert "optimizer.prodigy_steps" in flat and "epochs" in flat
    assert flat["output_model_destination"].endswith("Mr_X.safetensors") or \
        flat["output_model_destination"].endswith(".safetensors")
    assert str(tmp_path / "out") in flat["output_model_destination"]


def test_anima_base_config_is_the_generated_toml(tmp_path):
    t = _tab(tmp_path)
    flat, msg = t.base_config()
    assert flat is not None, msg
    assert "training_arguments.max_train_epochs" in flat or "saving_arguments.save_every_n_epochs" in flat
    assert flat["network_arguments.network_module"] == "networks.lora_anima"
    assert "dataset_toml.general.keep_tokens" in flat
    # paths name the real run folder, never the scratch folder it was built in
    assert str(tmp_path / "out").replace("\\", "/") in flat["dataset_arguments.dataset_config"]
    assert not (tmp_path / "out").exists()


def test_base_config_refuses_when_the_run_is_not_ready(tmp_path):
    t = _tab(tmp_path)
    t._validate_for_config = lambda: (False, "Set the Anima model files first.")
    assert t.base_config() == (None, "Set the Anima model files first.")


def test_generate_config_writes_overrides(tmp_path):
    t = _tab(tmp_path)
    t.set_config_overrides({"training_arguments.noise_offset": 0.05})
    t._generate_config()
    assert toml.load(t._config_path)["training_arguments"]["noise_offset"] == 0.05


def test_applying_a_preset_sets_its_overrides(tmp_path):
    from core import train_presets as tp
    t = _tab(tmp_path, "krea2")
    t.apply_preset(tp.TrainPreset("K", model="krea2", overrides={"learning_rate": 1}))
    assert t.config_overrides() == {"learning_rate": 1}
    t.apply_preset(tp.BUILTINS[0])               # a model-less preset leaves them alone
    assert t.config_overrides() == {"learning_rate": 1}
    t.apply_preset(tp.TrainPreset("A", model="anima", overrides={"x": 1}))
    assert t.config_overrides() == {"learning_rate": 1}   # another model's settings never land
