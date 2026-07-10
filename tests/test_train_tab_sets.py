import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from ui.train_tab import TrainTab
from core.batch import RunDefinition

_app = QApplication.instance() or QApplication([])


def test_apply_run_definition_populates_widgets():
    rd = RunDefinition(
        lora_name="Demo", dataset_folder="C:/ds", image_count=7,
        optimizer="prodigy", learning_rate=0.0001, network_dim=24,
        network_alpha=12, target_steps=1500, sample_enabled=True,
        sample_prompts=["x", "y"], subject_type="style",
    )
    t = TrainTab()
    t.apply_run_definition(rd)
    assert t._lora_name_edit.text() == "Demo"
    assert t._current_optimizer() == "prodigy"   # optimizer is fixed (selector removed)
    assert t._dim_spin.value() == 24
    assert t._alpha_spin.value() == 12
    assert t._target_steps_spin.value() == 1500
    assert t._sample_enable_check.isChecked() is True
    assert t._sample_prompts_edit.toPlainText() == "x\ny"
    assert t._subject_combo.currentIndex() == 2
    assert t._train_text_encoder is False


def test_build_run_definition_includes_sample_fields(tmp_path: Path):
    sd = tmp_path / "sd"; sd.mkdir()
    dit = tmp_path / "dit.safetensors"; dit.write_bytes(b"x")
    q = tmp_path / "q.safetensors"; q.write_bytes(b"x")
    vae = tmp_path / "vae.safetensors"; vae.write_bytes(b"x")
    out = tmp_path / "out"; out.mkdir()
    ds = tmp_path / "ds"; ds.mkdir()

    t = TrainTab()
    t.set_environment(str(sd), str(dit), str(q), str(vae), str(out))
    t.set_dataset(str(ds), 12)
    t._lora_name_edit.setText("Demo")
    t._sample_enable_check.setChecked(True)
    t._sample_prompts_edit.setPlainText("a portrait\nb landscape")

    rd, msg = t.build_run_definition()
    assert rd is not None, msg
    assert rd.sample_enabled is True
    assert rd.sample_prompts == ["a portrait", "b landscape"]


def _make_state_folder(out: Path, name: str, folder: str):
    """Create a minimal sd-scripts-style saved-state folder."""
    sf = out / folder
    sf.mkdir(parents=True)
    (sf / "train_state.json").write_text('{"current_epoch": 2, "current_step": 398}')
    (sf / "optimizer.bin").write_bytes(b"x")
    return sf


def test_stop_arms_resume_and_config_includes_it(tmp_path: Path):
    """Regression: after an in-session Stop, the next Start must resume from the
    saved state instead of restarting from zero (auto-resume, opt-out)."""
    sd = tmp_path / "sd"; sd.mkdir()
    dit = tmp_path / "dit.safetensors"; dit.write_bytes(b"x")
    q = tmp_path / "q.safetensors"; q.write_bytes(b"x")
    vae = tmp_path / "vae.safetensors"; vae.write_bytes(b"x")
    out = tmp_path / "out"; out.mkdir()
    ds = tmp_path / "ds"; (ds).mkdir()
    (ds / "a.png").write_bytes(b"x")

    t = TrainTab()
    t.set_environment(str(sd), str(dit), str(q), str(vae), str(out))
    t.set_dataset(str(ds), 12)
    t._lora_name_edit.setText("Demo")

    # No state yet → resume option hidden/unchecked.
    t._refresh_resume_option()
    assert t._resume_check.isChecked() is False

    # Simulate a run that saved a state, then a user Stop.
    _make_state_folder(out, "Demo", "Demo-000002-state")
    t._on_training_finished(False)  # the stop/failure path

    # Stop must arm resume.
    assert t._resume_state_path is not None
    assert t._resume_check.isChecked() is True
    assert "Demo-000002-state" in t._resume_state_path

    # And the generated config must actually carry the resume path.
    import toml
    t._generate_config()
    cfg = toml.load(t._config_path)
    assert cfg["training_arguments"]["resume"].endswith("Demo-000002-state")

    # Opting out (uncheck) must produce a fresh config with no resume.
    t._resume_check.setChecked(False)
    t._generate_config()
    cfg2 = toml.load(t._config_path)
    assert "resume" not in cfg2["training_arguments"]


def test_set_row_has_blank_name_field_and_selector():
    t = TrainTab()
    assert hasattr(t, "_set_name_edit")
    assert t._set_name_edit.text() == ""          # blank by default, no stale name
    assert hasattr(t, "_sets_combo")


def test_optimizer_preset_round_trips_adamw8bit():
    # The preset selector is back (user feedback): AdamW8bit must survive a
    # RunDefinition round-trip and reveal the LR row; Prodigy hides it.
    rd = RunDefinition(
        lora_name="Cmp", dataset_folder="C:/ds", image_count=3,
        optimizer="adamw8bit", learning_rate=0.0002, network_dim=16,
        network_alpha=8, target_steps=1000,
    )
    t = TrainTab()
    t.apply_run_definition(rd)
    assert t._current_optimizer() == "adamw8bit"
    assert t.optimizer_label() == "AdamW8bit"
    assert not t._lr_row_widget.isHidden()
    assert abs(t._lr_spin.value() - 0.0002) < 1e-9
    t._set_optimizer("prodigy")
    assert t._current_optimizer() == "prodigy"
    assert t._lr_row_widget.isHidden()


def test_build_run_definition_fills_empty_prompts_from_captions(tmp_path: Path):
    # load -> caption -> train: the box is empty at launch (load-time autofill ran
    # before captions existed), so the snapshot must self-fill from the dataset.
    sd = tmp_path / "sd"; sd.mkdir()
    dit = tmp_path / "dit.safetensors"; dit.write_bytes(b"x")
    q = tmp_path / "q.safetensors"; q.write_bytes(b"x")
    vae = tmp_path / "vae.safetensors"; vae.write_bytes(b"x")
    out = tmp_path / "out"; out.mkdir()
    ds = tmp_path / "ds"; ds.mkdir()
    for i, cap in enumerate(["a red fox in snow", "a knight at dusk",
                             "a cat on a mat", "a girl in a forest",
                             "a ship at sea", "a tower under stars"]):
        (ds / f"img{i}.png").write_bytes(b"x")
        (ds / f"img{i}.txt").write_text(cap, encoding="utf-8")

    t = TrainTab()
    t.set_environment(str(sd), str(dit), str(q), str(vae), str(out))
    t.set_dataset(str(ds), 6)
    t._lora_name_edit.setText("Demo")
    t._sample_enable_check.setChecked(True)
    t._sample_prompts_edit.clear()  # simulate the cleared-at-load box

    rd, msg = t.build_run_definition()
    assert rd is not None, msg
    # preview count (hardwired SAMPLE_COUNT) random verbatim captions from the dataset
    assert len(rd.sample_prompts) == 3
    assert all(p in {"a red fox in snow", "a knight at dusk", "a cat on a mat",
                     "a girl in a forest", "a ship at sea", "a tower under stars"}
               for p in rd.sample_prompts)


def test_build_run_definition_never_clobbers_authored_prompts(tmp_path: Path):
    sd = tmp_path / "sd"; sd.mkdir()
    dit = tmp_path / "dit.safetensors"; dit.write_bytes(b"x")
    q = tmp_path / "q.safetensors"; q.write_bytes(b"x")
    vae = tmp_path / "vae.safetensors"; vae.write_bytes(b"x")
    out = tmp_path / "out"; out.mkdir()
    ds = tmp_path / "ds"; ds.mkdir()
    (ds / "a.png").write_bytes(b"x")
    (ds / "a.txt").write_text("dataset caption", encoding="utf-8")

    t = TrainTab()
    t.set_environment(str(sd), str(dit), str(q), str(vae), str(out))
    t.set_dataset(str(ds), 1)
    t._lora_name_edit.setText("Demo")
    t._sample_prompts_edit.setPlainText("my own prompt")

    rd, msg = t.build_run_definition()
    assert rd is not None, msg
    assert rd.sample_prompts == ["my own prompt"]


def test_apply_run_definition_restores_the_quality_prefix():
    """Load Set must not leave the previous dataset's prefix on the run."""
    t = TrainTab()
    t.set_quality_prefix("OLD_LIVE_PREFIX")
    rd = RunDefinition(lora_name="Saved", dataset_folder="C:/ds", image_count=5,
                       trigger_word="savedtrigger", quality_prefix="SAVED_SET_PREFIX")
    t.apply_run_definition(rd)
    assert t._quality_prefix == "SAVED_SET_PREFIX"


def test_apply_run_definition_clears_a_stale_prefix_when_the_set_has_none():
    t = TrainTab()
    t.set_quality_prefix("LEFTOVER")
    rd = RunDefinition(lora_name="S", dataset_folder="C:/ds", image_count=5)
    t.apply_run_definition(rd)
    assert t._quality_prefix == ""


def test_load_set_requested_carries_the_quality_prefix():
    t = TrainTab()
    seen = []
    t.load_set_requested.connect(lambda f, tr, p: seen.append((f, tr, p)))
    rd = RunDefinition(lora_name="S", dataset_folder="C:/ds", image_count=5,
                       trigger_word="trig", quality_prefix="masterpiece")
    t.apply_run_definition(rd)
    assert seen == [("C:/ds", "trig", "masterpiece")]


def test_quality_prefix_survives_an_apply_then_build_round_trip(tmp_path: Path):
    """The snapshot must survive a load -> save cycle, not just a load."""
    sd = tmp_path / "sd"; sd.mkdir()
    dit = tmp_path / "dit.safetensors"; dit.write_bytes(b"x")
    q = tmp_path / "q.safetensors"; q.write_bytes(b"x")
    vae = tmp_path / "vae.safetensors"; vae.write_bytes(b"x")
    out = tmp_path / "out"; out.mkdir()
    ds = tmp_path / "ds"; ds.mkdir()

    t = TrainTab()
    t.set_environment(str(sd), str(dit), str(q), str(vae), str(out))
    rd = RunDefinition(lora_name="Demo", dataset_folder=str(ds), image_count=5,
                       trigger_word="trig", quality_prefix="masterpiece, best quality")
    t.apply_run_definition(rd)
    rd2, msg = t.build_run_definition()
    assert rd2 is not None, msg
    assert rd2.quality_prefix == rd.quality_prefix


def test_set_quality_prefix_reaches_build_run_definition(tmp_path: Path):
    """Home owns the quality-prefix control; TrainTab has no widget of its own for it,
    so set_quality_prefix (wired from Home via MainWindow) must be the only path a
    queued run's snapshot gets it through."""
    sd = tmp_path / "sd"; sd.mkdir()
    dit = tmp_path / "dit.safetensors"; dit.write_bytes(b"x")
    q = tmp_path / "q.safetensors"; q.write_bytes(b"x")
    vae = tmp_path / "vae.safetensors"; vae.write_bytes(b"x")
    out = tmp_path / "out"; out.mkdir()
    ds = tmp_path / "ds"; ds.mkdir()

    t = TrainTab()
    assert t._quality_prefix == ""  # default before Home ever sets it
    t.set_environment(str(sd), str(dit), str(q), str(vae), str(out))
    t.set_dataset(str(ds), 1)
    t._lora_name_edit.setText("Demo")
    t.set_quality_prefix("masterpiece, best quality")

    rd, msg = t.build_run_definition()
    assert rd is not None, msg
    assert rd.quality_prefix == "masterpiece, best quality"
