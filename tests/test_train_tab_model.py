import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from core import models

_app = QApplication.instance() or QApplication([])


def _tab(tmp_path):
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
    return t


def test_default_model_is_anima(tmp_path):
    t = _tab(tmp_path)
    assert t.model() == "anima" and t.engine() == models.KOHYA


def test_switching_models_resets_target_steps_to_each_models_default(tmp_path):
    t = _tab(tmp_path)
    t.set_model("krea2")
    assert t.get_target_steps() == 0 and t._target_steps_spin.specialValueText() == "Auto"
    t.set_model("anima")
    assert t.get_target_steps() >= 500 and t._target_steps_spin.minimum() == 500


def test_reselecting_the_same_model_keeps_a_custom_step_target(tmp_path):
    t = _tab(tmp_path)
    t.set_model("krea2")
    t._target_steps_spin.setValue(1200)
    t.set_model("krea2")
    assert t.get_target_steps() == 1200


def test_unknown_model_is_refused_loudly(tmp_path):
    t = _tab(tmp_path)
    with pytest.raises(KeyError, match="flux9"):
        t.set_model("flux9")


def test_krea_validation_asks_for_onetrainer_not_anima_files(tmp_path):
    t = _tab(tmp_path)
    t.set_model("krea2")
    ok, msg = t._validate_for_model()
    assert not ok and "OneTrainer" in msg and "sd-scripts" not in msg


def test_krea_run_definition_carries_model_root_style_and_auto_steps(tmp_path, monkeypatch):
    from core.engines.onetrainer import engine
    monkeypatch.setattr(engine, "validate", lambda root: (True, ""))
    t = _tab(tmp_path)
    t.set_model("krea2")
    t.set_onetrainer_root("D:/OT")
    t.set_caption_style("prose")
    rd, msg = t.build_run_definition()
    assert msg == "" and (rd.model, rd.onetrainer_root, rd.caption_style, rd.target_steps) == (
        "krea2", "D:/OT", "prose", 0)


def test_apply_run_definition_restores_model_and_style(tmp_path):
    from core.batch import RunDefinition
    t = _tab(tmp_path)
    seen = []
    t.model_changed.connect(seen.append)
    t.apply_run_definition(RunDefinition("K", str(tmp_path / "ds"), 4, model="krea2",
                                         caption_style="prose", target_steps=0))
    assert t.model() == "krea2" and t.caption_style() == "prose" and seen == ["krea2"]


def test_summary_steps_follow_the_model(tmp_path):
    from core.engines.onetrainer import steps
    t = _tab(tmp_path)
    assert t.summary_steps() == t.get_target_steps()
    t.set_model("krea2")
    assert t.summary_steps() == steps.plan(4, "character", 0)["total_steps"]


def test_uncap_text_follows_the_selected_models_step_cap(tmp_path):
    t = _tab(tmp_path)
    assert "3,000" in t._uncap_check.toolTip() and "3,000" in t._uncap_warn.text()
    t.set_model("moss")
    assert "5,000" in t._uncap_check.toolTip() and "5,000" in t._uncap_warn.text()
    assert "3,000" not in t._uncap_check.toolTip()
    t.set_model("anima")
    assert "3,000" in t._uncap_check.toolTip() and "3,000" in t._uncap_warn.text()
    t.set_model("krea2")   # no step profile: unchanged from the default text
    assert "3,000" in t._uncap_check.toolTip()


def test_lowvram_indicator_hides_for_moss_and_returns_for_anima(tmp_path):
    from core import lowvram
    t = _tab(tmp_path)
    lowvram.set_current({"micro_batch": 1, "grad_accum": 4, "blocks_to_swap": 8})
    try:
        t._refresh_lowvram_indicator()
        assert t._lowvram_indicator.isHidden() is False
        t.set_model("moss")
        assert t._lowvram_indicator.isHidden()
        t.set_model("anima")
        assert t._lowvram_indicator.isHidden() is False
    finally:
        lowvram.clear()
