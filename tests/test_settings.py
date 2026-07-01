import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QSettings
from core.settings import AppSettings


def _appsettings(tmp_path):
    s = QSettings(str(tmp_path / "t.ini"), QSettings.IniFormat)
    return AppSettings(s)


def test_min_free_vram_default_is_int(tmp_path):
    a = _appsettings(tmp_path)
    val = a.get("min_free_vram_mb")
    assert isinstance(val, int) and val == 12000


def test_roundtrip_typed(tmp_path):
    a = _appsettings(tmp_path)
    a.set("default_network_dim", 32)
    a.set("save_every_n_steps", 100)
    a.set("logit_mean", 0.5)
    a.set("lmstudio_model", "my-model")
    assert a.get("default_network_dim") == 32 and isinstance(a.get("default_network_dim"), int)
    assert a.get("save_every_n_steps") == 100 and isinstance(a.get("save_every_n_steps"), int)
    assert a.get("logit_mean") == 0.5
    assert a.get("lmstudio_model") == "my-model"


def test_sample_prompts_session_only_not_persisted(tmp_path):
    # The sample-prompts box must not carry text across sessions: set is in-memory
    # only, so a fresh AppSettings (a new app launch) starts clean.
    path = str(tmp_path / "t.ini")
    a = AppSettings(QSettings(path, QSettings.IniFormat))
    a.set("sample_prompts", "block one\nblock two")
    assert a.get("sample_prompts") == "block one\nblock two"  # in-session roundtrip
    fresh = AppSettings(QSettings(path, QSettings.IniFormat))
    assert fresh.get("sample_prompts") == ""                  # not carried over


def test_normal_key_still_persists_across_instances(tmp_path):
    # Only sample_prompts is ephemeral; ordinary keys must still survive a restart.
    path = str(tmp_path / "t.ini")
    a = AppSettings(QSettings(path, QSettings.IniFormat))
    a.set("default_network_dim", 64)
    fresh = AppSettings(QSettings(path, QSettings.IniFormat))
    assert fresh.get("default_network_dim") == 64


def test_migration_copies_legacy_once(tmp_path):
    import core.settings as S
    old = QSettings(str(tmp_path / "old.ini"), QSettings.IniFormat)
    new = QSettings(str(tmp_path / "new.ini"), QSettings.IniFormat)
    old.setValue("dit_path", "X")
    old.sync()
    S._migrate_between(old, new)
    assert new.value("dit_path") == "X"
    # Idempotent + non-destructive: a second run must not re-copy or clobber.
    new.setValue("dit_path", "Y")
    S._migrate_between(old, new)
    assert new.value("dit_path") == "Y"


def test_migration_skips_populated_new_store(tmp_path):
    import core.settings as S
    old = QSettings(str(tmp_path / "old.ini"), QSettings.IniFormat)
    new = QSettings(str(tmp_path / "new.ini"), QSettings.IniFormat)
    old.setValue("dit_path", "OLD")
    new.setValue("vae_path", "ALREADY")
    old.sync(); new.sync()
    S._migrate_between(old, new)
    assert new.value("dit_path") is None      # not copied — new store already had data
    assert new.value("vae_path") == "ALREADY"


def test_defaults(tmp_path):
    a = _appsettings(tmp_path)
    assert a.get("default_optimizer") == "prodigy"
    assert a.get("forge_api_url") == "http://127.0.0.1:7860"
    assert a.get("sample_enable") is True          # previews on by default
    assert a.get("save_every_n_steps") == 250


def test_fp8_scaled_is_not_a_setting(tmp_path):
    # fp8_scaled is a no-op on Anima and must not be exposed as a knob.
    try:
        _appsettings(tmp_path).get("fp8_scaled")
        raise AssertionError("fp8_scaled should not be a known setting")
    except KeyError:
        pass


def test_extra_args_empty_by_default(tmp_path):
    assert _appsettings(tmp_path).build_extra_training_args() == {}


def test_extra_args_includes_enabled_only(tmp_path):
    a = _appsettings(tmp_path)
    a.set("flip_aug", True)
    a.set("caption_dropout_rate", 0.1)
    a.set("network_dropout", 0.2)
    a.set("weighting_scheme", "logit_normal")
    e = a.build_extra_training_args()
    assert e == {"flip_aug": True, "caption_dropout_rate": 0.1,
                 "network_dropout": 0.2, "weighting_scheme": "logit_normal"}


def test_extra_args_omits_zero_dropout(tmp_path):
    a = _appsettings(tmp_path)
    a.set("caption_dropout_rate", 0.0)
    assert "caption_dropout_rate" not in a.build_extra_training_args()


def test_sample_args_disabled_returns_empty(tmp_path):
    a = _appsettings(tmp_path)
    a.set("sample_enable", False)
    assert a.prepare_sample_args(str(tmp_path), "lr") == {}


def test_sample_args_enabled_no_prompts_falls_back_to_trigger(tmp_path):
    # Enabled with no authored prompts: previews still render using the trigger alone.
    a = _appsettings(tmp_path)   # sample_enable defaults True
    a.set("sample_quality_prefix", "")  # isolate trigger behavior from the quality prefix
    args = a.prepare_sample_args(str(tmp_path), "lr", trigger_word="mychar")
    f = tmp_path / "configs" / "lr_sample.txt"
    assert f.is_file()
    assert f.read_text(encoding="utf-8").strip() == "mychar"
    assert args["sample_prompts"].endswith("lr_sample.txt")


def test_sample_args_writes_file_with_trigger(tmp_path):
    a = _appsettings(tmp_path)
    a.set("sample_enable", True)
    a.set("sample_quality_prefix", "")  # isolate trigger behavior from the quality prefix
    a.set("sample_prompts", "a girl on a bench\nmychar, a closeup")
    args = a.prepare_sample_args(str(tmp_path), "lr", trigger_word="mychar")
    f = tmp_path / "configs" / "lr_sample.txt"
    assert f.is_file()
    lines = f.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "mychar, a girl on a bench"   # trigger prepended
    assert lines[1] == "mychar, a closeup"            # already had trigger, not doubled
    assert args["sample_prompts"].endswith("lr_sample.txt")
    assert args["sample_every_n_epochs"] == 1


def test_prepare_sample_args_prepends_trigger_and_style_anchor(tmp_path):
    a = _appsettings(tmp_path)
    a.set("sample_enable", True)
    a.set("sample_quality_prefix", "")  # isolate trigger/anchor behavior from the quality prefix
    a.set("sample_prompts", "standing in a field\nmychar, sitting on a bench")
    out = a.prepare_sample_args(str(tmp_path), "demo", trigger_word="mychar", style_anchor="@mystyle")
    written = (tmp_path / "configs" / "demo_sample.txt").read_text(encoding="utf-8").splitlines()
    # both activation words prepended to the first prompt
    assert written[0] == "mychar, @mystyle, standing in a field"
    # trigger already present -> not duplicated; anchor still added
    assert written[1] == "@mystyle, mychar, sitting on a bench"
    assert out["sample_prompts"].endswith("demo_sample.txt")


def test_prepare_sample_args_count_default(tmp_path):
    a = _appsettings(tmp_path)
    assert a.get("sample_count") == 4


def test_quality_prefix_leads_then_trigger(tmp_path):
    # Default quality prefix is on: it leads, the trigger follows, then the prompt body.
    a = _appsettings(tmp_path)
    a.set("sample_enable", True)
    a.set("sample_prompts", "sitting on a bench")
    a.prepare_sample_args(str(tmp_path), "demo", trigger_word="mychar")
    line = (tmp_path / "configs" / "demo_sample.txt").read_text(encoding="utf-8").splitlines()[0]
    assert line == "masterpiece, best quality, score_7, safe, mychar, sitting on a bench"


def test_quality_prefix_not_duplicated(tmp_path):
    # A quality token already in the prompt is not added again.
    a = _appsettings(tmp_path)
    a.set("sample_enable", True)
    a.set("sample_prompts", "best quality, a closeup")
    a.prepare_sample_args(str(tmp_path), "demo", trigger_word="mychar")
    line = (tmp_path / "configs" / "demo_sample.txt").read_text(encoding="utf-8").splitlines()[0]
    assert line.count("best quality") == 1
    assert line == "masterpiece, score_7, safe, mychar, best quality, a closeup"


def test_quality_prefix_cleared_gives_raw(tmp_path):
    a = _appsettings(tmp_path)
    a.set("sample_enable", True)
    a.set("sample_quality_prefix", "")
    a.set("sample_prompts", "a closeup")
    a.prepare_sample_args(str(tmp_path), "demo", trigger_word="mychar")
    line = (tmp_path / "configs" / "demo_sample.txt").read_text(encoding="utf-8").splitlines()[0]
    assert line == "mychar, a closeup"
