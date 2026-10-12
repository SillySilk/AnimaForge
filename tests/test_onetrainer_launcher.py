"""The launcher that caps Prodigy's step size before handing over to train.py."""
from core.engines.onetrainer import onetrainer_train


class _Opt:
    def __init__(self, d, d_max):
        self.param_groups = [{"d": d, "d_max": d_max}]


def test_a_step_size_over_the_cap_is_clamped_to_it():
    opt = _Opt(7.3e-4, 9.0e-4)
    onetrainer_train._clamp(opt, 4.0e-4)
    assert opt.param_groups[0]["d"] == 4.0e-4
    assert opt.param_groups[0]["d_max"] == 4.0e-4


def test_a_step_size_under_the_cap_is_untouched():
    opt = _Opt(3.1e-4, 3.3e-4)
    onetrainer_train._clamp(opt, 4.0e-4)
    assert opt.param_groups[0] == {"d": 3.1e-4, "d_max": 3.3e-4}


def test_the_cap_is_only_for_prodigy_runs(tmp_path):
    import json
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"optimizer": {"optimizer": "ADAMW_8BIT"}}), encoding="utf-8")
    assert onetrainer_train._optimizer_name(["--config-path", str(cfg)]) == "ADAMW_8BIT"
    cfg.write_text(json.dumps({"optimizer": {"optimizer": "PRODIGY"}}), encoding="utf-8")
    assert onetrainer_train._optimizer_name(["--config-path", str(cfg)]) == "PRODIGY"
    assert onetrainer_train._optimizer_name(["--config-path", str(tmp_path / "gone.json")]) == ""
    assert onetrainer_train._optimizer_name([]) == ""
