import json
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from core.engines.onetrainer import concepts
from core.settings import AppSettings


def _settings(tmp_path):
    a = AppSettings(QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    a.set("sample_enable", True)
    a.set("sample_quality_prefix", "")
    return a


def test_kohya_prompts_get_fixed_seeds(tmp_path):
    a = _settings(tmp_path)
    args = a.prepare_sample_args(str(tmp_path), "L", "", prompts=["a man", "a dog --d 7"])
    lines = Path(args["sample_prompts"]).read_text(encoding="utf-8").splitlines()
    assert lines[0].endswith("--d 1000")
    assert lines[1].endswith("--d 7") and lines[1].count("--d") == 1


def test_samples_from_prompts_add_the_trigger_and_fix_seeds():
    out = concepts.samples_from_prompts(["a man", "mrx, b"], "mrx")
    assert [s["prompt"] for s in out] == ["mrx, a man", "mrx, b"]
    assert [s["seed"] for s in out] == [concepts._FIRST_SEED, concepts._FIRST_SEED + 1]
    assert all(s["random_seed"] is False and s["width"] == 512 for s in out)


def test_krea_runs_preview_the_given_prompts(tmp_path, monkeypatch):
    from test_onetrainer_session import FakeProc, _params, _quiet_preflight
    from core.engines.onetrainer.session import TrainingSession
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    assert s.start(_params(tmp_path, trigger_word="mrx", sample_prompts=["a man", "mrx, b"]))
    samples = json.loads((Path(s.run_dir()) / "samples.json").read_text(encoding="utf-8"))
    assert [x["prompt"] for x in samples] == ["mrx, a man", "mrx, b"]


def test_adapter_carries_sample_prompts():
    from core.batch import RunDefinition
    from core.engines.onetrainer.adapter import to_onetrainer_params
    rd = RunDefinition(lora_name="a", dataset_folder="d", image_count=1, model="krea2",
                       sample_prompts=["x"])
    assert to_onetrainer_params(rd).sample_prompts == ["x"]
