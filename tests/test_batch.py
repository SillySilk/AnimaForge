import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.batch import RunDefinition, save_queue, load_queue, QUEUED


def _rd(**kw):
    base = dict(lora_name="lr", dataset_folder="C:/d", image_count=20)
    base.update(kw)
    return RunDefinition(**base)


def test_run_definition_roundtrip():
    rd = _rd(trigger_word="mychar", network_dim=32, target_steps=900, output_dir="C:/out")
    d = rd.to_dict()
    rd2 = RunDefinition.from_dict(d)
    assert rd2 == rd
    assert rd2.status == QUEUED


def test_rundefinition_new_fields_default_and_roundtrip():
    rd = _rd()
    assert rd.sample_enabled is False
    assert rd.sample_prompts == []
    assert rd.sample_every == 1
    assert rd.subject_type == ""
    rd2 = _rd(sample_enabled=True, sample_prompts=["a", "b"],
              sample_every=3, subject_type="character")
    back = RunDefinition.from_dict(rd2.to_dict())
    assert back.sample_enabled is True
    assert back.sample_prompts == ["a", "b"]
    assert back.sample_every == 3
    assert back.subject_type == "character"


def test_from_dict_ignores_unknown_keys():
    rd = RunDefinition.from_dict({"lora_name": "x", "dataset_folder": "C:/d",
                                  "image_count": 5, "bogus_key": 123})
    assert rd.lora_name == "x" and rd.image_count == 5


def test_save_load_queue(tmp_path):
    runs = [_rd(lora_name="a"), _rd(lora_name="b", trigger_word="t")]
    p = str(tmp_path / "q.json")
    save_queue(p, runs)
    loaded = load_queue(p)
    assert [r.lora_name for r in loaded] == ["a", "b"]
    assert loaded[1].trigger_word == "t"


def test_load_missing_returns_empty(tmp_path):
    assert load_queue(str(tmp_path / "nope.json")) == []


def test_run_definition_sample_count_roundtrip():
    from core.batch import RunDefinition
    rd = RunDefinition(lora_name="x", dataset_folder="d", image_count=5, sample_count=7)
    rd2 = RunDefinition.from_dict(rd.to_dict())
    assert rd2.sample_count == 7
    # old sets without the key default to 4
    assert RunDefinition.from_dict({"lora_name": "y", "dataset_folder": "d", "image_count": 1}).sample_count == 4
