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
    assert rd.subject_type == ""
    rd2 = _rd(sample_enabled=True, sample_prompts=["a", "b"],
              subject_type="character")
    back = RunDefinition.from_dict(rd2.to_dict())
    assert back.sample_enabled is True
    assert back.sample_prompts == ["a", "b"]
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


def test_run_definition_drops_legacy_sample_keys():
    """Old sets/*.json carry sample_every: 2 and sample_count: 4. Both are dropped."""
    rd = RunDefinition.from_dict({
        "lora_name": "y", "dataset_folder": "d", "image_count": 1,
        "sample_every": 2, "sample_count": 4,
    })
    assert not hasattr(rd, "sample_every")
    assert not hasattr(rd, "sample_count")
    assert "sample_every" not in rd.to_dict()


def test_resolve_sample_prompts_prefers_snapshot():
    from core.batch import resolve_sample_prompts
    rd = _rd(sample_prompts=["one", "  ", "two"])
    assert resolve_sample_prompts(rd) == ["one", "two"]


def test_resolve_sample_prompts_grabs_from_dataset_when_empty(tmp_path):
    # Queued before captioning: draw real captions from the dataset at execution time.
    from core.batch import resolve_sample_prompts
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "a.txt").write_text("a cat on a mat", encoding="utf-8")
    rd = _rd(dataset_folder=str(tmp_path), sample_prompts=[])
    assert resolve_sample_prompts(rd) == ["a cat on a mat"]


def test_resolve_sample_prompts_captionless_dataset_returns_empty(tmp_path):
    from core.batch import resolve_sample_prompts
    rd = _rd(dataset_folder=str(tmp_path), sample_prompts=[])
    assert resolve_sample_prompts(rd) == []


def test_caption_fields_default_and_roundtrip():
    rd = _rd(quality_prefix="masterpiece", caption_order="tags_first",
             refine_enabled=True, lms_url="http://x/v1", max_tokens=900,
             tagger_threshold=0.4, caption_policy="keep")
    back = RunDefinition.from_dict(rd.to_dict())
    assert back == rd
    assert _rd().caption_policy == "ask"
    assert _rd().refine_enabled is False


def test_to_caption_job_carries_the_snapshot_not_live_ui():
    rd = _rd(trigger_word="manbag", quality_prefix="masterpiece",
             caption_order="tags_first", refine_enabled=True)
    job = rd.to_caption_job(sdscripts_path="C:/sd", characters_file="C:/c.json",
                            policy="keep")
    assert job.trigger == "manbag"
    assert job.prefix == "masterpiece"
    assert job.order == "tags_first"
    assert job.policy == "keep"
    assert job.chain == ["tag", "describe", "refine", "combine"]


def test_to_caption_job_drops_refine_when_disabled():
    job = _rd(refine_enabled=False).to_caption_job("C:/sd", "", "overwrite")
    assert job.chain == ["tag", "describe", "combine"]


def test_to_caption_job_sets_tagger_use_onnx_and_combine_prefix():
    rd = _rd(trigger_word="manbag", quality_prefix="masterpiece",
             tagger_use_onnx=False, tagger_model_id="some/repo")
    job = rd.to_caption_job(sdscripts_path="C:/sd", characters_file="", policy="keep")
    assert job.tagger_use_onnx is False
    assert job.tagger_model_id == "some/repo"
    assert job.combine_prefix() == "manbag, masterpiece"


def test_from_dict_on_json_lacking_new_keys_yields_documented_defaults():
    """Old sets/*.json (written before Task 10) carry none of the new caption fields —
    they must still load, falling back to the dataclass defaults."""
    rd = RunDefinition.from_dict({
        "lora_name": "old", "dataset_folder": "C:/d", "image_count": 3,
    })
    assert rd.quality_prefix == ""
    assert rd.caption_order == "nl_first"
    assert rd.refine_enabled is False
    assert rd.lms_url == ""
    assert rd.lms_model == ""
    assert rd.lms_focus == ""
    assert rd.lora_type == ""
    assert rd.max_tokens == 1200
    assert rd.tagger_model_id == ""
    assert rd.tagger_threshold == 0.35
    assert rd.tagger_use_onnx is True
    assert rd.style_anchor == ""
    assert rd.caption_policy == "ask"
