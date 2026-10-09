import pytest

from core import compare, sets
from core.batch import RunDefinition


def _folder(tmp_path, tags=True, nl=True):
    ds = tmp_path / "ds"
    ds.mkdir(exist_ok=True)
    for i in range(3):
        (ds / f"a{i}.png").write_bytes(b"x")
        if nl:
            (ds / f"a{i}.nl").write_text("A man stands.", encoding="utf-8")
        if tags:
            (ds / f"a{i}.tags").write_text("1boy, solo", encoding="utf-8")
    return ds


def _rd(ds, **kw):
    base = dict(lora_name="Mr X", dataset_folder=str(ds), image_count=3, model="krea2",
                caption_style="prose+tags", trigger_word="mrx",
                config_overrides={"learning_rate": 1})
    base.update(kw)
    return RunDefinition(**base)


def test_the_two_runs_differ_only_in_caption_style(tmp_path):
    a, b, entry = compare.plan(_rd(_folder(tmp_path)), "tags", ["p1", "p2"])
    da, db = a.to_dict(), b.to_dict()
    differ = {k for k in da if da[k] != db[k]}
    assert differ <= {"lora_name", "caption_style", "caption_policy"}
    assert a.caption_style == "prose+tags" and b.caption_style == "tags"
    assert a.sample_prompts == b.sample_prompts == ["p1", "p2"]
    assert a.compare_group == b.compare_group == entry["id"]
    assert a.sample_enabled and b.sample_enabled
    assert [r["name"] for r in entry["runs"]] == [a.lora_name, b.lora_name]


def test_comparison_runs_get_their_own_folders(tmp_path):
    a, b, _ = compare.plan(_rd(_folder(tmp_path)), "prose", ["p"])
    assert len({a.lora_name, b.lora_name, "Mr X"}) == 3
    assert a.lora_name.startswith("Mr X-prose-tags-") and b.lora_name.startswith("Mr X-prose-")


def test_both_runs_fill_sidecars_whatever_the_folder_holds(tmp_path):
    # Replaced KEEP/OVERWRITE planning (review C1/C2): both runs use FILL.
    from core.caption_policy import FILL
    for tags in (True, False):
        a, b, _ = compare.plan(_rd(_folder(tmp_path, tags=tags)), "tags", ["p"])
        assert a.caption_policy == b.caption_policy == FILL


def test_same_style_is_refused(tmp_path):
    with pytest.raises(ValueError):
        compare.plan(_rd(_folder(tmp_path)), "prose+tags", ["p"])
    with pytest.raises(ValueError):
        compare.plan(_rd(_folder(tmp_path)), "haiku", ["p"])


def test_records_round_trip_and_hide_from_sets(tmp_path):
    _a, _b, e1 = compare.plan(_rd(_folder(tmp_path)), "tags", ["p"])
    e1["id"] = "one"
    compare.record(e1, root=tmp_path)
    e2 = dict(e1, id="two")
    compare.record(e2, root=tmp_path)
    assert [e["id"] for e in compare.load_all(root=tmp_path)] == ["two", "one"]
    assert sets.list_sets(root=tmp_path) == []


def test_compare_group_round_trips():
    rd = RunDefinition(lora_name="a", dataset_folder="d", image_count=1, compare_group="g")
    assert RunDefinition.from_dict(rd.to_dict()).compare_group == "g"


def _entry(tmp_path, model="anima"):
    return {"id": "g", "model": model, "prompts": ["p0", "p1"],
            "runs": [{"name": "Mr X-prose-tags", "style": "prose+tags", "output_dir": ""},
                     {"name": "Mr X-tags", "style": "tags", "output_dir": ""}]}


def test_kohya_previews_pair_by_prompt_from_the_last_epoch(tmp_path):
    from core import paths
    entry = _entry(tmp_path)
    for run in entry["runs"]:
        sample = __import__("pathlib").Path(paths.run_output_dir(str(tmp_path), run["name"])) / "sample"
        sample.mkdir(parents=True)
        for epoch in (2, 10):
            for i in range(2):
                (sample / f"{run['name']}_e{epoch:06d}_{i:02d}_2026.png").write_bytes(b"x")
    rows = compare.final_previews(entry, str(tmp_path), {"Mr X-prose-tags": "done", "Mr X-tags": "done"})
    assert [r["prompt"] for r in rows] == ["p0", "p1"]
    assert rows[1]["a"].endswith("Mr X-prose-tags_e000010_01_2026.png")
    assert rows[0]["b"].endswith("Mr X-tags_e000010_00_2026.png")
    assert rows[0]["status"] == {"a": "done", "b": "done"}


def test_krea_previews_pair_by_prompt(tmp_path):
    from pathlib import Path
    from core import paths
    from core.engines.onetrainer import render
    entry = _entry(tmp_path, model="krea2")
    for run in entry["runs"]:
        dest = Path(render.destination(paths.run_output_dir(str(tmp_path), run["name"])))
        for i in range(2):
            (dest / f"{i} - final").mkdir(parents=True)
            (dest / f"{i} - final" / f"final-{i}.jpg").write_bytes(b"x")
    rows = compare.final_previews(entry, str(tmp_path), {})
    assert rows[1]["a"].endswith("final-1.jpg") and rows[1]["b"].endswith("final-1.jpg")


def test_a_failed_side_is_shown_as_failed(tmp_path):
    entry = _entry(tmp_path)
    rows = compare.final_previews(entry, str(tmp_path), {"Mr X-prose-tags": "failed"})
    assert rows[0]["a"] is None and rows[0]["status"]["a"] == "failed"
    assert rows[0]["status"]["b"] == "pending"
