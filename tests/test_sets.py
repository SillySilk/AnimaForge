import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import sets
from core.batch import RunDefinition


def _rd(name="Demo", out="", folder="ds"):
    return RunDefinition(lora_name=name, dataset_folder=folder, image_count=10,
                         output_dir=out)


def test_save_load_list_delete_roundtrip(tmp_path: Path):
    root = str(tmp_path)
    assert sets.list_sets(root) == []
    sets.save_set("My Set", _rd("My Set"), root)
    assert sets.list_sets(root) == ["My Set"]
    loaded = sets.load_set("My Set", root)
    assert loaded is not None and loaded.lora_name == "My Set"
    sets.delete_set("My Set", root)
    assert sets.list_sets(root) == []


def test_load_missing_or_corrupt_returns_none(tmp_path: Path):
    root = str(tmp_path)
    assert sets.load_set("nope", root) is None
    (sets.sets_dir(root) / "bad.json").write_text("{not json", encoding="utf-8")
    assert sets.load_set("bad", root) is None


def test_marker_excluded_from_list(tmp_path: Path):
    root = str(tmp_path)
    sets.mark_run_active(_rd("Run1"), root)
    assert sets.list_sets(root) == []


def test_interrupted_run_detects_state_without_final(tmp_path: Path):
    root = str(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "Bamford-step00000500-state").mkdir()
    sets.mark_run_active(_rd("Bamford", out=str(out)), root)
    rd = sets.interrupted_run(root)
    assert rd is not None and rd.lora_name == "Bamford"


def test_interrupted_run_none_when_final_exists(tmp_path: Path):
    root = str(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "Bamford-step00000500-state").mkdir()
    (out / "Bamford.safetensors").write_bytes(b"x")
    sets.mark_run_active(_rd("Bamford", out=str(out)), root)
    assert sets.interrupted_run(root) is None


def test_interrupted_run_none_after_clear(tmp_path: Path):
    root = str(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "Bamford-step00000500-state").mkdir()
    sets.mark_run_active(_rd("Bamford", out=str(out)), root)
    sets.clear_active_run(root)
    assert sets.interrupted_run(root) is None


def test_set_to_markdown_has_heading_and_key_fields():
    rd = RunDefinition(lora_name="Eve", dataset_folder="C:/1aa", image_count=136,
                       trigger_word="Eve", target_steps=500, network_dim=16,
                       network_alpha=8, output_dir="C:/out")
    md = sets.set_to_markdown(rd, "FamilyGuy")
    assert md.startswith("# FamilyGuy")
    assert "C:/1aa" in md
    assert "Eve" in md
    assert "500" in md
    assert "16 / 8" in md          # dim / alpha rendered together


def test_set_to_markdown_omits_empty_values():
    rd = RunDefinition(lora_name="X", dataset_folder="", image_count=3)
    md = sets.set_to_markdown(rd, "X")
    assert "Dataset folder" not in md   # empty -> omitted
    assert "Resume from weights" not in md


def test_set_save_decision():
    assert sets.set_save_decision("", ["a"]) == "empty"
    assert sets.set_save_decision("   ", ["a"]) == "empty"
    assert sets.set_save_decision("a", ["a", "b"]) == "exists"
    assert sets.set_save_decision("c", ["a", "b"]) == "ok"


def test_save_set_writes_json_and_markdown(tmp_path: Path):
    root = str(tmp_path)
    sets.save_set("My Set", _rd("My Set", folder="C:/ds"), root)
    d = sets.sets_dir(root)
    assert (d / "My Set.json").is_file()
    assert (d / "My Set.md").is_file()
    # JSON still round-trips; .md is ignored by list_sets
    assert sets.load_set("My Set", root).lora_name == "My Set"
    assert sets.list_sets(root) == ["My Set"]


def test_delete_set_removes_both_files(tmp_path: Path):
    root = str(tmp_path)
    sets.save_set("Gone", _rd("Gone"), root)
    sets.delete_set("Gone", root)
    d = sets.sets_dir(root)
    assert not (d / "Gone.json").exists()
    assert not (d / "Gone.md").exists()


def test_delete_set_tolerates_missing_markdown(tmp_path: Path):
    root = str(tmp_path)
    d = sets.sets_dir(root)
    (d / "Legacy.json").write_text("{}", encoding="utf-8")  # pre-existing, no .md
    sets.delete_set("Legacy", root)  # must not raise
    assert not (d / "Legacy.json").exists()
