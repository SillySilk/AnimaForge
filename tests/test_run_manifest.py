import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.batch import RunDefinition
from core import run_manifest as rm


def _run_dir(out: Path, name: str, *, status, state=True, final=False):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    rd = RunDefinition(lora_name=name, dataset_folder="C:/d", image_count=5,
                       output_dir=str(out))
    rm.write_start(str(d), rd)
    rm.mark(str(d), status)
    if state:
        (d / f"{name}-000004-state").mkdir(exist_ok=True)
    if final:
        (d / f"{name}.safetensors").write_bytes(b"x")
    return d


def test_load_missing_is_empty(tmp_path):
    assert rm.load(str(tmp_path)) == {}


def test_write_start_then_update_then_mark(tmp_path):
    rd = RunDefinition(lora_name="lr", dataset_folder="C:/d", image_count=5)
    rm.write_start(str(tmp_path), rd)
    assert rm.load(str(tmp_path))["status"] == "running"
    rm.update(str(tmp_path), epochs_done=4, global_step=812)
    rm.mark(str(tmp_path), "interrupted")
    d = rm.load(str(tmp_path))
    assert d["epochs_done"] == 4 and d["global_step"] == 812
    assert d["status"] == "interrupted"
    assert d["run"]["lora_name"] == "lr"


def test_find_resumable_picks_interrupted_with_state_and_no_final(tmp_path):
    _run_dir(tmp_path, "good", status="interrupted")
    rd = rm.find_resumable(str(tmp_path))
    assert rd is not None and rd.lora_name == "good"


def test_find_resumable_ignores_done_runs(tmp_path):
    _run_dir(tmp_path, "finished", status="done")
    assert rm.find_resumable(str(tmp_path)) is None


def test_find_resumable_ignores_runs_with_a_final_safetensors(tmp_path):
    _run_dir(tmp_path, "shipped", status="interrupted", final=True)
    assert rm.find_resumable(str(tmp_path)) is None


def test_find_resumable_ignores_runs_with_no_saved_state(tmp_path):
    _run_dir(tmp_path, "nostate", status="interrupted", state=False)
    assert rm.find_resumable(str(tmp_path)) is None


def test_find_resumable_picks_newest_started_among_several(tmp_path):
    _run_dir(tmp_path, "older", status="interrupted")
    d_newer = _run_dir(tmp_path, "newer", status="running")
    # Force a strictly later "started" timestamp than whatever write_start recorded,
    # since both runs may land in the same wall-clock second in a fast test run.
    data = rm.load(str(d_newer))
    data["started"] = "2999-01-01T00:00:00+00:00"
    rm._save(str(d_newer), data)
    rd = rm.find_resumable(str(tmp_path))
    assert rd is not None and rd.lora_name == "newer"


def test_find_resumable_survives_a_corrupt_run_json_sibling(tmp_path):
    _run_dir(tmp_path, "good", status="interrupted")
    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / rm.RUN_FILE).write_text("{ not json", encoding="utf-8")
    rd = rm.find_resumable(str(tmp_path))
    assert rd is not None and rd.lora_name == "good"


# --- falsy-folder guards -----------------------------------------------------
# Path("") normalizes to Path("."), so a falsy run_dir/output_dir must be guarded
# explicitly or a write would land in the process CWD, and find_resumable("") would
# silently scan the CWD instead of reporting "nothing to resume". This exact class
# of bug has shipped twice already (caption_policy.scan(), fixed in caption_manifest) --
# do not reintroduce it here.

def test_falsy_run_dir_readers_return_their_empty_value():
    for run_dir in ("", None):
        assert rm.load(run_dir) == {}


def test_falsy_output_dir_find_resumable_returns_none():
    for output_dir in ("", None):
        assert rm.find_resumable(output_dir) is None


def test_falsy_run_dir_writers_are_no_ops_and_never_touch_cwd(tmp_path):
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rd = RunDefinition(lora_name="lr", dataset_folder="C:/d", image_count=5)
        for run_dir in ("", None):
            rm.write_start(run_dir, rd)
            rm.update(run_dir, global_step=5)
            rm.mark(run_dir, "interrupted")
            assert not (tmp_path / rm.RUN_FILE).exists()
    finally:
        os.chdir(old_cwd)


def test_update_on_a_run_dir_with_no_manifest_yet_is_a_no_op(tmp_path):
    """update()/mark() only amend an existing manifest -- they never create one out
    of nothing (that is write_start's job)."""
    rm.update(str(tmp_path), global_step=5)
    assert not (tmp_path / rm.RUN_FILE).exists()
    rm.mark(str(tmp_path), "done")
    assert not (tmp_path / rm.RUN_FILE).exists()
