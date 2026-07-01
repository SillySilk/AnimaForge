import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.state_utils import find_saved_state


def test_none_when_absent(tmp_path):
    assert find_saved_state(str(tmp_path), "lora") is None


def test_finds_last_state(tmp_path):
    (tmp_path / "lora-state").mkdir()
    assert find_saved_state(str(tmp_path), "lora").endswith("lora-state")


def test_picks_newest_epoch_state(tmp_path):
    old = tmp_path / "lora-000002-state"
    old.mkdir()
    time.sleep(0.01)
    new = tmp_path / "lora-000006-state"
    new.mkdir()
    got = find_saved_state(str(tmp_path), "lora")
    assert got.endswith("lora-000006-state")


def test_ignores_other_names(tmp_path):
    (tmp_path / "other-state").mkdir()
    assert find_saved_state(str(tmp_path), "lora") is None


def test_finds_state_in_per_run_dir(tmp_path):
    run_dir = tmp_path / "lora"
    (run_dir / "lora-state").mkdir(parents=True)
    got = find_saved_state(str(tmp_path), "lora")
    assert got is not None and got.endswith("lora-state")
    assert str(run_dir) in got


def test_falls_back_to_legacy_flat_layout(tmp_path):
    # No per-run dir; state sits directly under the base (older runs).
    (tmp_path / "lora-000004-state").mkdir()
    got = find_saved_state(str(tmp_path), "lora")
    assert got is not None and got.endswith("lora-000004-state")
