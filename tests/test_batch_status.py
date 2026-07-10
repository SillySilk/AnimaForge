import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QObject, Signal

from core.batch import RunDefinition, QUEUED, RUNNING, DONE, FAILED
from core.batch_status import StatusWriter
from core.paths import run_output_dir
from core.step_calculator import calculate_training_params


class FakeRunner(QObject):
    """Emits the same signals BatchRunner does, with the same status mutations."""
    run_started = Signal(int)
    run_finished = Signal(int, bool)
    run_phase = Signal(int, str)
    batch_finished = Signal()
    log_line = Signal(str)
    progress_updated = Signal(int, int)


def _rd(**kw):
    base = dict(lora_name="verify_girl", dataset_folder="C:/d",
                image_count=12, target_steps=1000, output_dir="C:/out")
    base.update(kw)
    return RunDefinition(**base)


def _setup(tmp_path, runs=None):
    runner = FakeRunner()
    runs = runs if runs is not None else [_rd()]
    path = tmp_path / "batch_status.json"
    writer = StatusWriter(runner, runs, str(path))
    return runner, runs, path, writer


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_initial_snapshot_lists_all_runs_queued(tmp_path):
    _, runs, path, _ = _setup(tmp_path, runs=[_rd(), _rd(lora_name="second")])
    data = _read(path)
    assert [d["lora_name"] for d in data] == ["verify_girl", "second"]
    total = calculate_training_params(12, target_steps=1000)["total_steps"]
    for d in data:
        assert d["state"] == QUEUED
        assert d["current_step"] == 0
        assert d["target_steps"] == total
        assert d["started_at"] is None
        assert d["finished_at"] is None
        assert d["lora_path"] is None
        assert d["error_tail"] is None


def test_run_started_marks_running_with_timestamp(tmp_path):
    runner, runs, path, _ = _setup(tmp_path)
    runs[0].status = RUNNING          # BatchRunner mutates before emitting
    runner.run_started.emit(0)
    d = _read(path)[0]
    assert d["state"] == RUNNING
    assert d["started_at"] is not None


def test_progress_updates_current_step(tmp_path):
    runner, runs, path, _ = _setup(tmp_path)
    runs[0].status = RUNNING
    runner.run_started.emit(0)
    runner.progress_updated.emit(0, 120)
    assert _read(path)[0]["current_step"] == 120


def test_success_records_done_and_lora_path(tmp_path):
    runner, runs, path, _ = _setup(tmp_path)
    runs[0].status = RUNNING
    runner.run_started.emit(0)
    runs[0].status = DONE
    runner.run_finished.emit(0, True)
    d = _read(path)[0]
    assert d["state"] == DONE
    assert d["finished_at"] is not None
    expected = str(Path(run_output_dir("C:/out", "verify_girl")) / "verify_girl.safetensors")
    assert d["lora_path"] == expected
    assert d["error_tail"] is None


def test_failure_records_error_tail_from_log_lines(tmp_path):
    runner, runs, path, _ = _setup(tmp_path)
    runs[0].status = RUNNING
    runner.run_started.emit(0)
    for i in range(20):
        runner.log_line.emit(f"line {i}")
    runs[0].status = FAILED
    runner.run_finished.emit(0, False)
    d = _read(path)[0]
    assert d["state"] == FAILED
    assert d["lora_path"] is None
    assert "line 19" in d["error_tail"]
    assert "line 4" not in d["error_tail"]   # ring buffer keeps only the last 15


def test_file_is_valid_json_after_every_event(tmp_path):
    runner, runs, path, _ = _setup(tmp_path)
    runs[0].status = RUNNING
    runner.run_started.emit(0)
    for step in range(0, 50, 10):
        runner.progress_updated.emit(0, step)
        _read(path)   # raises if a write was ever non-atomic/partial
    runs[0].status = DONE
    runner.run_finished.emit(0, True)
    runner.batch_finished.emit()
    assert _read(path)[0]["state"] == DONE


def test_status_payload_carries_the_phase(tmp_path):
    runner, runs, path, _ = _setup(tmp_path, runs=[_rd(lora_name="a"),
                                                    _rd(lora_name="b")])
    # Initial snapshot: no phase reported yet for any run.
    assert _read(path)[0]["phase"] == ""

    runner.run_phase.emit(0, "captioning")
    assert _read(path)[0]["phase"] == "captioning"

    runner.run_phase.emit(0, "training")
    data = _read(path)
    assert data[0]["phase"] == "training"
    assert data[1]["phase"] == ""      # untouched run has no phase yet
