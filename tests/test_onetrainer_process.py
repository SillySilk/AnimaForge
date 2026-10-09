import json
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from core.engines.onetrainer.engine import TrainerInstall
from core.engines.onetrainer.process import TrainerProcess, parse_progress_line, split_lines

FAKE = str(Path(__file__).resolve().parent / "fake_trainer.py")

def _fake_install(root):
    return TrainerInstall(root=str(root), python=sys.executable, train_script=FAKE)

def _write_cfg(tmp_path, mode, out=""):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(
        {"fake_mode": mode, "output_model_destination": out}), encoding="utf-8")
    return str(cfg)

def _run(tmp_path, mode):
    proc = TrainerProcess()
    got = {"lines": [], "progress": [], "finished": None}
    proc.line.connect(got["lines"].append)
    proc.progress.connect(got["progress"].append)
    proc.finished.connect(lambda code, oom: got.__setitem__("finished", (code, oom)))
    assert proc.start(_fake_install(tmp_path), _write_cfg(tmp_path, mode))
    assert proc.wait(20000)     # test-only pump: waitForFinished + drain
    return got

# ---- pure parser ----

def test_parse_step_line_with_loss():
    d = parse_progress_line(
        "step:  50%|#####     | 2/4 [00:00<00:00,  9.90it/s, loss=0.25, smooth loss=0.24]")
    assert d == {"name": "step", "n": 2, "total": 4, "loss": 0.25}

def test_parse_epoch_line_without_loss():
    d = parse_progress_line("epoch:   0%|          | 0/2 [00:00<?, ?it/s]")
    assert d == {"name": "epoch", "n": 0, "total": 2, "loss": None}

def test_parse_rejects_ordinary_log_lines():
    assert parse_progress_line("Saving C:\\x\\model.safetensors") is None
    assert parse_progress_line("") is None

def test_split_lines_handles_tqdm_carriage_returns():
    lines, rest = split_lines("step: 1/4\rstep: 2/4\nSaving\npartial")
    assert lines == ["step: 1/4", "step: 2/4", "Saving"]
    assert rest == "partial"

# ---- real subprocess against the fake trainer ----

def test_start_script_runs_any_script_under_the_installs_python(tmp_path):
    """The final-preview render is a second script run the same way train.py
    is: the install's interpreter, cwd at its root, output merged and lined."""
    script = tmp_path / "hello.py"
    script.write_text("import sys; print('hello', sys.argv[1:]); sys.exit(3)",
                      encoding="utf-8")
    proc = TrainerProcess()
    got = {"lines": [], "finished": None}
    proc.line.connect(got["lines"].append)
    proc.finished.connect(lambda code, oom: got.__setitem__("finished", (code, oom)))
    assert proc.start_script(_fake_install(tmp_path), str(script), ["--x", "1"])
    assert proc.wait(20000)
    assert got["finished"] == (3, False)
    assert any("hello ['--x', '1']" in line for line in got["lines"])


def test_extra_env_reaches_the_trainer(tmp_path):
    """OneTrainer's own switches (OT_DEBUG_PROFILES) travel as environment, laid
    over the inherited one -- the child must see them, and PYTHONUNBUFFERED must
    survive alongside."""
    proc = TrainerProcess()
    lines = []
    proc.line.connect(lines.append)
    assert proc.start(_fake_install(tmp_path), _write_cfg(tmp_path, "ok"),
                      extra_env={"KF_ECHO_ENV": "seen"})
    assert proc.wait(20000)
    assert "env:seen" in lines


def test_ok_run_emits_progress_and_exit_zero(tmp_path):
    got = _run(tmp_path, "ok")
    assert got["finished"] == (0, False)
    assert any(p["name"] == "step" and p["n"] == 4 for p in got["progress"])
    assert any(p.get("loss") == 0.25 for p in got["progress"])

def test_failing_run_reports_exit_code_and_no_oom(tmp_path):
    got = _run(tmp_path, "fail")
    assert got["finished"] == (1, False)

def test_oom_is_detected(tmp_path):
    got = _run(tmp_path, "oom")
    assert got["finished"] == (1, True)

def test_cancel_kills_a_hung_run(tmp_path):
    proc = TrainerProcess()
    done = []
    proc.finished.connect(lambda code, oom: done.append(code))
    assert proc.start(_fake_install(tmp_path), _write_cfg(tmp_path, "hang"))
    proc.cancel()
    assert proc.wait(20000)
    assert done and done[0] != 0

def test_oom_truncated_is_detected(tmp_path):
    """OOM marker in partial line (no trailing newline) is detected."""
    got = _run(tmp_path, "oom-truncated")
    assert got["finished"] == (1, True)

def test_progress_arrives_while_running(tmp_path):
    """Progress signals arrive incrementally, not buffered until exit."""
    import time
    proc = TrainerProcess()
    progress_while_running = []
    proc.progress.connect(
        lambda d: progress_while_running.append(proc.is_running()))
    assert proc.start(_fake_install(tmp_path), _write_cfg(tmp_path, "slow"))
    # Pump events with a loop to simulate live event processing
    start = time.time()
    timeout_s = 25
    while proc.is_running() and (time.time() - start) < timeout_s:
        _app.processEvents()
        time.sleep(0.1)
    # Clean up after loop
    if proc.is_running():
        proc.wait(5000)
    else:
        proc.wait(5000)  # drain any remaining data
    # Assert at least one progress signal arrived while process was running
    assert any(progress_while_running), "Progress signals must arrive live, not buffered"
