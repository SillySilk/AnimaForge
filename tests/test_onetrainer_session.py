import json
import os
import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from core.engines.onetrainer import engine, render, session as session_mod, verify
from core.engines.onetrainer.session import RunParams, TrainingSession
from lora_fixture import DEAD_B, write_lora


class FakeProc(QObject):
    line = Signal(str)
    progress = Signal(dict)
    finished = Signal(int, bool)
    last = None
    start_result = True

    def __init__(self, parent=None):
        super().__init__(parent)
        FakeProc.last = self
        self.started_with = None
        self.cancelled = False

    def start(self, install, config_path, extra_env=None):
        self.started_with = (install, config_path)
        self.extra_env = extra_env
        return type(self).start_result

    def start_script(self, install, script, args, extra_env=None):
        self.started_with = (install, script, list(args))
        self.extra_env = extra_env
        return type(self).start_result

    def cancel(self):
        self.cancelled = True

    def is_running(self):
        return self.started_with is not None


def _params(tmp_path, **over):
    data = tmp_path / "data"; data.mkdir(exist_ok=True)
    out = tmp_path / "out"; out.mkdir(exist_ok=True)
    base = dict(lora_name="Aria", dataset_dir=str(data), image_count=20,
                concept="character", trigger_word="aria_x",
                output_base=str(out), onetrainer_root=str(tmp_path / "ot"))
    base.update(over)
    return RunParams(**base)


def _quiet_preflight(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "preflight", lambda *a, **k: [])
    monkeypatch.setattr(engine, "locate", lambda root: engine.TrainerInstall(
        root=str(tmp_path), python="py.exe", train_script="train.py"))


def test_preflight_failure_emits_failed_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "preflight", lambda *a, **k: ["no trainer"])
    s = TrainingSession(process_factory=FakeProc)
    msgs = []; s.failed.connect(msgs.append)
    assert s.start(_params(tmp_path)) is False
    assert msgs == ["no trainer"]
    assert list((tmp_path / "out").iterdir()) == []


def test_start_writes_the_three_run_files_and_spawns(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    states = []; s.state_changed.connect(states.append)
    assert s.start(_params(tmp_path)) is True
    run = Path(s.run_dir())
    assert (run / "config.json").is_file()
    assert (run / "concepts.json").is_file()
    assert (run / "samples.json").is_file()
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert cfg["model_type"] == "KREA_2"
    assert cfg["concept_file_name"] == str(run / "concepts.json")
    assert FakeProc.last.started_with[1] == str(run / "config.json")
    assert states[-1] == TrainingSession.TRAINING


def test_profiles_switch_travels_as_onetrainer_environment(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    assert s.start(_params(tmp_path)) is True
    assert FakeProc.last.extra_env is None            # off by default: nothing added
    s2 = TrainingSession(process_factory=FakeProc)
    assert s2.start(_params(tmp_path, lora_name="Bria", debug_profiles=True)) is True
    assert FakeProc.last.extra_env == {"OT_DEBUG_PROFILES": "1"}


def test_profiler_traces_are_moved_into_the_run_folder(tmp_path, monkeypatch):
    """OneTrainer drops them in its own folder (its cwd); the run folder is the
    record, so they belong there. A trace older than this run stays put."""
    _quiet_preflight(monkeypatch, tmp_path)
    root = tmp_path                                   # locate() stub: install == tmp_path
    stale = root / "profile-step10-old.json"
    stale.write_text("{}", encoding="utf-8")
    os.utime(stale, (time.time() - 3600, time.time() - 3600))
    s = TrainingSession(process_factory=FakeProc)
    assert s.start(_params(tmp_path, debug_profiles=True)) is True
    fresh = root / "profile-step10-new.json"
    fresh.write_text("{}", encoding="utf-8")
    (root / "memory-step0-new.pickle").write_bytes(b"x")
    FakeProc.last.finished.emit(1, False)             # the outcome is beside the point
    run = Path(s.run_dir())
    assert (run / "profile-step10-new.json").is_file()
    assert (run / "memory-step0-new.pickle").is_file()
    assert not fresh.exists()
    assert stale.exists()


def test_relative_output_base_is_resolved_before_anything_is_written(tmp_path, monkeypatch):
    # The subprocess runs with cwd = the OneTrainer root, so EVERY path the session
    # hands it must be absolute — a portable output_base has to be resolved against
    # the app root here, not left for whatever cwd the trainer happens to have.
    _quiet_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(session_mod.paths, "app_root", lambda: tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    assert s.start(_params(tmp_path, output_base="out")) is True

    run = tmp_path / "out" / "Aria"
    assert Path(s.run_dir()) == run
    assert FakeProc.last.started_with[1] == str(run / "config.json")
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    for key in ("concept_file_name", "sample_definition_file_name",
                "workspace_dir", "cache_dir", "output_model_destination"):
        assert os.path.isabs(cfg[key]), f"{key} is relative: {cfg[key]!r}"


def test_target_steps_shortens_the_run_and_scales_the_warmup(tmp_path, monkeypatch):
    # 47 images at batch 2 -> 24 steps/epoch. An 800-step target is 33 epochs, and
    # the LR warmup has to come down with it or 200 of those 792 steps are warmup.
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    assert s.start(_params(tmp_path, image_count=47, target_steps=800)) is True
    cfg = json.loads((Path(s.run_dir()) / "config.json").read_text(encoding="utf-8"))
    assert cfg["epochs"] == 33
    assert cfg["learning_rate_warmup_steps"] == 79


def test_the_run_config_freezes_prodigy_at_three_quarters_of_the_run(tmp_path, monkeypatch):
    # 47 images -> 24 steps/epoch, 57 epochs = 1368 steps; Prodigy stops
    # re-estimating at step 1026 and the closing quarter trains at a settled rate.
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    assert s.start(_params(tmp_path, image_count=47, target_steps=0)) is True
    cfg = json.loads((Path(s.run_dir()) / "config.json").read_text(encoding="utf-8"))
    assert cfg["optimizer"]["prodigy_steps"] == 1026


def test_auto_target_keeps_the_per_image_recipe(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    assert s.start(_params(tmp_path, image_count=47, target_steps=0)) is True
    cfg = json.loads((Path(s.run_dir()) / "config.json").read_text(encoding="utf-8"))
    assert cfg["epochs"] == 57              # 47 images on the sublinear auto curve
    assert cfg["learning_rate_warmup_steps"] == 137


def test_progress_total_is_the_planned_step_count_from_the_first_event(tmp_path, monkeypatch):
    # The band has to be sized before the first `step:` bar arrives, or global_total
    # is epoch_total * 0 = 0 and the bar reads 0/0 through the whole opening stretch.
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    got = []; s.progress.connect(got.append)
    s.start(_params(tmp_path, image_count=47))
    FakeProc.last.progress.emit({"name": "epoch", "n": 0, "total": 78, "loss": None})
    assert got[-1]["global_total"] == 78 * 24
    assert got[-1]["step_total"] == 24
    FakeProc.last.progress.emit({"name": "step", "n": 5, "total": 24, "loss": 0.2})
    assert got[-1]["global_step"] == 5
    assert got[-1]["global_total"] == 78 * 24


def test_progress_is_globalized_across_epochs(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    got = []; s.progress.connect(got.append)
    s.start(_params(tmp_path))
    FakeProc.last.progress.emit({"name": "epoch", "n": 1, "total": 5, "loss": None})
    FakeProc.last.progress.emit({"name": "step", "n": 3, "total": 10, "loss": 0.5})
    d = got[-1]
    assert d["epoch"] == 1 and d["epoch_total"] == 5
    assert d["step"] == 3 and d["step_total"] == 10
    assert d["global_step"] == 1 * 10 + 3 and d["global_total"] == 5 * 10
    assert d["loss"] == 0.5


def _output_path(s) -> str:
    return json.loads((Path(s.run_dir()) / "config.json").read_text(
        encoding="utf-8"))["output_model_destination"]


def _render_ok(s):
    """The second process of a finished run: the final-preview render. A run
    lands as DONE only after it has come back -- here with one preview written,
    which is what a clean exit means."""
    assert FakeProc.last.started_with[1] == render.SCRIPT, FakeProc.last.started_with
    folder = Path(render.destination(s.run_dir())) / "0 - final"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "final-0.jpg").write_bytes(b"jpg")
    FakeProc.last.finished.emit(0, False)


def test_exit_zero_with_a_trained_output_is_done(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    finished = []; s.done.connect(finished.append)
    s.start(_params(tmp_path))
    out = _output_path(s)
    write_lora(out)
    FakeProc.last.finished.emit(0, False)
    _render_ok(s)
    assert finished == [out]
    assert not s.is_running()


def test_a_finished_run_labels_the_lora_with_its_trained_words(tmp_path, monkeypatch):
    """OneTrainer writes the file with no metadata at all, so nothing downstream can
    say what word calls it — and the upload form that used to work it out from the
    tags no longer does. The run that made it is the last place that knows."""
    from core.engines.onetrainer import metadata
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    finished = []; s.done.connect(finished.append)
    p = _params(tmp_path, quality_prefix="masterpiece")
    for i in range(3):
        (Path(p.dataset_dir) / f"img{i}.txt").write_text(
            "aria_x, masterpiece, a kitchen", encoding="utf-8")
    s.start(p)
    out = _output_path(s)
    write_lora(out)
    FakeProc.last.finished.emit(0, False)
    _render_ok(s)

    assert finished == [out]
    meta = metadata.read_metadata(out)
    assert meta["ss_output_name"] == "Aria"
    assert "aria_x" in meta["ss_trained_words"]
    assert "masterpiece" in meta["ss_trained_words"]
    assert Path(metadata.sidecar_path(out)).is_file()


def test_a_label_that_cannot_be_written_never_fails_a_finished_run(tmp_path, monkeypatch):
    """The LoRA is an hour old and already verified. Losing it because a metadata
    rewrite failed would be the worst trade in the app — so the failure goes to
    trainer.log and the run still lands as done."""
    from core.engines.onetrainer import metadata
    _quiet_preflight(monkeypatch, tmp_path)

    def _boom(*a, **k):
        raise metadata.MetadataError("disk went away")
    monkeypatch.setattr(metadata, "stamp", _boom)

    s = TrainingSession(process_factory=FakeProc)
    finished = []; s.done.connect(finished.append)
    failed = []; s.failed.connect(failed.append)
    s.start(_params(tmp_path))
    out = _output_path(s)
    write_lora(out)
    FakeProc.last.finished.emit(0, False)
    _render_ok(s)

    assert finished == [out] and failed == []
    log = (Path(s.run_dir()) / "trainer.log").read_text(encoding="utf-8")
    assert "disk went away" in log


def test_a_no_op_lora_fails_instead_of_reporting_success(tmp_path, monkeypatch):
    """A clean exit and a fresh file are not proof the run learned anything.
    run_B (2026-08-17) satisfied both and produced a LoRA that changed
    nothing, and the session reported DONE."""
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    done = []; s.done.connect(done.append)
    msgs = []; s.failed.connect(msgs.append)
    states = []; s.state_changed.connect(states.append)
    s.start(_params(tmp_path))
    write_lora(_output_path(s), b_value=DEAD_B)
    FakeProc.last.finished.emit(0, False)
    assert done == []
    assert msgs and "did not learn" in msgs[0].lower()
    assert states[-1] == TrainingSession.FAILED


def test_a_rejected_lora_is_left_on_disk_for_inspection(tmp_path, monkeypatch):
    """An hour of training. Report it as failed, never delete it."""
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    s.start(_params(tmp_path))
    out = _output_path(s)
    write_lora(out, b_value=DEAD_B)
    FakeProc.last.finished.emit(0, False)
    assert Path(out).is_file()


def test_an_unloadable_namespace_fails_the_run(tmp_path, monkeypatch):
    """The other silent failure: a trained LoRA the image tool cannot match."""
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    done = []; s.done.connect(done.append)
    msgs = []; s.failed.connect(msgs.append)
    s.start(_params(tmp_path))
    write_lora(_output_path(s), prefix="transformer")
    FakeProc.last.finished.emit(0, False)
    assert done == []
    assert msgs and "diffusion_model" in msgs[0]


def test_exit_zero_without_output_fails(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    msgs = []; s.failed.connect(msgs.append)
    s.start(_params(tmp_path))
    FakeProc.last.finished.emit(0, False)
    assert msgs and "no new output" in msgs[0].lower()


def test_nonzero_exit_fails_with_log_tail(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    msgs = []; s.failed.connect(msgs.append)
    s.start(_params(tmp_path))
    FakeProc.last.line.emit("RuntimeError: boom")
    FakeProc.last.finished.emit(1, False)
    assert msgs and "boom" in msgs[0]
    log = (Path(s.run_dir()) / "trainer.log").read_text(encoding="utf-8")
    assert "boom" in log


def test_oom_failure_carries_the_vram_hint(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    msgs = []; s.failed.connect(msgs.append)
    s.start(_params(tmp_path))
    FakeProc.last.finished.emit(1, True)
    assert msgs and "memory" in msgs[0].lower() and "GPU" in msgs[0]


def test_cancel_lands_in_canceled_not_failed(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    msgs = []; s.failed.connect(msgs.append)
    states = []; s.state_changed.connect(states.append)
    s.start(_params(tmp_path))
    s.cancel()
    assert FakeProc.last.cancelled
    FakeProc.last.finished.emit(-1, False)
    assert states[-1] == TrainingSession.CANCELED
    assert msgs == []


def test_second_start_while_running_is_refused(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    assert s.start(_params(tmp_path)) is True
    assert s.start(_params(tmp_path)) is False


def test_spawn_failure_fails_cleanly(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    msgs = []; s.failed.connect(msgs.append)
    FakeProc.start_result = False
    assert s.start(_params(tmp_path)) is False
    assert msgs and "Could not start" in msgs[0]
    assert not s.is_running()
    FakeProc.start_result = True


def test_stale_output_from_previous_run_is_not_done(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    done = []; s.done.connect(done.append)
    msgs = []; s.failed.connect(msgs.append)
    s.start(_params(tmp_path))
    out = json.loads((Path(s.run_dir()) / "config.json").read_text(
        encoding="utf-8"))["output_model_destination"]
    Path(out).write_bytes(b"x")
    old_time = time.time() - 100
    os.utime(out, (old_time, old_time))
    FakeProc.last.finished.emit(0, False)
    assert done == []
    assert msgs and "no new output" in msgs[0].lower()


def test_run_dir_points_at_failed_attempt_after_generate_error(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    assert s.start(_params(tmp_path)) is True
    first_run_dir = s.run_dir()
    assert first_run_dir
    out = json.loads((Path(first_run_dir) / "config.json").read_text(
        encoding="utf-8"))["output_model_destination"]
    Path(out).write_bytes(b"x")
    FakeProc.last.finished.emit(0, False)
    assert not s.is_running()

    monkeypatch.setattr(session_mod.preset, "load_template",
                       lambda: (_ for _ in ()).throw(session_mod.preset.TemplateError("boom")))
    s2 = TrainingSession(process_factory=FakeProc)
    assert s2.start(_params(tmp_path)) is False
    assert s2.run_dir() == ""


# ---------------------------------------------------------------------------
# The failure the user actually hit: Windows killed python.exe with an access
# violation when the system commit limit was reached mid-quantization. There is
# no traceback and no CUDA OOM marker to catch -- the exit code is the only
# evidence, and "Training failed (exit code -1073741819)" under a wall of tqdm
# output explains nothing.

def test_a_native_crash_is_explained_instead_of_shown_as_a_number(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(session_mod.sysmem, "biggest_consumer_note", lambda: "")
    s = TrainingSession(process_factory=FakeProc)
    msgs = []; s.failed.connect(msgs.append)
    s.start(_params(tmp_path))
    FakeProc.last.line.emit("Quantizing model weights:  36%|###5 | 212/595")
    FakeProc.last.finished.emit(session_mod.ACCESS_VIOLATION, False)
    assert msgs
    assert "memory" in msgs[0].lower()
    assert "-1073741819" not in msgs[0]
    # The tail still ships: it is how the user knows where it died.
    assert "212/595" in msgs[0]


def test_a_native_crash_names_what_was_holding_the_memory(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(session_mod.sysmem, "biggest_consumer_note",
                        lambda: " The biggest consumer right now is thing.exe (52 GB).")
    s = TrainingSession(process_factory=FakeProc)
    msgs = []; s.failed.connect(msgs.append)
    s.start(_params(tmp_path))
    FakeProc.last.finished.emit(session_mod.ACCESS_VIOLATION, False)
    assert msgs and "thing.exe" in msgs[0]


def test_a_cuda_oom_still_gets_the_vram_hint_not_the_system_one(tmp_path, monkeypatch):
    """A real CUDA OOM is more specific than the crash code; it wins."""
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    msgs = []; s.failed.connect(msgs.append)
    s.start(_params(tmp_path))
    FakeProc.last.finished.emit(session_mod.ACCESS_VIOLATION, True)
    assert msgs and "GPU" in msgs[0]


def test_an_ordinary_failure_still_reports_its_exit_code(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    msgs = []; s.failed.connect(msgs.append)
    s.start(_params(tmp_path))
    FakeProc.last.finished.emit(9, False)
    assert msgs and "exit code 9" in msgs[0]


def test_cancel_tells_the_ui_the_run_was_canceled(tmp_path, monkeypatch):
    """A cancelled run used to end in silence: the state went CANCELED, no signal
    left the session, and Home kept "Starting trainer…" with Train greyed out
    until the app was restarted (run_C, 2026-09-04)."""
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    told = []; s.canceled.connect(lambda: told.append(True))
    msgs = []; s.failed.connect(msgs.append)
    s.start(_params(tmp_path))
    s.cancel()
    FakeProc.last.finished.emit(1, False)   # taskkill's exit code
    assert told == [True]
    assert msgs == []


# ---------------------------------------------------------------------------
# The end of the run went bad. run_H (2026-09-04): every check green, and the
# file rendered a mosaic. Its loss had bottomed at epoch 52 of 67 and finished 38%
# higher; the snapshot from epoch 50 was the LoRA that worked.

def _overshoot_lines(total=67, best_epoch=52, best=0.0613, final=0.0844):
    lines = []
    for e in range(1, total + 1):
        if e <= best_epoch:
            loss = best + (0.09 - best) * (best_epoch - e) / (best_epoch - 1)
        else:
            loss = best + (final - best) * (e - best_epoch) / (total - best_epoch)
        lines.append(f"step: 100%|##########| 17/17 [00:57<00:00, loss={loss:.4f}, smooth loss={loss:.4f}]")
        lines.append(f"epoch: {int(100 * e / total):3d}%|##########| {e}/{total} [1:12:15<00:00, 62.34s/it]")
    return lines


def _snapshot(run_dir: Path, epoch: int):
    save = run_dir / "workspace" / "save"
    save.mkdir(parents=True, exist_ok=True)
    return write_lora(save / f"2026-09-04_22-00-00-save-{epoch * 17}-{epoch}-0.safetensors",
                      b_value=5.0e-2)


def test_an_overshot_run_ships_the_snapshot_from_before_the_turn(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    done = []; s.done.connect(done.append)
    failed = []; s.failed.connect(failed.append)
    s.start(_params(tmp_path))
    run = Path(s.run_dir())
    for line in _overshoot_lines():
        FakeProc.last.line.emit(line)
    for epoch in (45, 50, 55, 60, 65):
        _snapshot(run, epoch)
    out = _output_path(s)
    write_lora(out, b_value=9.0e-2)                        # the overshot end
    FakeProc.last.finished.emit(0, False)
    _render_ok(s)

    assert failed == [] and done == [out]
    # The snapshot's B side was written at 5e-2, the overshot end's at 9e-2.
    assert 4.0e-2 < verify.peak_lora_b_abs(out) < 6.0e-2
    kept = Path(out).with_name("Aria.overshot.safetensors")
    assert kept.is_file() and 8.0e-2 < verify.peak_lora_b_abs(str(kept)) < 1.0e-1
    assert any("epoch 50" in n for n in s.notes())
    assert any("overshot" in n.lower() for n in s.notes())
    assert not list((run / "workspace" / "save").glob("*.safetensors"))   # tidied
    log = (run / "trainer.log").read_text(encoding="utf-8")
    assert "epoch 50" in log


def test_the_shipped_snapshot_carries_the_trained_words(tmp_path, monkeypatch):
    from core.engines.onetrainer import metadata
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    p = _params(tmp_path)
    (Path(p.dataset_dir) / "img0.txt").write_text("aria_x, a kitchen", encoding="utf-8")
    s.start(p)
    run = Path(s.run_dir())
    for line in _overshoot_lines():
        FakeProc.last.line.emit(line)
    _snapshot(run, 50)
    out = _output_path(s)
    write_lora(out, b_value=9.0e-2)
    FakeProc.last.finished.emit(0, False)
    _render_ok(s)
    assert 4.0e-2 < verify.peak_lora_b_abs(out) < 6.0e-2        # it IS the snapshot
    assert metadata.read_metadata(out)["ss_trained_words"].startswith("aria_x")


def test_an_overshot_run_with_no_snapshot_fails_with_the_explanation(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    done = []; s.done.connect(done.append)
    failed = []; s.failed.connect(failed.append)
    s.start(_params(tmp_path))
    for line in _overshoot_lines():
        FakeProc.last.line.emit(line)
    out = _output_path(s)
    write_lora(out, b_value=9.0e-2)
    FakeProc.last.finished.emit(0, False)
    assert done == []
    assert failed and "overshot" in failed[0].lower() and "epoch 52" in failed[0]
    assert "snapshot" in failed[0].lower()
    assert Path(out).is_file()                              # never deleted


def test_a_snapshot_only_from_after_the_turn_does_not_ship(tmp_path, monkeypatch):
    """A snapshot from epoch 55 is already past the best epoch (52): shipping it
    would ship a milder version of the same overshoot."""
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    done = []; s.done.connect(done.append)
    failed = []; s.failed.connect(failed.append)
    s.start(_params(tmp_path))
    run = Path(s.run_dir())
    for line in _overshoot_lines():
        FakeProc.last.line.emit(line)
    _snapshot(run, 55)
    write_lora(_output_path(s), b_value=9.0e-2)
    FakeProc.last.finished.emit(0, False)
    assert done == [] and failed


def test_a_healthy_run_discards_its_snapshots_and_ships_the_end(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    done = []; s.done.connect(done.append)
    s.start(_params(tmp_path))
    run = Path(s.run_dir())
    for line in _overshoot_lines(best_epoch=46, final=0.0756, total=46):   # ends at its best
        FakeProc.last.line.emit(line)
    for epoch in (40, 45):
        _snapshot(run, epoch)
    out = _output_path(s)
    write_lora(out)
    FakeProc.last.finished.emit(0, False)
    _render_ok(s)
    assert done == [out]
    assert s.notes() == []
    assert not Path(out).with_name("Aria.overshot.safetensors").exists()
    assert not list((run / "workspace" / "save").glob("*.safetensors"))


# ---------------------------------------------------------------------------
# The trainer's previews stop at the last sampling interval; run_H's showed
# step 989 of 1139 and the file it shipped was the state at 1139. Once the file
# has passed every check, it is loaded back and its preview prompts rendered,
# so the previews always show the file that ships.

def test_a_finished_lora_is_rendered_before_the_run_lands_as_done(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    done = []; s.done.connect(done.append)
    states = []; s.state_changed.connect(states.append)
    s.start(_params(tmp_path))
    run = Path(s.run_dir())
    out = _output_path(s)
    write_lora(out)
    trainer = FakeProc.last
    trainer.finished.emit(0, False)

    assert done == []                                   # not yet
    assert states[-1] == TrainingSession.RENDERING
    assert s.is_running()
    renderer = FakeProc.last
    assert renderer is not trainer
    install, script, args = renderer.started_with
    assert script == render.SCRIPT
    assert args == ["--config-path", str(run / "config.json"),
                    "--lora", out,
                    "--destination", str(run / "workspace" / "samples" / "final")]
    renderer.finished.emit(0, False)
    assert done == [out]
    assert states[-1] == TrainingSession.DONE
    assert not s.is_running()


def test_the_renders_output_is_kept_in_the_run_log(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    s.start(_params(tmp_path))
    write_lora(_output_path(s))
    FakeProc.last.finished.emit(0, False)
    FakeProc.last.line.emit("[animaforge] final previews written: 3")
    FakeProc.last.finished.emit(0, False)
    log = (Path(s.run_dir()) / "trainer.log").read_text(encoding="utf-8")
    assert "final previews written: 3" in log


def test_a_failed_render_never_fails_the_finished_run(tmp_path, monkeypatch):
    """The LoRA is an hour old and verified; the previews are a courtesy."""
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    done = []; s.done.connect(done.append)
    failed = []; s.failed.connect(failed.append)
    s.start(_params(tmp_path))
    out = _output_path(s)
    write_lora(out)
    FakeProc.last.finished.emit(0, False)
    FakeProc.last.line.emit("RuntimeError: CUDA error: out of memory")
    FakeProc.last.finished.emit(1, True)
    assert done == [out] and failed == []
    assert any("preview" in n.lower() for n in s.notes())
    log = (Path(s.run_dir()) / "trainer.log").read_text(encoding="utf-8")
    assert "out of memory" in log


def test_a_render_that_cannot_start_is_a_note_not_a_failure(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    done = []; s.done.connect(done.append)
    s.start(_params(tmp_path))
    out = _output_path(s)
    write_lora(out)
    FakeProc.start_result = False
    try:
        FakeProc.last.finished.emit(0, False)
    finally:
        FakeProc.start_result = True
    assert done == [out]
    assert any("preview" in n.lower() for n in s.notes())
    assert not s.is_running()


def test_cancel_during_the_render_still_lands_the_lora_as_done(tmp_path, monkeypatch):
    """Cancel kills the render, not the hour of training behind it."""
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    done = []; s.done.connect(done.append)
    told = []; s.canceled.connect(lambda: told.append(True))
    s.start(_params(tmp_path))
    out = _output_path(s)
    write_lora(out)
    FakeProc.last.finished.emit(0, False)
    s.cancel()
    assert FakeProc.last.cancelled
    FakeProc.last.finished.emit(1, False)
    assert done == [out] and told == []
    assert any("preview" in n.lower() for n in s.notes())


def test_a_rejected_lora_is_never_rendered(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    s.start(_params(tmp_path))
    write_lora(_output_path(s), b_value=DEAD_B)
    trainer = FakeProc.last
    trainer.finished.emit(0, False)
    assert FakeProc.last is trainer                     # no second process
    assert not s.is_running()


# ---------------------------------------------------------------------------
# The loss blew up mid-run. run_S (2026-10-08) went from 0.129
# to 0.215 at epoch 11 of 33, trained two more hours of noise, drifted back to
# +12% over its best -- under the overshoot line -- and shipped the noise.

def _blowup_lines(upto=11, total=33):
    losses = [0.129, 0.136, 0.133, 0.134, 0.135, 0.145, 0.135, 0.137, 0.138, 0.129,
              0.215, 0.263]
    out = []
    for e, loss in enumerate(losses[:upto], start=1):
        out.append(f"step: 100%|##########| 72/72 [06:18<00:00, loss={loss}, smooth loss={loss}]")
        out.append(f"epoch: {int(100 * e / total):3d}%|###| {e}/{total} [1:00:00<00:00, 298.93s/it]")
    return out


def test_a_blowup_stops_the_trainer_at_the_epoch_it_happens(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    s.start(_params(tmp_path))
    proc = FakeProc.last
    for line in _blowup_lines(upto=10):
        proc.line.emit(line)
    assert not proc.cancelled
    for line in _blowup_lines(upto=11)[-2:]:
        proc.line.emit(line)
    assert proc.cancelled
    log = (Path(s.run_dir()) / "trainer.log")
    s._log.flush()
    assert "blew up" in log.read_text(encoding="utf-8")


def test_a_stopped_blowup_ships_the_last_snapshot_before_it(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    done = []; s.done.connect(done.append)
    failed = []; s.failed.connect(failed.append)
    s.start(_params(tmp_path))
    run = Path(s.run_dir())
    for epoch in (4, 8):
        _snapshot(run, epoch)
    for line in _blowup_lines():
        FakeProc.last.line.emit(line)
    FakeProc.last.finished.emit(1, False)                 # killed by the stop
    _render_ok(s)
    out = _output_path(s)
    assert failed == [] and done == [out]
    assert 4.0e-2 < verify.peak_lora_b_abs(out) < 6.0e-2   # it IS the snapshot
    assert any("blew up" in n and "epoch 8" in n for n in s.notes())
    assert not list((run / "workspace" / "save").glob("*.safetensors"))


def test_a_stopped_blowup_with_no_snapshot_fails_and_says_why(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    done = []; s.done.connect(done.append)
    failed = []; s.failed.connect(failed.append)
    s.start(_params(tmp_path))
    for line in _blowup_lines():
        FakeProc.last.line.emit(line)
    FakeProc.last.finished.emit(1, False)
    assert done == []
    assert failed and "blew up" in failed[0] and "snapshot" in failed[0]
