import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from core import past_runs, run_manifest
from core.batch import RunDefinition
from core.engines.onetrainer import engine
from core.engines.onetrainer.session import TrainingSession
from core.trainer import TrainingProcess

from test_onetrainer_session import FakeProc, _params, _quiet_preflight


def _dataset(tmp_path, name="data", images=3):
    d = tmp_path / name
    d.mkdir()
    for i in range(images):
        (d / f"img{i}.png").write_bytes(b"x")
        (d / f"img{i}.txt").write_text("caption", encoding="utf-8")
    return d


def _kohya_run(out, name, dataset, steps=1152, epochs=12, status=run_manifest.DONE,
               final=True):
    run = out / name
    run.mkdir(parents=True)
    rd = RunDefinition(lora_name=name, dataset_folder=str(dataset), image_count=3,
                       target_steps=1000, output_dir=str(out), model="anima")
    run_manifest.write_start(str(run), rd)
    run_manifest.mark(str(run), status)
    state = run / f"{name}-state"
    state.mkdir()
    (state / "train_state.json").write_text(
        json.dumps({"current_epoch": epochs, "current_step": steps}), encoding="utf-8")
    (run / f"{name}-000012.safetensors").write_bytes(b"x")
    (run / "sample").mkdir()
    (run / "sample" / "a.png").write_bytes(b"x")
    if final:
        (run / f"{name}.safetensors").write_bytes(b"lora")
    return run


def _onetrainer_run(out, name, dataset, backups=((871, 36), (1368, 57)), sidecar=True):
    """A Krea 2 folder as runs left them before they carried a run.json."""
    run = out / name
    (run / "workspace").mkdir(parents=True)
    (run / "cache").mkdir()
    (run / "config.json").write_text("{}", encoding="utf-8")
    (run / "concepts.json").write_text(
        json.dumps([{"name": name, "path": str(dataset)}]), encoding="utf-8")
    (run / "trainer.log").write_text("log", encoding="utf-8")
    for i, (step, epoch) in enumerate(backups):
        b = run / "workspace" / "backup" / f"2026-10-14_22-{i:02d}-00-backup-{step}-{epoch}-0"
        b.mkdir(parents=True)
        (b / "meta.json").write_text(json.dumps(
            {"train_progress": {"epoch": epoch, "global_step": step}}), encoding="utf-8")
    (run / f"{name}.safetensors").write_bytes(b"lora")
    if sidecar:
        (run / f"{name}.trained-words.txt").write_text(
            f"Trained words — {name}\n\n    cat_x  —  the set's trigger word — in 3 of 3 "
            "captions\n", encoding="utf-8")
    return run


def test_lists_kohya_and_onetrainer_runs_with_their_steps(tmp_path):
    data = _dataset(tmp_path)
    out = tmp_path / "out"
    _kohya_run(out, "Joe", data)
    _onetrainer_run(out, "Cat", data)
    runs = {r.name: r for r in past_runs.list_runs(str(out), "C:/ot")}
    assert set(runs) == {"Joe", "Cat"}
    joe, cat = runs["Joe"], runs["Cat"]
    assert (joe.engine, joe.steps_done, joe.epochs_done, joe.status) == ("kohya", 1152, 12, "done")
    assert joe.can_continue and joe.has_lora
    assert Path(joe.resume_from).name == "Joe-state"
    assert (cat.engine, cat.steps_done, cat.epochs_done) == ("onetrainer", 1368, 57)
    assert Path(cat.resume_from).name.endswith("backup-1368-57-0")    # the newest backup
    assert cat.rd.model == "krea2" and cat.rd.dataset_folder == str(data)
    assert cat.rd.trigger_word == "cat_x" and cat.rd.image_count == 3
    assert cat.can_continue


def test_onetrainer_run_without_sidecar_is_called_by_its_name(tmp_path):
    data = _dataset(tmp_path)
    out = tmp_path / "out"
    _onetrainer_run(out, "Cat", data, sidecar=False)
    (run,) = past_runs.list_runs(str(out))
    assert run.rd.trigger_word == "Cat"


def test_a_run_that_cannot_continue_says_why(tmp_path):
    data = _dataset(tmp_path)
    out = tmp_path / "out"
    _onetrainer_run(out, "NoBackup", data, backups=())
    gone = _kohya_run(out, "Gone", tmp_path / "missing")
    runs = {r.name: r for r in past_runs.list_runs(str(out))}
    assert runs["NoBackup"].problem == "no OneTrainer backup was saved"
    assert runs["Gone"].problem == "dataset folder is gone"
    assert not runs["Gone"].can_continue and gone.is_dir()


def test_listing_ignores_folders_that_are_not_runs_and_falsy_paths(tmp_path):
    (tmp_path / "out" / "random").mkdir(parents=True)
    assert past_runs.list_runs(str(tmp_path / "out")) == []
    assert past_runs.list_runs("") == []
    assert past_runs.list_runs(str(tmp_path / "nope")) == []


def test_delete_keeps_only_the_finished_lora_and_sidecar(tmp_path):
    data = _dataset(tmp_path)
    out = tmp_path / "out"
    kohya = _kohya_run(out, "Joe", data)
    ot = _onetrainer_run(out, "Cat", data)
    for run in past_runs.list_runs(str(out)):
        past_runs.delete_training_files(run, str(out))
    assert sorted(p.name for p in kohya.iterdir()) == ["Joe.safetensors"]
    assert sorted(p.name for p in ot.iterdir()) == ["Cat.safetensors",
                                                    "Cat.trained-words.txt"]
    assert sorted(p.name for p in data.iterdir()) == [
        "img0.png", "img0.txt", "img1.png", "img1.txt", "img2.png", "img2.txt"]
    assert past_runs.list_runs(str(out)) == []          # nothing left to list


def test_delete_removes_the_folder_of_a_run_that_never_finished(tmp_path):
    data = _dataset(tmp_path)
    out = tmp_path / "out"
    run_dir = _kohya_run(out, "Half", data, status=run_manifest.INTERRUPTED, final=False)
    (run,) = past_runs.list_runs(str(out))
    past_runs.delete_training_files(run, str(out))
    assert not run_dir.exists()


def test_delete_refuses_a_folder_outside_the_output_folder(tmp_path):
    data = _dataset(tmp_path)
    out = tmp_path / "out"
    _kohya_run(out, "Joe", data)
    (run,) = past_runs.list_runs(str(out))
    run.run_dir = str(tmp_path)
    with pytest.raises(ValueError):
        past_runs.delete_training_files(run, str(out))
    assert data.is_dir()


# ---- OneTrainer: continuing from the newest backup ----

def test_continue_turns_on_continue_last_backup_and_keeps_the_old_log(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    run = tmp_path / "out" / "Aria"
    run.mkdir(parents=True)
    (run / "trainer.log").write_text("first stretch", encoding="utf-8")
    s = TrainingSession(process_factory=FakeProc)
    assert s.start(_params(tmp_path, target_steps=1000, continue_epoch=20)) is True
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert cfg["continue_last_backup"] is True
    assert cfg["epochs"] == 100                       # 20 images, batch 2 -> 10/epoch
    assert (run / "trainer.to-epoch-20.log").read_text(encoding="utf-8") == "first stretch"


def test_fresh_run_does_not_continue_a_backup(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    assert s.start(_params(tmp_path)) is True
    cfg = json.loads((Path(s.run_dir()) / "config.json").read_text(encoding="utf-8"))
    assert cfg["continue_last_backup"] is False


def test_continue_refuses_a_plan_no_longer_than_what_is_trained(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    msgs = []
    s.failed.connect(msgs.append)
    assert s.start(_params(tmp_path, target_steps=1000, continue_epoch=100)) is False
    assert "more than the 100 epochs" in msgs[0]


def test_continued_progress_reports_the_runs_real_epoch(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    seen = []
    s.progress.connect(seen.append)
    assert s.start(_params(tmp_path, target_steps=1000, continue_epoch=20)) is True
    FakeProc.last.progress.emit({"name": "epoch", "n": 3, "total": 80})
    FakeProc.last.progress.emit({"name": "step", "n": 4, "total": 10})
    last = seen[-1]
    assert (last["epoch"], last["epoch_total"]) == (23, 100)
    assert (last["global_step"], last["global_total"]) == (234, 1000)


def test_loss_checks_are_shifted_onto_the_real_epoch(tmp_path, monkeypatch):
    _quiet_preflight(monkeypatch, tmp_path)
    s = TrainingSession(process_factory=FakeProc)
    assert s.start(_params(tmp_path, target_steps=1000, continue_epoch=20)) is True
    found = s._absolute({"kind": "overshoot", "last_good_epoch": 5, "best_epoch": 5,
                         "final_epoch": 9, "total_epochs": 80})
    assert (found["last_good_epoch"], found["final_epoch"], found["total_epochs"]) == (25, 29, 100)


# ---- Kohya: the step bar of a resumed run ----

def test_resumed_kohya_bar_counts_from_the_steps_already_trained():
    t = TrainingProcess()
    t.set_step_offset(1152)
    t._handle_line("steps:  10%|█         | 48/1440 [00:10<01:00,  1.0it/s, loss=0.1]")
    assert t.current_step == 1200
    t._on_finished(0, None)
    t._handle_line("steps:  10%|█         | 48/1440 [00:10<01:00,  1.0it/s, loss=0.1]")
    assert t.current_step == 48                        # cleared when the run ended


# ---- the Train tab's Past runs list ----

@pytest.fixture
def no_dialogs(monkeypatch):
    from PySide6.QtWidgets import QInputDialog, QMessageBox
    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    monkeypatch.setattr(QInputDialog, "getInt", lambda *a, **k: (500, True))


def test_train_tab_lists_runs_and_trains_a_krea_run_further(tmp_path, monkeypatch, no_dialogs):
    from test_train_tab_onetrainer import _krea_tab
    t, s = _krea_tab(tmp_path, monkeypatch)
    data = _dataset(tmp_path, "cat_ds", images=20)
    _onetrainer_run(tmp_path / "out", "Cat", data)
    t.refresh_past_runs()
    assert [r.name for r in t._past_runs] == ["Cat"]
    assert not t._past_box.isHidden()
    t._train_further(t._past_runs[0])
    assert s.params.lora_name == "Cat" and s.params.continue_epoch == 57
    assert s.params.target_steps == 1368 + 500
    assert t._continue_epoch == 0                       # a later Start is a fresh run


def test_train_further_resumes_a_kohya_run_for_more_epochs(tmp_path, monkeypatch, no_dialogs):
    from ui.train_tab import TrainTab
    t = TrainTab()
    t._output_dir = str(tmp_path / "out")
    t.set_gpu_busy_check(lambda: "")
    monkeypatch.setattr(t, "_validate_for_training", lambda: (True, ""))
    monkeypatch.setattr(t, "_validate_for_config", lambda: (True, ""))
    from core import gpu_check, sets
    monkeypatch.setattr(gpu_check, "free_vram_mb", lambda: None)   # never probe the real GPU
    # Start writes the crash-recovery marker; keep it out of the real sets/ folder.
    monkeypatch.setattr(sets, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(t, "_check_undersized_images", lambda interactive=True: True)
    monkeypatch.setattr(t, "_confirm_preflight", lambda: True)      # its own modal box
    configs = []
    monkeypatch.setattr("ui.train_tab.generate_configs",
                        lambda **kw: configs.append(kw) or (str(tmp_path / "c.toml"),
                                                            str(tmp_path / "d.toml")))
    (tmp_path / "c.toml").write_text("", encoding="utf-8")
    started = []
    monkeypatch.setattr(t._trainer, "start", lambda *a, **k: started.append(a))
    data = _dataset(tmp_path)
    _kohya_run(tmp_path / "out", "Joe", data, steps=1152)
    t.refresh_past_runs()
    (run,) = t._past_runs
    t._train_further(run)
    assert started, "the resumed run was never launched"
    kw = configs[-1]
    assert kw["resume_state_path"] == run.resume_from
    per_epoch = t._training_params["steps_per_epoch"]
    assert kw["epochs"] == 1152 // per_epoch + -(-500 // per_epoch)
    assert t._trainer._step_offset == (1152 // per_epoch) * per_epoch


def test_delete_button_cleans_the_run_and_drops_its_row(tmp_path, monkeypatch, no_dialogs):
    from test_train_tab_onetrainer import _krea_tab
    t, _s = _krea_tab(tmp_path, monkeypatch)
    data = _dataset(tmp_path, "cat_ds")
    run_dir = _onetrainer_run(tmp_path / "out", "Cat", data)
    t.refresh_past_runs()
    t._delete_past_run(t._past_runs[0])
    assert sorted(p.name for p in run_dir.iterdir()) == ["Cat.safetensors",
                                                         "Cat.trained-words.txt"]
    assert t._past_runs == [] and t._past_box.isHidden()
