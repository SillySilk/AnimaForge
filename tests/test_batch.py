import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from core.batch import RunDefinition, save_queue, load_queue, QUEUED

_app = QApplication.instance() or QApplication([])


def _rd(**kw):
    base = dict(lora_name="lr", dataset_folder="C:/d", image_count=20)
    base.update(kw)
    return RunDefinition(**base)


def _png(folder: Path, name: str):
    (folder / name).write_bytes(b"\x89PNG\r\n\x1a\n")


def _neuter_caption_processes(runner, monkeypatch):
    """Silence the captioner's real subprocess wrappers so no process is spawned;
    its `finished` signals stay wired and the chain is driven by emitting them."""
    monkeypatch.setattr(runner._captioner._tagger, "start", lambda **kw: None)
    monkeypatch.setattr(runner._captioner._joy, "start", lambda **kw: None)
    monkeypatch.setattr(runner._captioner._llm, "start", lambda **kw: None)


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


# ----------------------------------------------------------------------------
# Pure helpers: next_index / reset_statuses / restart
# ----------------------------------------------------------------------------

def test_next_index_skips_done_runs():
    from core.batch import next_index, DONE, QUEUED
    runs = [_rd(status=DONE), _rd(status=DONE), _rd(status=QUEUED)]
    assert next_index(runs, 0, skip_done=True) == 2
    assert next_index(runs, 0, skip_done=False) == 0


def test_next_index_returns_len_when_exhausted():
    from core.batch import next_index, DONE
    runs = [_rd(status=DONE)]
    assert next_index(runs, 0, skip_done=True) == 1


def test_restart_resets_every_status_to_queued():
    from core.batch import BatchRunner, DONE, FAILED, QUEUED
    runs = [_rd(status=DONE), _rd(status=FAILED)]
    r = BatchRunner()
    r.reset_statuses(runs)
    assert [x.status for x in runs] == [QUEUED, QUEUED]


# ----------------------------------------------------------------------------
# Caption phase state machine. Neuters the captioner's subprocess wrappers and
# the trainer's start, then advances by emitting the real `finished` signals —
# exactly how tests/test_caption_runner.py drives the caption chain.
# ----------------------------------------------------------------------------

def _caption_run(tmp_path, name, captioned=False, **kw):
    folder = tmp_path / name
    folder.mkdir()
    _png(folder, "a.png")
    if captioned:
        (folder / "a.txt").write_text("a cat on a mat", encoding="utf-8")
    base = dict(lora_name=name, dataset_folder=str(folder), sdscripts_path="C:/sd",
                enable_bucket=False, caption_policy="keep",
                output_dir=str(tmp_path / "out"), image_count=4, target_steps=100)
    base.update(kw)
    return RunDefinition(**base)


def test_uncaptioned_run_emits_captioning_then_training(tmp_path, monkeypatch):
    from core.batch import BatchRunner, CAPTIONING, TRAINING, DONE
    import core.caption_runner as cr_mod

    runner = BatchRunner()
    _neuter_caption_processes(runner, monkeypatch)
    monkeypatch.setattr(cr_mod, "combine_all", lambda *a, **kw: (1, 0))
    trained = []
    monkeypatch.setattr(runner._trainer, "start", lambda cfg, sd: trained.append(sd))

    phases = []
    runner.run_phase.connect(lambda i, p: phases.append((i, p)))

    run = _caption_run(tmp_path, "ds")
    runner.start([run])
    assert phases == [(0, CAPTIONING)]           # captioning first, no training yet
    assert runner._captioner.is_running() is True

    runner._captioner._tagger.finished.emit(True)
    runner._captioner._joy.finished.emit(True)   # combine runs synchronously here
    assert phases == [(0, CAPTIONING), (0, TRAINING)]
    assert trained == ["C:/sd"]                  # training only started after captions

    runner._trainer.training_finished.emit(True)
    assert run.status == DONE


def test_fully_captioned_keep_skips_straight_to_training(tmp_path, monkeypatch):
    from core.batch import BatchRunner, TRAINING

    runner = BatchRunner()
    trained = []
    monkeypatch.setattr(runner._trainer, "start", lambda cfg, sd: trained.append(sd))
    phases = []
    runner.run_phase.connect(lambda i, p: phases.append((i, p)))

    run = _caption_run(tmp_path, "ds", captioned=True)
    runner.start([run])
    assert phases == [(0, TRAINING)]             # no captioning phase at all
    assert trained == ["C:/sd"]
    assert runner._captioner.is_running() is False


def test_caption_failure_fails_run_and_continues(tmp_path, monkeypatch):
    from core.batch import BatchRunner, FAILED, RUNNING, QUEUED

    runner = BatchRunner()
    _neuter_caption_processes(runner, monkeypatch)
    trained = []
    monkeypatch.setattr(runner._trainer, "start", lambda cfg, sd: trained.append(sd))
    finished_events = []
    runner.run_finished.connect(lambda i, ok: finished_events.append((i, ok)))

    run0 = _caption_run(tmp_path, "ds0")                    # needs captioning
    run1 = _caption_run(tmp_path, "ds1", captioned=True)    # trains straight away
    runner.start([run0, run1], continue_on_error=True)

    runner._captioner._tagger.finished.emit(False)         # tag stage fails
    assert run0.status == FAILED
    assert (0, False) in finished_events
    assert trained == ["C:/sd"]                            # advanced to run1
    assert run1.status == RUNNING


def test_caption_failure_stops_queue_when_not_continuing(tmp_path, monkeypatch):
    from core.batch import BatchRunner, FAILED, QUEUED

    runner = BatchRunner()
    _neuter_caption_processes(runner, monkeypatch)
    trained = []
    monkeypatch.setattr(runner._trainer, "start", lambda cfg, sd: trained.append(sd))

    run0 = _caption_run(tmp_path, "ds0")
    run1 = _caption_run(tmp_path, "ds1", captioned=True)
    runner.start([run0, run1], continue_on_error=False)

    runner._captioner._tagger.finished.emit(False)
    assert run0.status == FAILED
    assert run1.status == QUEUED       # never reached
    assert trained == []
    assert runner.is_running() is False


def test_stop_during_captioning_does_not_advance(tmp_path, monkeypatch):
    from core.batch import BatchRunner

    runner = BatchRunner()
    _neuter_caption_processes(runner, monkeypatch)
    trained = []
    monkeypatch.setattr(runner._trainer, "start", lambda cfg, sd: trained.append(sd))
    run_starts = []
    runner.run_started.connect(run_starts.append)

    run0 = _caption_run(tmp_path, "ds0")
    run1 = _caption_run(tmp_path, "ds1")
    runner.start([run0, run1])
    assert run_starts == [0]
    assert runner._captioner.is_running() is True

    runner.stop()   # CaptionRunner.stop() re-enters _on_caption_finished(False)

    assert runner.is_running() is False
    assert run_starts == [0]            # queue never advanced to run1
    assert trained == []               # no training ever started
    assert run1.status == "queued"


def test_start_skips_done_runs_by_default(tmp_path, monkeypatch):
    from core.batch import BatchRunner, DONE, RUNNING

    runner = BatchRunner()
    monkeypatch.setattr(runner._trainer, "start", lambda cfg, sd: None)
    started = []
    runner.run_started.connect(started.append)

    done = _caption_run(tmp_path, "ds0", captioned=True, status=DONE)
    todo = _caption_run(tmp_path, "ds1", captioned=True)
    runner.start([done, todo])                  # skip_done defaults True
    assert started == [1]                        # index 0 (DONE) is skipped
    assert done.status == DONE
    assert todo.status == RUNNING


def test_restart_reruns_finished_queue_from_the_top(tmp_path, monkeypatch):
    from core.batch import BatchRunner, DONE, RUNNING, QUEUED

    runner = BatchRunner()
    monkeypatch.setattr(runner._trainer, "start", lambda cfg, sd: None)
    started = []
    runner.run_started.connect(started.append)

    r0 = _caption_run(tmp_path, "ds0", captioned=True, status=DONE)
    r1 = _caption_run(tmp_path, "ds1", captioned=True, status=DONE)
    runner.restart([r0, r1])                     # resets all -> QUEUED, skip_done off
    assert started == [0]                        # starts from the very top again
    assert r0.status == RUNNING
    assert r1.status == QUEUED


# ----------------------------------------------------------------------------
# Task 11 fix pass: _advance must never recurse on a synchronous run failure.
# ----------------------------------------------------------------------------

def test_500_synchronous_caption_refusals_do_not_recurse(tmp_path):
    """A queue where every run's captioning refuses synchronously (blank
    sdscripts_path) must not recurse through _advance -> _begin_run -> _mark_failed
    -> _advance. Before the fix this raised RecursionError around run 330."""
    from core.batch import BatchRunner, FAILED

    runner = BatchRunner()
    runs = [_caption_run(tmp_path, f"ds{i}", sdscripts_path="") for i in range(500)]
    finished = []
    runner.batch_finished.connect(lambda: finished.append(True))

    runner.start(runs, continue_on_error=True)

    assert all(r.status == FAILED for r in runs)
    assert finished == [True]
    assert runner.is_running() is False


def test_config_generation_failure_honors_continue_on_error_false(tmp_path, monkeypatch):
    """Finding 2: config-generation failure must respect continue_on_error like
    every other failure path, instead of always marching on."""
    from core.batch import BatchRunner, FAILED, QUEUED
    import core.batch as batch_mod

    def _boom(**kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(batch_mod, "generate_configs", _boom)

    runner = BatchRunner()
    run0 = _caption_run(tmp_path, "ds0", captioned=True)   # already captioned -> straight to training
    run1 = _caption_run(tmp_path, "ds1", captioned=True)
    runner.start([run0, run1], continue_on_error=False)

    assert run0.status == FAILED
    assert run1.status == QUEUED       # never reached
    assert runner.is_running() is False


def test_config_generation_failure_continues_when_continue_on_error_true(tmp_path, monkeypatch):
    from core.batch import BatchRunner, FAILED, RUNNING
    import core.batch as batch_mod

    def _boom_for_ds0(**kw):
        if kw.get("lora_name") == "ds0":
            raise RuntimeError("boom")
        return ("cfg", None)
    monkeypatch.setattr(batch_mod, "generate_configs", _boom_for_ds0)

    runner = BatchRunner()
    trained = []
    monkeypatch.setattr(runner._trainer, "start", lambda cfg, sd: trained.append(sd))
    run0 = _caption_run(tmp_path, "ds0", captioned=True)
    run1 = _caption_run(tmp_path, "ds1", captioned=True)
    runner.start([run0, run1], continue_on_error=True)

    assert run0.status == FAILED
    assert run1.status == RUNNING
    assert trained == ["C:/sd"]


def test_scan_oserror_fails_only_that_run_and_queue_continues(tmp_path, monkeypatch):
    """Finding 3: cp.scan() walks the dataset folder with iterdir()/is_file(), which
    can raise OSError if the folder vanishes (deleted, or a network share drops)
    mid-batch. That must fail only the one run, not the whole queue."""
    from core.batch import BatchRunner, FAILED, RUNNING
    from core import caption_policy as cp

    runner = BatchRunner()
    trained = []
    monkeypatch.setattr(runner._trainer, "start", lambda cfg, sd: trained.append(sd))

    run0 = _caption_run(tmp_path, "ds0")                    # needs captioning
    run1 = _caption_run(tmp_path, "ds1", captioned=True)    # trains straight away

    real_scan = cp.scan

    def _scan(folder):
        if folder == run0.dataset_folder:
            raise OSError("dataset folder vanished mid-batch")
        return real_scan(folder)
    monkeypatch.setattr(cp, "scan", _scan)

    runner.start([run0, run1], continue_on_error=True)

    assert run0.status == FAILED
    assert run1.status == RUNNING
    assert trained == ["C:/sd"]


def test_unknown_caption_stage_fails_only_that_run_and_queue_continues(tmp_path, monkeypatch):
    """Finding 4: no BatchRunner-level test previously exercised the ValueError
    branch (unknown chain stage) of _begin_run."""
    from core.batch import BatchRunner, FAILED, RUNNING
    from core.caption_runner import CaptionJob

    runner = BatchRunner()
    trained = []
    monkeypatch.setattr(runner._trainer, "start", lambda cfg, sd: trained.append(sd))

    run0 = _caption_run(tmp_path, "ds0")
    run1 = _caption_run(tmp_path, "ds1", captioned=True)
    monkeypatch.setattr(
        run0, "to_caption_job",
        lambda sdscripts_path, characters_file, policy: CaptionJob(
            dataset_folder=run0.dataset_folder, sdscripts_path=sdscripts_path,
            chain=["not_a_real_stage"]))

    runner.start([run0, run1], continue_on_error=True)

    assert run0.status == FAILED
    assert run1.status == RUNNING
    assert trained == ["C:/sd"]


def test_caption_refusal_fails_only_that_run_and_queue_continues(tmp_path, monkeypatch):
    """Finding 4: no BatchRunner-level test previously exercised the `if not
    started` refusal branch of _begin_run (blank sdscripts_path)."""
    from core.batch import BatchRunner, FAILED, RUNNING

    runner = BatchRunner()
    trained = []
    monkeypatch.setattr(runner._trainer, "start", lambda cfg, sd: trained.append(sd))

    run0 = _caption_run(tmp_path, "ds0", sdscripts_path="")   # blank -> CaptionRunner.start() refuses
    run1 = _caption_run(tmp_path, "ds1", captioned=True)

    runner.start([run0, run1], continue_on_error=True)

    assert run0.status == FAILED
    assert run1.status == RUNNING
    assert trained == ["C:/sd"]


def test_on_caption_finished_guard_blocks_stray_true_after_stop(tmp_path, monkeypatch):
    """Finding 4 (mutation result): deleting `if not self._running: return` from
    _on_caption_finished isn't caught by test_stop_during_captioning_does_not_advance
    (that one is protected by _advance()'s own guard on the ok=False path). But a
    stray finished(True) landing after stop() calls _start_training directly —
    _advance() is never in the way — so without the guard this would resurrect a
    batch the user already stopped and launch training. This test makes the guard
    load-bearing."""
    from core.batch import BatchRunner, RUNNING

    runner = BatchRunner()
    _neuter_caption_processes(runner, monkeypatch)
    trained = []
    monkeypatch.setattr(runner._trainer, "start", lambda cfg, sd: trained.append(sd))

    run0 = _caption_run(tmp_path, "ds0")
    runner.start([run0])
    assert run0.status == RUNNING

    runner.stop()
    assert runner.is_running() is False

    # Simulate a stray finished(True) slipping in after stop() (e.g. a signal
    # already in flight before the captioner was torn down).
    runner._on_caption_finished(True)

    assert trained == []
    assert runner.is_running() is False
