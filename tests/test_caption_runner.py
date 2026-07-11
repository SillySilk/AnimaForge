import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
import core.caption_runner as caption_runner_mod
from core.caption_runner import CaptionJob, CaptionRunner, plan_stages
from core.caption_policy import FolderCaptionState, KEEP, OVERWRITE
from core.caption_manifest import images_dict

_app = QApplication.instance() or QApplication([])


def _job(**kw):
    base = dict(dataset_folder="C:/d", sdscripts_path="C:/sd")
    base.update(kw)
    return CaptionJob(**base)


def _state(captioned=0, partial=0, untouched=0):
    return FolderCaptionState(
        total=captioned + partial + untouched,
        captioned=[f"c{i}.png" for i in range(captioned)],
        partial=[f"p{i}.png" for i in range(partial)],
        untouched=[f"u{i}.png" for i in range(untouched)],
        foreign=0)


def test_plan_stages_keep_skips_captioned_images():
    job = _job(policy=KEEP, chain=["tag", "describe", "combine"])
    stages = plan_stages(job, _state(captioned=47, untouched=33))
    assert [s for s, _ in stages] == ["tag", "describe", "combine"]
    for _stage, imgs in stages:
        assert len(imgs) == 33


def test_plan_stages_keep_on_fully_captioned_folder_is_empty():
    job = _job(policy=KEEP, chain=["tag", "describe", "combine"])
    assert plan_stages(job, _state(captioned=80)) == []


def test_plan_stages_overwrite_takes_everything():
    job = _job(policy=OVERWRITE, chain=["tag", "combine"])
    stages = plan_stages(job, _state(captioned=47, untouched=33))
    assert [len(i) for _s, i in stages] == [80, 80]


def test_plan_stages_honours_chain_order():
    job = _job(policy=OVERWRITE, chain=["tag", "describe", "combine"])
    got = [s for s, _ in plan_stages(job, _state(untouched=2))]
    assert got == ["tag", "describe", "combine"]


def test_plan_stages_raises_on_unknown_stage():
    import pytest
    job = _job(chain=["tag", "bogus", "combine"])
    with pytest.raises(ValueError, match="bogus"):
        plan_stages(job, _state(untouched=2))


def test_runner_start_refuses_without_sdscripts():
    r = CaptionRunner()
    assert r.start(_job(sdscripts_path="")) is False
    assert r.is_running() is False


# ----------------------------------------------------------------------------
# State-machine tests. These drive the real TaggerProcess/JoyCaptionProcess
# wrapper objects (their `finished` signals are wired to _step_done in
# CaptionRunner.__init__ and must stay wired), neutering only their `start`
# methods so no real subprocess is spawned. The chain is then advanced by
# emitting the wrappers' real `finished` signals, exactly as QProcess would.
# ----------------------------------------------------------------------------

def _png(folder: Path, name: str):
    (folder / name).write_bytes(b"\x89PNG\r\n\x1a\n")


def test_runner_full_chain_advance(tmp_path, monkeypatch):
    _png(tmp_path, "a.png")
    _png(tmp_path, "b.png")

    runner = CaptionRunner()
    started = []
    monkeypatch.setattr(runner._tagger, "start", lambda **kw: started.append("tag"))
    monkeypatch.setattr(runner._joy, "start", lambda **kw: started.append("describe"))
    monkeypatch.setattr(caption_runner_mod, "combine_all", lambda *a, **kw: (2, 0))

    stages_done = []
    runner.stage_done.connect(stages_done.append)
    finished = []
    runner.finished.connect(finished.append)

    job = _job(dataset_folder=str(tmp_path), chain=["tag", "describe", "combine"])
    assert runner.start(job) is True
    assert started == ["tag"]
    assert runner.is_running() is True

    runner._tagger.finished.emit(True)
    assert started == ["tag", "describe"]
    assert stages_done == ["tag"]
    assert runner.is_running() is True

    runner._joy.finished.emit(True)   # combine runs synchronously off this call
    assert stages_done == ["tag", "describe", "combine"]
    assert finished == [True]
    assert runner.is_running() is False

    images = images_dict(str(tmp_path))
    assert images is not None
    for name in ("a.png", "b.png"):
        assert images[name].get("tag") == "done"
        assert images[name].get("describe") == "done"
        assert images[name].get("combine") == "done"


def test_runner_failure_mid_chain_stops_and_skips_rest(tmp_path, monkeypatch):
    _png(tmp_path, "a.png")

    runner = CaptionRunner()
    started = []
    monkeypatch.setattr(runner._tagger, "start", lambda **kw: started.append("tag"))
    monkeypatch.setattr(runner._joy, "start", lambda **kw: started.append("describe"))
    finished = []
    runner.finished.connect(finished.append)

    job = _job(dataset_folder=str(tmp_path), chain=["tag", "describe", "combine"])
    runner.start(job)

    runner._tagger.finished.emit(False)
    assert finished == [False]
    assert started == ["tag"]          # describe never started
    assert runner.is_running() is False


def test_runner_combine_errors_fails_chain(tmp_path, monkeypatch):
    _png(tmp_path, "a.png")
    monkeypatch.setattr(caption_runner_mod, "combine_all", lambda *a, **kw: (0, 3))

    runner = CaptionRunner()
    finished = []
    runner.finished.connect(finished.append)

    job = _job(dataset_folder=str(tmp_path), chain=["combine"])
    runner.start(job)

    assert finished == [False]
    assert runner.is_running() is False


def test_runner_stop_emits_finished_false_exactly_once(tmp_path, monkeypatch):
    _png(tmp_path, "a.png")

    runner = CaptionRunner()
    monkeypatch.setattr(runner._tagger, "start", lambda **kw: None)
    finished = []
    runner.finished.connect(finished.append)

    job = _job(dataset_folder=str(tmp_path), chain=["tag", "combine"])
    runner.start(job)
    assert runner.is_running() is True

    runner.stop()
    assert finished == [False]
    assert runner.is_running() is False

    # Simulate the reentrant _step_done a real waitForFinished() would trigger
    # synchronously during terminate() -- it must no-op now that _running is
    # already False, not double-emit.
    runner._step_done("tag", False)
    assert finished == [False]

    runner.stop()   # a second stop() call is also a no-op
    assert finished == [False]


def test_only_file_removed_after_failed_stage_and_after_stop(tmp_path, monkeypatch):
    _png(tmp_path, "a.png")

    runner = CaptionRunner()
    monkeypatch.setattr(runner._tagger, "start", lambda **kw: None)
    job = _job(dataset_folder=str(tmp_path), chain=["tag", "combine"])
    runner.start(job)
    only_path = runner._only_file
    assert only_path and Path(only_path).exists()

    runner._tagger.finished.emit(False)
    assert not Path(only_path).exists()

    runner2 = CaptionRunner()
    monkeypatch.setattr(runner2._tagger, "start", lambda **kw: None)
    job2 = _job(dataset_folder=str(tmp_path), chain=["tag", "combine"])
    runner2.start(job2)
    only_path2 = runner2._only_file
    assert only_path2 and Path(only_path2).exists()

    runner2.stop()
    assert not Path(only_path2).exists()
