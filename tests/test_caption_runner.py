import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from core.caption_runner import CaptionJob, CaptionRunner, plan_stages
from core.caption_policy import FolderCaptionState, KEEP, OVERWRITE

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


def test_plan_stages_honours_chain_order_and_refine():
    job = _job(policy=OVERWRITE, chain=["tag", "describe", "refine", "combine"])
    got = [s for s, _ in plan_stages(job, _state(untouched=2))]
    assert got == ["tag", "describe", "refine", "combine"]


def test_runner_start_refuses_without_sdscripts():
    r = CaptionRunner()
    assert r.start(_job(sdscripts_path="")) is False
    assert r.is_running() is False
