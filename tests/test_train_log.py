import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from ui.train_tab import phase_for_line, LogDenoiser


def test_phase_caching():
    assert phase_for_line("2026-06-18 INFO caching latents to disk...") == "Caching latents…"


def test_phase_warming_up_english():
    assert phase_for_line("running training / start") == "Warming up…"


def test_phase_warming_up_japanese():
    assert phase_for_line("running training / 学習開始") == "Warming up…"


def test_phase_none_for_ordinary_line():
    assert phase_for_line("steps:  1%| | 14/1352 [00:30<35:00]") is None


def test_denoiser_drops_bare_continuation():
    d = LogDenoiser()
    assert d.filter("   current_epoch: 0, epoch: 1") is None


def test_denoiser_dedupes_epoch_increment_per_epoch():
    d = LogDenoiser()
    line0 = "2026 INFO epoch is incremented. current_epoch: 0, epoch: 1   train_util.py:784"
    assert d.filter(line0) == line0          # first epoch-0 line shows
    assert d.filter(line0) is None           # duplicate epoch-0 line dropped
    line1 = "2026 INFO epoch is incremented. current_epoch: 1, epoch: 2   train_util.py:784"
    assert d.filter(line1) == line1          # new epoch value shows again


def test_denoiser_passes_ordinary_lines():
    d = LogDenoiser()
    assert d.filter("prepare optimizer, data loader etc.") == "prepare optimizer, data loader etc."
