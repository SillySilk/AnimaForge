import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.train_metrics import parse_tqdm, format_eta


def test_full_line():
    line = "steps:  42%|####      | 672/1600 [05:23<08:37,  1.79it/s, avr_loss=0.0834]"
    m = parse_tqdm(line)
    assert m["step"] == 672 and m["total"] == 1600
    assert m["elapsed"] == 5 * 60 + 23
    assert m["eta"] == 8 * 60 + 37
    assert m["it_s"] == 1.79
    assert m["loss"] == 0.0834


def test_partial_line_without_loss():
    m = parse_tqdm("steps:   1%|          | 8/1600 [00:04<14:12,  1.87it/s]")
    assert m["step"] == 8 and m["total"] == 1600
    assert m["it_s"] == 1.87
    assert "loss" not in m


def test_seconds_per_iter_inverted():
    m = parse_tqdm("steps:   0%|          | 2/1600 [00:05<1:20:00,  2.50s/it]")
    assert m["it_s"] == 0.4              # 1 / 2.5
    assert m["eta"] == 80 * 60           # 1:20:00


def test_non_progress_line_is_empty():
    assert parse_tqdm("INFO: caching latents") == {}
    assert parse_tqdm("") == {}


def test_format_eta():
    assert format_eta(517) == "8m 37s"
    assert format_eta(3720) == "1h 02m"
    assert format_eta(42) == "42s"
