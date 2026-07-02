import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.samples import group_by_round, round_key


def test_round_key_epoch_step_and_start():
    assert round_key("lora_e000003_00_20260702.png") == ((1, 3), "epoch 3")
    assert round_key("lora_000250_01_20260702.png") == ((0, 250), "step 250")
    assert round_key("lora_000000_00_20260702.png") == ((0, 0), "start")
    assert round_key("random.png") is None


def test_group_by_round_newest_first_prompt_order():
    files = [
        "/s/lora_e000001_01_x.png",
        "/s/lora_e000002_00_x.png",
        "/s/lora_e000001_00_x.png",
        "/s/lora_000000_00_x.png",
        "/s/stray.png",
    ]
    groups = group_by_round(files)
    labels = [g[0] for g in groups]
    assert labels == ["epoch 2", "epoch 1", "start", "other"]
    # within a round: prompt-index (name) order
    assert [Path(f).name for f in groups[1][1]] == [
        "lora_e000001_00_x.png", "lora_e000001_01_x.png"]


def test_group_by_round_empty():
    assert group_by_round([]) == []
