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


def test_onetrainer_samples_group_by_epoch_and_find_their_prompt(tmp_path):
    import json
    from core.samples import group_by_round, prompt_for_file
    run = tmp_path / "Run"
    (run / "workspace" / "samples" / "0 - Ann reads").mkdir(parents=True)
    (run / "workspace" / "samples" / "1 - Bob waters").mkdir(parents=True)
    (run / "samples.json").write_text(json.dumps(
        [{"prompt": "Ann reads"}, {"prompt": "Bob waters"}]), encoding="utf-8")
    files = []
    for d, step, ep in (("0 - Ann reads", 0, 0), ("0 - Ann reads", 326, 2),
                        ("1 - Bob waters", 326, 2)):
        f = run / "workspace" / "samples" / d / f"2026-10-12_11-44-30-training-sample-{step}-{ep}-0.jpg"
        f.write_bytes(b"x")
        files.append(str(f))
    labels = [label for label, _ in group_by_round(files)]
    assert labels == ["epoch 2", "start"]
    assert prompt_for_file(files[2], "unused.txt") == "Bob waters"
