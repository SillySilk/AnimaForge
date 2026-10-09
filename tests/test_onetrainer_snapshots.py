"""The snapshots OneTrainer writes every few epochs (preset `save_every`), and
how the session picks one to ship when the end of the run overshot.

Hailey (2026-09-04): the loss bottomed at epoch 52 of 67 and the shipped file
was the overshot state. A snapshot from epoch 50 was the file that worked.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.engines.onetrainer import snapshots
from lora_fixture import write_lora


def _save(dir_: Path, step: int, epoch: int, stamp="2026-09-04_22-51-27"):
    dir_.mkdir(parents=True, exist_ok=True)
    return write_lora(dir_ / f"{stamp}-save-{step}-{epoch}-0.safetensors")


def test_snapshots_are_read_from_onetrainer_save_names(tmp_path):
    """OneTrainer names a save <timestamp>-save-<step>-<epoch>-<epoch_step>; the
    epoch is how many epochs were complete when it was written."""
    save = tmp_path / "save"
    _save(save, 850, 50)
    _save(save, 935, 55, stamp="2026-09-04_22-56-00")
    (save / "notes.txt").write_text("not a snapshot", encoding="utf-8")
    found = snapshots.find(str(save))
    assert [(s.epoch, s.step) for s in found] == [(50, 850), (55, 935)]
    assert all(Path(s.path).is_file() for s in found)


def test_a_missing_save_folder_has_no_snapshots(tmp_path):
    assert snapshots.find(str(tmp_path / "nope")) == []


def test_the_pick_is_the_last_snapshot_at_or_before_the_best_epoch(tmp_path):
    save = tmp_path / "save"
    for epoch in (45, 50, 55, 60):
        _save(save, epoch * 17, epoch)
    found = snapshots.find(str(save))
    assert snapshots.pick(found, best_epoch=52).epoch == 50
    assert snapshots.pick(found, best_epoch=45).epoch == 45
    assert snapshots.pick(found, best_epoch=60).epoch == 60
    assert snapshots.pick(found, best_epoch=3) is None


def test_promote_ships_the_snapshot_and_keeps_the_overshot_file_beside_it(tmp_path):
    save = tmp_path / "save"
    snap = _save(save, 850, 50)
    out = write_lora(tmp_path / "Hailey.safetensors", b_value=9.0e-2)   # the overshot end
    snap_bytes = snap.read_bytes()
    kept = snapshots.promote(str(snap), str(out))
    assert out.read_bytes() == snap_bytes                  # the snapshot is what ships
    assert Path(kept).name == "Hailey.overshot.safetensors"
    assert Path(kept).is_file()
    assert not snap.exists()


def test_discard_removes_every_snapshot_and_says_how_many(tmp_path):
    save = tmp_path / "save"
    for epoch in (5, 10, 15):
        _save(save, epoch * 17, epoch)
    assert snapshots.discard(str(save)) == 3
    assert not any(save.glob("*.safetensors"))
    assert snapshots.discard(str(save)) == 0
    assert snapshots.discard(str(tmp_path / "nope")) == 0
