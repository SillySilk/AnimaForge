"""The snapshots a run leaves behind, and which one ships when its end went bad.

The preset has OneTrainer write a snapshot of the LoRA every five epochs
(`save_every`), in the same format as the final file, into workspace/save. They
exist for one reason: a run can train cleanly for fifty epochs and then overshoot
in its last fifteen, and the final file is the overshot state. Hailey
(2026-09-04) did exactly that -- its loss bottomed at epoch 52 of 67, finished
38% higher, and the shipped file rendered a mosaic; the epoch-50 snapshot was the
LoRA that worked. When core.engines.onetrainer.verify says the curve turned, the session
ships the last snapshot from at or before the best epoch and keeps the overshot
file beside it. When the run was healthy, the snapshots are deleted: at ~114 MB
each they are not worth keeping once the question they answer is settled.

Stdlib only, like the rest of the trainer package.
"""
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

# OneTrainer names a save <prefix><timestamp>-save-<global_step>-<epoch>-<epoch_step>
# (TrainProgress.filename_string). `epoch` is how many epochs were complete.
_SAVE_RE = re.compile(r"-save-(\d+)-(\d+)-(\d+)\.safetensors$")

# The overshot end is kept under this name beside the file that ships. It is an
# hour of GPU time and the evidence of what went wrong; it is never deleted here.
OVERSHOT_SUFFIX = ".overshot.safetensors"


@dataclass(frozen=True)
class Snapshot:
    path: str
    step: int
    epoch: int


def find(save_dir: str) -> list:
    """Every snapshot in `save_dir`, oldest first. [] when there is no folder."""
    folder = Path(save_dir or "")
    if not folder.is_dir():
        return []
    found = []
    for path in folder.iterdir():
        m = _SAVE_RE.search(path.name)
        if m and path.is_file():
            found.append(Snapshot(str(path), int(m.group(1)), int(m.group(2))))
    return sorted(found, key=lambda s: (s.epoch, s.step))


def candidates(found: list, best_epoch: int) -> list:
    """The snapshots worth shipping for a run whose loss was best at `best_epoch`:
    those from at or before it, newest first. One from after the turn would ship
    a milder version of the same overshoot, so it is never a candidate."""
    return sorted((s for s in found if 0 < s.epoch <= int(best_epoch)),
                  key=lambda s: (s.epoch, s.step), reverse=True)


def pick(found: list, best_epoch: int):
    """The snapshot to ship, or None when nothing from before the turn exists."""
    ranked = candidates(found, best_epoch)
    return ranked[0] if ranked else None


def promote(snapshot_path: str, output_path: str) -> str:
    """Ship the snapshot under the run's output name; keep the overshot file
    beside it. Returns the path the overshot file was kept under.

    Two renames, no copies: the overshot file steps aside first, so a failure in
    the second move leaves both files on disk rather than neither.
    """
    out = Path(output_path)
    kept = out.with_name(out.stem + OVERSHOT_SUFFIX)
    if out.exists():
        os.replace(out, kept)
    shutil.move(snapshot_path, str(out))
    return str(kept)


def discard(save_dir: str) -> int:
    """Delete every snapshot in `save_dir`. Returns how many went."""
    removed = 0
    for snap in find(save_dir):
        try:
            os.remove(snap.path)
            removed += 1
        except OSError:
            continue
    return removed
