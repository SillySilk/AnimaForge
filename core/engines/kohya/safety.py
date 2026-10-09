"""Run safety for Anima (Kohya sd-scripts) runs.

The same three guards a Krea 2 run has (core/engines/onetrainer/verify.py), read
from sd-scripts' own output: a loss that blows up mid-run stops the run and ships
the last good epoch save; a run that ends well above its best ships the best
epoch save; a final file that never learned is refused.

sd-scripts prints `epoch N/M` as each epoch opens and carries `avr_loss=` on its
step bar. avr_loss is the mean over the last epoch's worth of steps, so the value
standing when the next epoch opens is the loss that closed the previous one.

Anima's per-epoch loss is much noisier than Krea 2's smooth loss: healthy past
runs wander up to +26% over their running best and finish up to +23% above it
(tools/anima_loss_curves.py over the maintainer's runs, 2026-10-09). The
thresholds sit at twice / one and a half times those, so a healthy run is never
stopped and only a real blow-up is.
"""
import os
import re
from pathlib import Path

from core.engines.onetrainer import snapshots, verify

ANIMA_DIVERGENCE_RISE = 0.55
ANIMA_OVERSHOOT_RISE = 0.35
SETTLE = verify.SETTLING_FRACTION

# Kohya's LoRA keys: lora_unet_blocks_0_..., lora_up / lora_down.
NAMESPACE = "lora_unet"

_EPOCH_RE = re.compile(r"^\s*epoch (\d+)/(\d+)\s*$")
_AVR_LOSS_RE = re.compile(r"avr_loss[=:]\s*([0-9.eE+-]+)")


def _save_re(output_name: str):
    return re.compile(rf"^{re.escape(output_name)}-(\d{{6}})\.safetensors$")


def loss_by_epoch(lines, finished: bool = False) -> dict:
    """{epoch: loss that closed it}. The last epoch only counts once the run ended."""
    out, current, epoch = {}, None, 0
    for line in lines:
        m = _EPOCH_RE.match(line)
        if m:
            if epoch and current is not None:
                out[epoch] = current
            epoch = int(m.group(1))
            continue
        m = _AVR_LOSS_RE.search(line)
        if m:
            try:
                current = float(m.group(1))
            except ValueError:
                pass
    if finished and epoch and current is not None:
        out[epoch] = current
    return out


def verdict(losses: dict, total: int):
    """A blow-up anywhere in the curve, else an overshoot at its end, else None."""
    if not losses:
        return None
    return (verify.divergence(losses, total, rise=ANIMA_DIVERGENCE_RISE, settle=SETTLE)
            or verify.overshoot(losses, total, rise=ANIMA_OVERSHOOT_RISE, settle=SETTLE))


class LossWatch:
    """Fed sd-scripts' output one line at a time; feed() returns the blow-up the
    moment the epoch that carries it closes, once."""

    def __init__(self):
        self.losses = {}
        self.total = 0
        self._epoch = 0
        self._current = None
        self._found = None

    def feed(self, line: str):
        m = _EPOCH_RE.match(line)
        if m:
            if self._epoch and self._current is not None:
                self.losses[self._epoch] = self._current
            self._epoch, self.total = int(m.group(1)), int(m.group(2))
            if self._found is None and self.losses:
                found = verify.divergence(self.losses, self.total,
                                          rise=ANIMA_DIVERGENCE_RISE, settle=SETTLE)
                if found:
                    self._found = found
                    return found
            return None
        m = _AVR_LOSS_RE.search(line)
        if m:
            try:
                self._current = float(m.group(1))
            except ValueError:
                pass
        return None

    def final_losses(self) -> dict:
        """The curve including the last epoch, for a run that has ended."""
        out = dict(self.losses)
        if self._epoch and self._current is not None:
            out[self._epoch] = self._current
        return out


def epoch_saves(run_dir: str, output_name: str) -> list:
    """sd-scripts' epoch saves ({name}-{NNNNNN}.safetensors), oldest first."""
    folder = Path(run_dir or "")
    if not folder.is_dir():
        return []
    pattern = _save_re(output_name)
    found = []
    for path in folder.iterdir():
        m = pattern.match(path.name)
        if m and path.is_file():
            found.append(snapshots.Snapshot(str(path), 0, int(m.group(1))))
    return sorted(found, key=lambda s: s.epoch)


def final_path(run_dir: str, output_name: str) -> str:
    return str(Path(run_dir) / f"{output_name}.safetensors")


def final_problem(run_dir: str, output_name: str):
    """Why the run's final file is not a usable LoRA, or None."""
    return verify.verify_output(final_path(run_dir, output_name), namespace=NAMESPACE)


def ship(run_dir: str, output_name: str, found: dict) -> tuple:
    """Ship the newest healthy epoch save from at or before the last good epoch.
    Returns (shipped_epoch | None, sentence for the user)."""
    said = verify.describe(found)
    for snap in snapshots.candidates(epoch_saves(run_dir, output_name),
                                     found["last_good_epoch"]):
        if verify.verify_output(snap.path, namespace=NAMESPACE) is not None:
            continue
        kept = snapshots.promote(snap.path, final_path(run_dir, output_name))
        return snap.epoch, (f"{said} Shipped the epoch {snap.epoch} snapshot instead; "
                            f"the last file is kept as {Path(kept).name}.")
    return None, (f"{said} No usable snapshot from before that point was saved, "
                  "so the final file was kept.")


def prune(run_dir: str, output_name: str, keep_last: int, keep_epoch=None) -> int:
    """Apply the user's keep-last to the epoch saves (sd-scripts keeps them all
    during the run so safety can reach the best one). Never removes the epoch that
    shipped or the final file. Returns how many went."""
    if not keep_last or keep_last <= 0:
        return 0
    saves = epoch_saves(run_dir, output_name)
    doomed = [s for s in saves[:-keep_last] if s.epoch != keep_epoch]
    removed = 0
    for snap in doomed:
        try:
            os.remove(snap.path)
            removed += 1
        except OSError:
            continue
    return removed
