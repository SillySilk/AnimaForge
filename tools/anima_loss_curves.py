"""Print the per-epoch loss curve of every past Anima run, for calibrating run safety.

Usage: python tools/anima_loss_curves.py OUTPUT_DIR

Reads sd-scripts' TensorBoard per-epoch loss (`loss/epoch_average`) from OUTPUT_DIR/*/logs/**. For each
run prints the curve, the largest rise over the running best after settling
(max_rise, what would trip the blow-up guard) and the final rise over the best
(final_rise, what would trip the overshoot check).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.engines.onetrainer.verify import SETTLING_FRACTION  # noqa: E402


# sd-scripts has named the per-epoch mean both ways.
EPOCH_TAGS = ("loss/epoch_average", "loss/epoch")


def curve(run_dir: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out = {}
    for events in sorted(run_dir.glob("logs/**/events.out.tfevents.*")):
        acc = EventAccumulator(str(events), size_guidance={"scalars": 0})
        acc.Reload()
        tag = next((t for t in EPOCH_TAGS if t in acc.Tags().get("scalars", [])), None)
        if tag is None:
            continue
        for ev in acc.Scalars(tag):
            out[int(ev.step)] = float(ev.value)
    return out


def rises(losses: dict) -> tuple:
    epochs = sorted(losses)
    first = max(1, round(len(epochs) * SETTLING_FRACTION))
    best, max_rise = None, 0.0
    for e in epochs:
        if e < first:
            continue
        if best is not None and losses[best] > 0:
            max_rise = max(max_rise, losses[e] / losses[best] - 1.0)
        if best is None or losses[e] < losses[best]:
            best = e
    final_rise = losses[epochs[-1]] / losses[best] - 1.0 if best else 0.0
    return max_rise, final_rise


def main(output_dir: str) -> None:
    for run_dir in sorted(p for p in Path(output_dir).iterdir() if p.is_dir()):
        losses = curve(run_dir)
        if len(losses) < 2:
            continue
        max_rise, final_rise = rises(losses)
        shown = ", ".join(f"{e}:{v:.4f}" for e, v in sorted(losses.items()))
        print(f"{run_dir.name}: epochs={len(losses)} max_rise={max_rise:.3f} "
              f"final_rise={final_rise:.3f}\n  {{{shown}}}")


if __name__ == "__main__":
    main(sys.argv[1])
