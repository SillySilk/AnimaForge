"""Post-run check: is the .safetensors we just spent an hour on a LoRA the image
tool will actually apply?

Two things can go wrong in a way nothing else notices. Both happened on
run_B (2026-08-17), which trained all 1254 steps, saved cleanly, reported
success, and produced a file that changed nothing in the image:

  1. It never learned. `lora_B` is zero-initialised, so its magnitude IS the
     training signal -- a working Krea 2 LoRA carries a peak around 4e-02, that
     run shipped 1.7e-08. `lora_A` keeps its random init either way and so tells
     you nothing; judging by `lora_A` is how this got missed.
  2. It cannot load. The image tool matches `diffusion_model.`-prefixed native
     names; OneTrainer's DIFFUSERS_LORA writes the diffusers namespace instead
     (`transformer.transformer_blocks...`). The two share zero keys, so the
     loader applies nothing and reports nothing.

Stdlib only, on purpose: requirements.txt is PySide6/Pillow/toml, so this cannot
assume numpy or safetensors are installed in the app venv. Parsing the
safetensors header is 8 bytes of length plus JSON, and float magnitude ordering
is monotonic in the sign-cleared bit pattern, so the peak needs no float decoding
until the very end. Cost is ~2 s on a 114 MB file -- once, at the end of a run.
"""
import array
import json
import re
import struct
import sys
from pathlib import Path

# A working Krea 2 LoRA peaks around 4e-02 on the B side; a run that never learned
# sits near 1e-08. Anything under this is not a LoRA worth shipping.
MIN_TRAINED_ABSMAX = 1e-3

# The top-level key namespace the image tool matches on (OneTrainer's COMFY_LORA).
READABLE_PREFIX = "diffusion_model"

# A run that overshot: its end-of-epoch smooth loss finished this far above the
# best it reached. run_H (2026-09-04) bottomed at epoch 52 of 67 and finished
# +38%; the 13 healthy runs before it all finished within +0..8% of their best.
OVERSHOOT_RISE = 0.15

# A run that blew up: an end-of-epoch smooth loss this far above the best before
# it. Not a slow drift but a cliff -- run_S (2026-10-08) went
# from 0.129 to 0.215 (+67%) in one epoch and its samples turned to noise; the
# biggest epoch-to-epoch wobble on a run that stayed healthy was +12%. Checked
# live, at every epoch, so the run is stopped there instead of training another
# two hours of noise -- and once it has blown up the loss can drift back down
# (that run finished only +12% over its best) while the LoRA stays broken, which
# is why OVERSHOOT_RISE on the final epoch alone never fired.
DIVERGENCE_RISE = 0.5

# The smooth loss starts as an average over a handful of steps and can sit low by
# chance (bruises: minimum at epoch 0, then a healthy-looking +22%). Epochs in the
# first tenth of the run are read but never taken as the best.
SETTLING_FRACTION = 0.1

_SMOOTH_LOSS_RE = re.compile(r"smooth loss=([0-9.eE+-]+)")
_EPOCH_BAR_RE = re.compile(r"^epoch:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)")

_MAX_HEADER_BYTES = 100 * 1024 * 1024

# itemsize and the struct code that turns a sign-cleared bit pattern back into a float
_FLOAT_DTYPES = {
    "BF16": (2, "H"),
    "F16": (2, "H"),
    "F32": (4, "I"),
}
_ARRAY_CODE = {2: "H", 4: "I"}
_SIGN_MASK = {2: 0x7FFF, 4: 0x7FFFFFFF}


class _Unreadable(Exception):
    """The file is not a safetensors we can inspect."""


def _read_header(f) -> dict:
    raw_len = f.read(8)
    if len(raw_len) < 8:
        raise _Unreadable("file is too short to be a safetensors file")
    (length,) = struct.unpack("<Q", raw_len)
    if length <= 0 or length > _MAX_HEADER_BYTES:
        raise _Unreadable(f"header length {length} is not plausible")
    raw = f.read(length)
    if len(raw) < length:
        raise _Unreadable("header is truncated — the file did not finish writing")
    try:
        header = json.loads(raw)
    except ValueError as e:
        raise _Unreadable(f"header is not JSON ({e})") from e
    if not isinstance(header, dict):
        raise _Unreadable("header is not a JSON object")
    header.pop("__metadata__", None)
    return header


def _bits_to_float(bits: int, dtype: str) -> float:
    """A sign-cleared bit pattern back to its magnitude."""
    if dtype == "BF16":                       # BF16 is the top 16 bits of a float32
        return struct.unpack("<f", struct.pack("<HH", 0, bits))[0]
    if dtype == "F16":
        return struct.unpack("<e", struct.pack("<H", bits))[0]
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _tensor_peak(f, base: int, spec: dict) -> float:
    """Largest |value| in one tensor.

    Magnitude ordering of IEEE-754 floats is monotonic in the sign-cleared bit
    pattern, so the largest magnitude is the largest masked integer — no
    per-element float conversion needed.
    """
    dtype = spec["dtype"]
    itemsize, _ = _FLOAT_DTYPES[dtype]
    start, end = spec["data_offsets"]
    f.seek(base + start)
    raw = f.read(end - start)
    if len(raw) < end - start:
        raise _Unreadable("tensor data is truncated — the file did not finish writing")
    values = array.array(_ARRAY_CODE[itemsize])
    values.frombytes(raw[: len(raw) - len(raw) % itemsize])
    if sys.byteorder != "little":
        values.byteswap()
    mask = _SIGN_MASK[itemsize]
    return _bits_to_float(max((v & mask for v in values), default=0), dtype)


def _b_side(header: dict) -> list:
    """The up-projection tensors, under either suffix convention."""
    return [(k, v) for k, v in header.items()
            if ".lora_B." in k or ".lora_up." in k]


def peak_lora_b_abs(path: str) -> float:
    """Largest |weight| across every lora_B tensor. 0.0 if there are none.

    Raises _Unreadable for a file that is not an inspectable safetensors.
    """
    with open(path, "rb") as f:
        header = _read_header(f)
        base = f.tell()          # data_offsets are relative to the end of the header
        peak = 0.0
        for name, spec in _b_side(header):
            if spec.get("dtype") not in _FLOAT_DTYPES:
                raise _Unreadable(
                    f"{name} has dtype {spec.get('dtype')!r}, which this check "
                    "cannot measure")
            peak = max(peak, _tensor_peak(f, base, spec))
    return peak


def verify_output(path: str) -> str | None:
    """None when the file is a LoRA worth shipping, otherwise why it is not.

    The message is the whole point: it has to be enough to act on without
    re-opening the file by hand.
    """
    p = Path(path)
    if not p.is_file():
        return f"No LoRA was written to {path}."
    try:
        with open(p, "rb") as f:
            header = _read_header(f)
    except (_Unreadable, OSError) as e:
        return (f"{p.name} is not a readable safetensors file: {e}. "
                "The run may have been interrupted mid-write.")

    if not header:
        return f"{p.name} contains no tensors."

    if not any(k.startswith(READABLE_PREFIX + ".") for k in header):
        found = sorted({k.split(".", 1)[0] for k in header})[:3]
        return (f"{p.name} is in the wrong key namespace: it has no "
                f"'{READABLE_PREFIX}.' keys (found {', '.join(found)}). The image "
                "tool matches on the 'diffusion_model.' names, so it would load "
                "this file and apply nothing. Check output_model_format is "
                "COMFY_LORA, not DIFFUSERS_LORA.")

    if not _b_side(header):
        return (f"{p.name} has no lora_B tensors, so there is no trained "
                "up-projection in it — this is not a usable LoRA.")

    try:
        peak = peak_lora_b_abs(str(p))
    except (_Unreadable, OSError) as e:
        return f"{p.name} could not be measured: {e}"

    if peak < MIN_TRAINED_ABSMAX:
        return (f"{p.name} did not learn: its strongest lora_B weight is "
                f"{peak:.3e}, below the {MIN_TRAINED_ABSMAX:.0e} floor "
                "(a working LoRA sits near 4e-02). lora_B starts at zero, so "
                "this means training never moved it — the LoRA will load and "
                "change nothing. Check the optimizer's learning rate: PRODIGY "
                "reads it as a multiplier on its own step size and wants 1.0.")
    return None


# ----------------------------------------------------------------------------
# Did the run overshoot? Read from trainer.log, which is the run's record anyway.
# ----------------------------------------------------------------------------

def loss_by_epoch(log_text: str) -> dict:
    """{completed_epochs: smooth loss at that moment}, from OneTrainer's tqdm bars.

    The step bar carries `smooth loss=` (an EMA over steps); the epoch bar that
    follows it reads `N/T` with N epochs complete. The last smooth loss seen before
    each epoch bar is the loss that closes that epoch. An epoch bar with no loss
    before it (the 0/T bar that opens the run) is skipped.
    """
    out = {}
    current = None
    for line in log_text.splitlines():
        m = _SMOOTH_LOSS_RE.search(line)
        if m:
            try:
                current = float(m.group(1))
            except ValueError:
                pass
        e = _EPOCH_BAR_RE.match(line)
        if e and current is not None:
            out[int(e.group(1))] = current
    return out


def overshoot(losses: dict, total_epochs: int, rise: float = OVERSHOOT_RISE,
              settle: float = SETTLING_FRACTION) -> dict | None:
    """The overshoot in this loss curve, or None when the run ended near its best.

    The best epoch is the lowest end-of-epoch loss after SETTLING_FRACTION of the
    run; the run overshot when its final loss sits OVERSHOOT_RISE or more above it.
    """
    if not losses:
        return None
    epochs = sorted(losses)
    final_epoch = epochs[-1]
    first_counted = max(1, round(int(total_epochs or final_epoch) * settle))
    candidates = [e for e in epochs if e >= first_counted] or epochs
    best_epoch = min(candidates, key=lambda e: losses[e])
    best, final = losses[best_epoch], losses[final_epoch]
    if best <= 0:
        return None
    grew = final / best - 1.0
    if grew < rise:
        return None
    return {"kind": "overshoot", "last_good_epoch": best_epoch, "best_epoch": best_epoch, "best": best, "final_epoch": final_epoch,
            "final": final, "rise": grew, "total_epochs": int(total_epochs or final_epoch)}


def divergence(losses: dict, total_epochs: int, rise: float = DIVERGENCE_RISE,
               settle: float = SETTLING_FRACTION) -> dict | None:
    """The first epoch whose loss jumped DIVERGENCE_RISE or more above the best
    before it, or None. Epochs in the first SETTLING_FRACTION of the run are
    neither a best nor a jump."""
    total = int(total_epochs or (max(losses) if losses else 0))
    first_counted = max(1, round(total * settle))
    best_epoch = None
    for e in sorted(losses):
        if e < first_counted:
            continue
        if best_epoch is not None and losses[best_epoch] > 0:
            jump = losses[e] / losses[best_epoch] - 1.0
            if jump >= rise:
                return {"kind": "diverged", "last_good_epoch": e - 1,
                        "best_epoch": best_epoch, "best": losses[best_epoch],
                        "spike_epoch": e, "spike": losses[e], "rise": jump,
                        "total_epochs": total}
        if best_epoch is None or losses[e] < losses[best_epoch]:
            best_epoch = e
    return None


def _curve_from_log(log_path: str):
    """(losses, total_epochs) from the run's trainer.log, or None without one."""
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    losses = loss_by_epoch(text)
    if not losses:
        return None
    total = 0
    for line in text.splitlines():
        e = _EPOCH_BAR_RE.match(line)
        if e:
            total = int(e.group(2))
    return losses, total or max(losses)


def overshoot_from_log(log_path: str) -> dict | None:
    """overshoot() over the run's trainer.log; None when there is no log to read."""
    curve = _curve_from_log(log_path)
    return overshoot(*curve) if curve else None


def trouble_from_log(log_path: str) -> dict | None:
    """What went wrong with the run's loss curve: a blow-up anywhere in it, else
    an overshoot at its end, else None."""
    curve = _curve_from_log(log_path)
    if not curve:
        return None
    return divergence(*curve) or overshoot(*curve)


class DivergenceWatch:
    """divergence(), fed the trainer's output one line at a time while it runs.
    feed() returns the divergence the moment an epoch closes on one."""

    def __init__(self):
        self._losses = {}
        self._current = None
        self._found = None

    def feed(self, line: str) -> dict | None:
        if self._found is not None:
            return None
        m = _SMOOTH_LOSS_RE.search(line)
        if m:
            try:
                self._current = float(m.group(1))
            except ValueError:
                pass
        e = _EPOCH_BAR_RE.match(line)
        if e and self._current is not None:
            epoch, total = int(e.group(1)), int(e.group(2))
            if epoch not in self._losses:
                self._losses[epoch] = self._current
                self._found = divergence(self._losses, total)
                return self._found
        return None


def describe(found: dict) -> str:
    """Either kind of trouble as one sentence a user can act on."""
    if found.get("kind") == "diverged":
        return describe_divergence(found)
    return describe_overshoot(found)


def describe_divergence(found: dict) -> str:
    return (f"Training blew up: the loss was {found['best']:.4f} at epoch "
            f"{found['best_epoch']} of {found['total_epochs']}, then jumped "
            f"{found['rise'] * 100:.0f}% to {found['spike']:.4f} at epoch "
            f"{found['spike_epoch']}. Everything trained after that renders as "
            "noise, so the run was stopped there.")


def describe_overshoot(found: dict) -> str:
    """The overshoot as one sentence a user can act on."""
    return (f"Training overshot: the loss was lowest at epoch {found['best_epoch']} "
            f"of {found['total_epochs']} ({found['best']:.4f}) and finished "
            f"{found['rise'] * 100:.0f}% above that ({found['final']:.4f} at epoch "
            f"{found['final_epoch']}). The saved file is the overshot state, which "
            "renders as a mosaic instead of the subject.")
