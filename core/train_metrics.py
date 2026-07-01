"""Parse live training metrics from sd-scripts' tqdm progress lines.

sd-scripts prints a tqdm bar to stderr each step, e.g.::

    steps:  42%|####      | 672/1600 [05:23<08:37,  1.79it/s, avr_loss=0.0834]

Everything the Train dials need is in that one line — step/total, elapsed, ETA,
iterations/second, and the running average loss. :func:`parse_tqdm` pulls out
whatever is present (a partial line may omit ``avr_loss``) and returns a dict of
just the keys it found, so callers can update only the dials that have data.
"""

from __future__ import annotations

import re

_STEP = re.compile(r"(\d+)\s*/\s*(\d+)")
_TIMES = re.compile(r"\[(\d+:\d+(?::\d+)?)<(\d+:\d+(?::\d+)?)")
_ITS = re.compile(r"([\d.]+)\s*it/s")
_SPI = re.compile(r"([\d.]+)\s*s/it")
_LOSS = re.compile(r"avr_loss[=:]\s*([\d.]+)")


def _to_seconds(clock: str) -> int:
    parts = [int(p) for p in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3], parts[-2], parts[-1]
    return h * 3600 + m * 60 + s


def parse_tqdm(line: str) -> dict:
    """Extract training metrics from one tqdm line. Returns only the keys found.

    Possible keys: ``step`` (int), ``total`` (int), ``elapsed`` (s), ``eta`` (s),
    ``it_s`` (float, iterations/second), ``loss`` (float).
    """
    if not line or "|" not in line:
        return {}
    out: dict = {}
    m = _STEP.search(line)
    if m:
        out["step"], out["total"] = int(m.group(1)), int(m.group(2))
    m = _TIMES.search(line)
    if m:
        out["elapsed"] = _to_seconds(m.group(1))
        out["eta"] = _to_seconds(m.group(2))
    m = _ITS.search(line)
    if m:
        out["it_s"] = float(m.group(1))
    else:
        m = _SPI.search(line)
        if m and float(m.group(1)) > 0:
            out["it_s"] = round(1.0 / float(m.group(1)), 3)
    m = _LOSS.search(line)
    if m:
        out["loss"] = float(m.group(1))
    return out


def format_eta(seconds: int) -> str:
    """Human ETA like ``8m 37s`` / ``1h 02m`` from a second count."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
