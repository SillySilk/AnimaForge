import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from ui.gauge import DialRow

_app = QApplication.instance() or QApplication([])


def test_dialrow_constructs_with_four_gauges():
    d = DialRow()
    assert all(g is not None for g in (d.epoch, d.loss, d.speed, d.eta))


def test_loss_is_informational_with_trend_no_zones():
    # Loss dial is informational only (no green/amber/red 'danger' zones) — value + trend.
    d = DialRow()
    d.set_loss(0.120)
    assert d.loss._value_text == "0.120"
    d.set_loss(0.100)
    assert d.loss._note == "falling"
    d.set_loss(0.130)
    assert d.loss._note == "rising"
    assert not hasattr(DialRow, "_GREEN")   # zones removed


def test_epoch_and_speed_fractions():
    d = DialRow()
    d.set_epoch(5, 10)
    assert d.epoch._value_text == "5 / 10" and abs(d.epoch._frac - 0.5) < 1e-6
    d.set_speed(2.0, ceiling=4.0)
    assert abs(d.speed._frac - 0.5) < 1e-6


def test_gauge_paint_uses_resolved_family(monkeypatch):
    from ui import gauge as gauge_mod
    calls = []
    monkeypatch.setattr(
        gauge_mod, "primary_family",
        lambda role: (calls.append(role), "Segoe UI")[1],
    )
    d = DialRow()
    d.epoch.grab()  # offscreen render forces a paintEvent
    assert calls and all(r == "type" for r in calls)
