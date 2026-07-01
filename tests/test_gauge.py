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


def test_loss_zone_colours():
    d = DialRow()
    d.set_loss(0.05)
    assert d.loss._fill is DialRow._GREEN and d.loss._note == "healthy"
    d.set_loss(0.13)
    assert d.loss._fill is DialRow._AMBER and d.loss._note == "watch"
    d.set_loss(0.20)
    assert d.loss._fill is DialRow._RED and d.loss._note == "high"


def test_epoch_and_speed_fractions():
    d = DialRow()
    d.set_epoch(5, 10)
    assert d.epoch._value_text == "5 / 10" and abs(d.epoch._frac - 0.5) < 1e-6
    d.set_speed(2.0, ceiling=4.0)
    assert abs(d.speed._frac - 0.5) < 1e-6
