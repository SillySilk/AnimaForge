"""Analog semicircular gauges for the Train monitor (Epoch / Loss / Speed / ETA).

Each :class:`Gauge` custom-paints a 180° dial: a dim track, a molten-gold filled
arc + needle for the current value, a big value readout, and a label. The Loss
gauge is *zoned* — its fill colour reflects a health band (green/amber/red) fed
in via :meth:`set`. :class:`DialRow` lays four of them out in a row.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import QWidget, QHBoxLayout, QFrame

_TRACK = QColor("#2a2a1e")
_GOLD = QColor("#d4af37")
_GOLD_HI = QColor("#f4d160")
_MUTE = QColor("#8a8a93")
_CREAM = QColor("#e8e0c8")


class Gauge(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = label.upper()
        self._value_text = "—"
        self._frac = 0.0
        self._fill = _GOLD
        self._note = ""
        self.setMinimumSize(128, 104)

    def set(self, value_text: str, fraction: float, fill: QColor | None = None,
            note: str = "") -> None:
        """Update the readout: display text, 0..1 needle fraction, optional fill
        colour (for the zoned Loss dial) and a small note under the label."""
        self._value_text = value_text
        self._frac = max(0.0, min(1.0, float(fraction)))
        self._fill = fill or _GOLD
        self._note = note
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        pad = 12
        r = min((w - 2 * pad) / 2.0, h - 2 * pad - 14)
        cx, cy = w / 2.0, pad + r + 2
        rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)

        # dim track (top semicircle)
        pen = QPen(_TRACK, 7, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, 0, 180 * 16)

        # filled arc from the left end up to the value
        angle = 180.0 * (1.0 - self._frac)              # degrees, 3 o'clock = 0
        pen.setColor(self._fill)
        p.setPen(pen)
        p.drawArc(rect, int(angle * 16), int((180.0 - angle) * 16))

        # needle
        rad = math.radians(angle)
        tip = QPointF(cx + (r - 2) * math.cos(rad), cy - (r - 2) * math.sin(rad))
        npen = QPen(_GOLD_HI, 2.4, Qt.SolidLine, Qt.RoundCap)
        p.setPen(npen)
        p.drawLine(QPointF(cx, cy), tip)
        p.setBrush(_GOLD_HI)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), 3.2, 3.2)

        # value readout (inside the arc)
        p.setPen(_CREAM)
        vf = QFont("Special Elite")
        vf.setPixelSize(15)
        vf.setBold(True)
        p.setFont(vf)
        p.drawText(QRectF(0, cy - r * 0.55, w, 22), Qt.AlignHCenter | Qt.AlignVCenter,
                   self._value_text)

        # label (+ optional note) under the dial
        p.setPen(_MUTE)
        lf = QFont("Special Elite")
        lf.setPixelSize(10)
        p.setFont(lf)
        text = self._label + (f"  ·  {self._note}" if self._note else "")
        p.drawText(QRectF(0, h - 16, w, 14), Qt.AlignHCenter | Qt.AlignVCenter, text)
        p.end()


class DialRow(QFrame):
    """Row of the four training dials. Update via the typed setters below."""

    _LOSS_FACE = 0.25

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("af_card")
        self._last_loss = None
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 12, 16, 12)
        row.setSpacing(10)
        self.epoch = Gauge("Epoch")
        self.loss = Gauge("Loss")
        self.speed = Gauge("Speed")
        self.eta = Gauge("ETA")
        for g in (self.epoch, self.loss, self.speed, self.eta):
            row.addWidget(g, 1)

    def set_epoch(self, current: int, total: int):
        frac = (current / total) if total else 0.0
        self.epoch.set(f"{current} / {total}" if total else f"{current}", frac)

    def set_loss(self, loss: float):
        # Informational only — no green/amber/red 'danger' zones. For a flow-matching model
        # like Anima the absolute loss is a weak, noisy overfitting signal; the user judges by
        # the sample previews and stops by eye. We show the value + a simple trend arrow.
        frac = min(1.0, loss / self._LOSS_FACE)
        trend = ""
        if self._last_loss is not None:
            if loss < self._last_loss - 1e-4:
                trend = "falling"
            elif loss > self._last_loss + 1e-4:
                trend = "rising"
            else:
                trend = "flat"
        self._last_loss = loss
        self.loss.set(f"{loss:.3f}", frac, note=trend)

    def set_speed(self, it_s: float, ceiling: float = 4.0):
        # Fast GPUs (RTX 5090) exceed a fixed 4 it/s and would peg the needle for the
        # whole run — let the session's observed peak stretch the dial instead.
        self._speed_peak = max(getattr(self, "_speed_peak", 0.0), it_s)
        self.speed.set(f"{it_s:.2f} it/s", min(1.0, it_s / max(ceiling, self._speed_peak)))

    def set_eta(self, eta_seconds: int, elapsed_seconds: int = 0):
        from core.train_metrics import format_eta
        total = eta_seconds + elapsed_seconds
        frac = (elapsed_seconds / total) if total else 0.0
        self.eta.set(format_eta(eta_seconds), frac)
