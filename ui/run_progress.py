"""A thick, shared run-progress widget used on both Home and the Train tab.

Replaces the thin QProgressBar that lived only on the Train tab. It shows a phase
label (Detecting / Captioning / Training / Done / Error), a chunky bar, and a
step / total · percent readout. The same instance type is reused in both places so
the two surfaces always look and behave identically.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget


class RunProgress(QWidget):
    _BAR_QSS = (
        "QProgressBar {{ border: 1px solid #3a3a1f; border-radius: 7px; "
        "background-color: #0c0b0a; color: {text}; font-size: 12px; font-weight: 700; "
        "text-align: center; }}"
        "QProgressBar::chunk {{ border-radius: 6px; background-color: {chunk}; }}"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        top = QHBoxLayout()
        self._phase = QLabel("Idle")
        self._phase.setStyleSheet("color: #d4af37; font-size: 12px; font-weight: 700;")
        top.addWidget(self._phase)
        top.addStretch()
        self._counter = QLabel("")
        self._counter.setStyleSheet("color: #8a8a93; font-size: 12px; font-weight: 600;")
        top.addWidget(self._counter)
        v.addLayout(top)

        self._bar = QProgressBar()
        self._bar.setMinimum(0)
        self._bar.setMaximum(100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFixedHeight(26)
        self._bar.setFormat("Waiting…")
        v.addWidget(self._bar)
        self._style_bar("#d4af37", "#d4af37")

    # ---- styling helper ----
    def _style_bar(self, chunk: str, text: str):
        self._bar.setStyleSheet(self._BAR_QSS.format(chunk=chunk, text=text))

    # ---- public API ----
    def set_phase(self, label: str):
        self._phase.setText(label or "")
        self._style_bar("#d4af37", "#d4af37")

    def set_progress(self, step: int, total: int):
        """Determinate progress. total <= 0 falls back to a waiting state."""
        total = int(total or 0)
        step = max(0, int(step or 0))
        if total <= 0:
            self._bar.setMaximum(100)
            self._bar.setValue(0)
            self._bar.setFormat("Waiting…")
            self._counter.setText("")
            return
        self._bar.setMaximum(total)
        self._bar.setValue(min(step, total))
        pct = int(100 * min(step, total) / total)
        self._bar.setFormat(f"{pct}%")
        self._counter.setText(f"{min(step, total)} / {total}")

    def set_indeterminate(self, label: str = ""):
        """Busy mode for phases with no known step total (detect / caption warmup)."""
        if label:
            self.set_phase(label)
        self._bar.setMaximum(0)  # Qt renders a busy/marquee bar
        self._bar.setFormat("")
        self._counter.setText("")

    def set_done(self, label: str = "Done"):
        self.set_phase(label)
        self._bar.setMaximum(100)
        self._bar.setValue(100)
        self._bar.setFormat("100%")
        self._style_bar("#7ed957", "#0c0b0a")

    def set_error(self, label: str = "Error"):
        self._phase.setText(label)
        self._phase.setStyleSheet("color: #d9534f; font-size: 12px; font-weight: 700;")
        self._bar.setMaximum(100)
        self._bar.setFormat("Stopped")
        self._style_bar("#d9534f", "#0c0b0a")

    def reset(self):
        self._phase.setText("Idle")
        self._phase.setStyleSheet("color: #d4af37; font-size: 12px; font-weight: 700;")
        self._counter.setText("")
        self._bar.setMaximum(100)
        self._bar.setValue(0)
        self._bar.setFormat("Waiting…")
        self._style_bar("#d4af37", "#d4af37")

    def apply(self, payload: dict):
        """Drive the widget from a serializable payload, so one signal can mirror this
        progress onto multiple RunProgress instances (Train + Home)."""
        kind = (payload or {}).get("kind")
        if kind == "phase":
            self.set_phase(payload.get("label", ""))
        elif kind == "progress":
            self.set_progress(payload.get("step", 0), payload.get("total", 0))
        elif kind == "indeterminate":
            self.set_indeterminate(payload.get("label", ""))
        elif kind == "done":
            self.set_done(payload.get("label", "Done"))
        elif kind == "error":
            self.set_error(payload.get("label", "Stopped"))
        elif kind == "reset":
            self.reset()
