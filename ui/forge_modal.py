"""Reusable forge-style modal overlay.

The redesign leans on modals throughout (Caption Options, Train Presets, Ready
checklist, the Dataset editor, confirms, etc.). Rather than native ``QDialog``
windows — which pop a separate OS window and don't match the mock's dimmed,
in-canvas scrim — :class:`ForgeModal` is an overlay ``QWidget`` parented to the
main window: a dark blurred scrim behind a centred card with a top gold hairline,
an ✕ close button, and an ``af-rise`` entrance (fade + slight upward move).

Usage::

    m = ForgeModal(main_window, title="Captioning",
                   eyebrow="Step 01 · Options",
                   subtitle="Pick the passes. The set runs top to bottom.")
    m.body.addWidget(...)            # content
    cancel = m.add_footer_button("Cancel")
    run = m.add_footer_button("Run Captioning", primary=True)
    cancel.clicked.connect(m.close_modal)
    m.open()

The scrim swallows clicks to the app behind it; clicking the scrim (or Esc, or
the ✕) closes. Connect :attr:`closed` for teardown.
"""

from __future__ import annotations

from PySide6.QtCore import (
    Qt, Signal, QEvent, QObject, QPoint, QPropertyAnimation, QEasingCurve,
    QParallelAnimationGroup,
)
from PySide6.QtWidgets import (
    QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)


class ForgeModal(QWidget):
    closed = Signal()

    def __init__(self, parent: QWidget, title: str, eyebrow: str | None = None,
                 subtitle: str | None = None, max_width: int = 520):
        super().__init__(parent)
        self.setObjectName("af_modal_scrim")
        self._max_width = max_width
        self._anim = None

        # ---- card ----
        self._card = QWidget(self)
        self._card.setObjectName("af_modal_card")
        card_v = QVBoxLayout(self._card)
        card_v.setContentsMargins(0, 0, 0, 0)
        card_v.setSpacing(0)

        rule = QFrame()
        rule.setObjectName("af_modal_rule")
        rule.setFixedHeight(2)
        card_v.addWidget(rule)

        # scrolling inner content so tall modals stay within the window
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setObjectName("af_modal_card")
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(28, 22, 28, 24)
        iv.setSpacing(8)

        if eyebrow:
            lbl_eb = QLabel(eyebrow.upper())
            lbl_eb.setObjectName("af_eyebrow_flame")
            iv.addWidget(lbl_eb)
        lbl_title = QLabel(title)
        lbl_title.setObjectName("af_modal_title")
        iv.addWidget(lbl_title)
        if subtitle:
            lbl_sub = QLabel(subtitle)
            lbl_sub.setObjectName("af_marker")
            lbl_sub.setWordWrap(True)
            lbl_sub.setContentsMargins(0, 2, 0, 10)
            iv.addWidget(lbl_sub)
        else:
            iv.addSpacing(8)

        # caller-populated content
        self.body = QVBoxLayout()
        self.body.setSpacing(12)
        iv.addLayout(self.body)

        iv.addStretch()

        # footer (right-aligned action buttons)
        self._footer = QHBoxLayout()
        self._footer.setSpacing(12)
        self._footer.addStretch()
        iv.addSpacing(12)
        iv.addLayout(self._footer)

        scroll.setWidget(inner)
        card_v.addWidget(scroll)

        # floating close button
        self._close_btn = QPushButton("✕", self._card)
        self._close_btn.setObjectName("af_modal_close")
        self._close_btn.setFixedSize(32, 32)
        self._close_btn.setCursor(Qt.PointingHandCursor)
        self._close_btn.clicked.connect(self.close_modal)

        # opacity effect used for the fade portion of af-rise
        self._fx = QGraphicsOpacityEffect(self._card)
        self._card.setGraphicsEffect(self._fx)

        if parent is not None:
            parent.installEventFilter(self)

    # ------------------------------------------------------------------
    def add_footer_button(self, text: str, primary: bool = False,
                          danger: bool = False) -> QPushButton:
        btn = QPushButton(text)
        if primary:
            btn.setObjectName("btn_primary")
        elif danger:
            btn.setObjectName("btn_danger")
        else:
            btn.setObjectName("af_btn_ghost")
        btn.setMinimumHeight(40)
        btn.setCursor(Qt.PointingHandCursor)
        self._footer.addWidget(btn)
        return btn

    def open(self):
        """Show the modal centred over the parent with the af-rise entrance."""
        self._reposition()
        self.show()
        self.raise_()
        self._animate_in()
        self.setFocus()

    def close_modal(self):
        self.hide()
        self.closed.emit()
        self.deleteLater()

    # ------------------------------------------------------------------
    def _reposition(self):
        parent = self.parentWidget()
        if parent is None:
            return
        self.setGeometry(parent.rect())
        pw, ph = parent.width(), parent.height()
        w = min(self._max_width, max(320, pw - 64))
        self._card.setFixedWidth(w)
        self._card.adjustSize()
        h = min(self._card.sizeHint().height(), int(ph * 0.86))
        self._card.resize(w, h)
        x = (pw - w) // 2
        y = (ph - h) // 2
        self._card.move(x, max(24, y))
        self._card_home = self._card.pos()
        self._close_btn.move(w - 32 - 12, 12)
        self._close_btn.raise_()

    def _animate_in(self):
        home = self._card_home
        start = QPoint(home.x(), home.y() + 16)
        move = QPropertyAnimation(self._card, b"pos", self)
        move.setDuration(360)
        move.setStartValue(start)
        move.setEndValue(home)
        move.setEasingCurve(QEasingCurve.OutCubic)
        fade = QPropertyAnimation(self._fx, b"opacity", self)
        fade.setDuration(220)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setEasingCurve(QEasingCurve.OutCubic)
        group = QParallelAnimationGroup(self)
        group.addAnimation(move)
        group.addAnimation(fade)
        group.start(QPropertyAnimation.DeleteWhenStopped)
        self._anim = group

    # ---- input: scrim click / Esc close, keep parent-sized ----
    def mousePressEvent(self, event):
        # clicks that reach the scrim (not the card) dismiss the modal
        self.close_modal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close_modal()
        else:
            super().keyPressEvent(event)

    def eventFilter(self, obj: QObject, event: QEvent):
        if obj is self.parentWidget() and event.type() == QEvent.Resize:
            if self.isVisible():
                self._reposition()
        return False
