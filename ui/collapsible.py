"""CollapsibleBox — a titled, click-to-expand section themed for the black/gold UI.

Used to sink set-once Train-tab configuration (Anima settings, optimizer, run
options, sample previews) below the primary flow while keeping it one click away.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class CollapsibleBox(QWidget):
    def __init__(self, title: str, expanded: bool = False, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._toggle.setCursor(Qt.PointingHandCursor)
        self._toggle.setStyleSheet(
            "QToolButton { background-color:#141312; color:#d4af37; border:1px solid #2a2a1e;"
            " border-radius:6px; padding:8px 10px; font-weight:700; letter-spacing:1px;"
            " text-align:left; }"
            "QToolButton:hover { border:1px solid #8a5a12; color:#f4d160; }"
        )
        self._toggle.toggled.connect(self._on_toggled)
        v.addWidget(self._toggle)

        self._content = QFrame()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 8, 8, 8)
        self._content.setVisible(expanded)
        v.addWidget(self._content)

    def content_layout(self) -> QVBoxLayout:
        """The layout to add this section's widgets to."""
        return self._content_layout

    def set_expanded(self, expanded: bool) -> None:
        self._toggle.setChecked(expanded)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content.setVisible(checked)
