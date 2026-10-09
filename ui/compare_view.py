"""Side-by-side view of a caption-style comparison: same prompt, same seed, two styles.

A preview surface only (hosted in a ForgeModal from Home): it shows images and
starts nothing.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGridLayout, QLabel, QScrollArea, QVBoxLayout, QWidget,
)

from core.compare import STYLE_LABEL

IMAGE_SIZE = 384
_MISSING = {"failed": "failed — no previews", "pending": "not finished yet",
            "done": "no preview for this prompt"}


class CompareView(QWidget):
    def __init__(self, entry: dict, rows: list, parent=None):
        super().__init__(parent)
        runs = entry.get("runs", [])[:2]
        styles = [STYLE_LABEL.get(r.get("style"), r.get("style", "")) for r in runs]
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        header = QLabel(f"{styles[0]} vs {styles[1]} — {entry.get('set', '')}")
        header.setObjectName("af_modal_title")
        lay.addWidget(header)

        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        for col, style in enumerate(styles):
            lbl = QLabel(style)
            lbl.setObjectName("af_eyebrow_mute")
            lbl.setAlignment(Qt.AlignCenter)
            grid.addWidget(lbl, 0, col)
        r = 1
        for row in rows:
            prompt = QLabel(row["prompt"])
            prompt.setWordWrap(True)
            prompt.setObjectName("label_field")
            grid.addWidget(prompt, r, 0, 1, 2)
            for col, side in enumerate(("a", "b")):
                grid.addWidget(self._image(row.get(side), row["status"][side]), r + 1, col)
            r += 2

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        scroll.setMinimumHeight(min(720, 120 + len(rows) * (IMAGE_SIZE + 40)))
        lay.addWidget(scroll, 1)

    @staticmethod
    def _image(path, status: str) -> QLabel:
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedSize(IMAGE_SIZE, IMAGE_SIZE)
        pix = QPixmap(path) if path else QPixmap()
        if pix.isNull():
            lbl.setText(_MISSING.get(status, "no preview"))
            lbl.setObjectName("af_eyebrow_mute")
        else:
            lbl.setPixmap(pix.scaled(IMAGE_SIZE, IMAGE_SIZE, Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation))
            lbl.setToolTip(path)
        return lbl
