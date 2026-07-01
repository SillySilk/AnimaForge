"""Large modal editor for a single dataset image: big preview + caption editor +
cast assignment. Opened from an image card.

Decoupled from DatasetTab via signals: it mutates the shared DatasetCharacters
object in place and emits `caption_saved` / `cast_changed` so the tab can persist
and refresh the gallery card.
"""
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QTextEdit, QVBoxLayout, QWidget,
)

from core.dataset_manager import save_caption
from core import characters as characters_mod
from ui.collapsible import CollapsibleBox

PREVIEW = 540


class ImageEditorDialog(QDialog):
    caption_saved = Signal(str, str)   # (txt_path, text)
    cast_changed = Signal(str)         # (image_name)

    def __init__(self, items, index, characters, parent=None):
        super().__init__(parent)
        self._items = items or []
        self._index = max(0, min(index, len(self._items) - 1)) if self._items else 0
        self._chars = characters
        self._cast_checks = []         # list[(token, QCheckBox)]

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._flush_caption)

        self.setWindowTitle("Edit Image")
        self.setMinimumSize(940, 640)
        self._build_ui()
        self._load_current()

    # ------------------------------------------------------------------
    def _current(self):
        return self._items[self._index] if self._items else None

    def _image_name(self):
        c = self._current()
        return Path(c["image_path"]).name if c else ""

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(14)

        # ---- left: navigation + image ----
        left = QVBoxLayout()
        nav = QHBoxLayout()
        self._prev_btn = QPushButton("‹ Prev")
        self._prev_btn.clicked.connect(lambda: self._go(-1))
        self._counter = QLabel("")
        self._counter.setAlignment(Qt.AlignCenter)
        self._next_btn = QPushButton("Next ›")
        self._next_btn.clicked.connect(lambda: self._go(1))
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._counter, 1)
        nav.addWidget(self._next_btn)
        left.addLayout(nav)

        self._image_label = QLabel()
        self._image_label.setObjectName("image_thumb")
        self._image_label.setFixedSize(PREVIEW, PREVIEW)
        self._image_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self._image_label)

        self._name_label = QLabel("")
        self._name_label.setObjectName("image_filename")
        self._name_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self._name_label)
        left.addStretch()
        root.addLayout(left)

        # ---- right: caption + cast ----
        right = QVBoxLayout()
        cap_title = QLabel("CAPTION")
        cap_title.setObjectName("label_section")
        right.addWidget(cap_title)

        self._caption_edit = QTextEdit()
        self._caption_edit.setPlaceholderText("Caption…")
        self._caption_edit.textChanged.connect(self._on_caption_changed)
        right.addWidget(self._caption_edit, 1)

        # Read-only sidecar accordions (view the .tags / .nl the caption was built from).
        self._tags_box = CollapsibleBox("Tags · .tags")
        self._tags_view = QLabel("—")
        self._tags_view.setWordWrap(True)
        self._tags_view.setObjectName("label_field")
        self._tags_box.content_layout().addWidget(self._tags_view)
        right.addWidget(self._tags_box)

        self._nl_box = CollapsibleBox("Description · .nl")
        self._nl_view = QLabel("—")
        self._nl_view.setWordWrap(True)
        self._nl_view.setObjectName("label_field")
        self._nl_box.content_layout().addWidget(self._nl_view)
        right.addWidget(self._nl_box)

        cast_title = QLabel("CHARACTERS IN THIS IMAGE")
        cast_title.setObjectName("label_section")
        right.addWidget(cast_title)

        self._cast_area = QScrollArea()
        self._cast_area.setWidgetResizable(True)
        self._cast_area.setFrameShape(QFrame.NoFrame)
        self._cast_area.setMaximumHeight(150)
        self._cast_host = QWidget()
        self._cast_layout = QVBoxLayout(self._cast_host)
        self._cast_layout.setContentsMargins(0, 0, 0, 0)
        self._cast_area.setWidget(self._cast_host)
        right.addWidget(self._cast_area)

        oneoff_row = QHBoxLayout()
        oneoff_row.addWidget(QLabel("One-off:"))
        self._oneoff_tok = QLineEdit()
        self._oneoff_tok.setPlaceholderText("token")
        self._oneoff_tok.setFixedWidth(140)
        self._oneoff_desc = QLineEdit()
        self._oneoff_desc.setPlaceholderText("recognition description")
        self._oneoff_tok.editingFinished.connect(self._commit_cast)
        self._oneoff_desc.editingFinished.connect(self._commit_cast)
        oneoff_row.addWidget(self._oneoff_tok)
        oneoff_row.addWidget(self._oneoff_desc, 1)
        right.addLayout(oneoff_row)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Done")
        close_btn.setObjectName("btn_primary")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        right.addLayout(close_row)

        root.addLayout(right, 1)

    # ------------------------------------------------------------------
    def _go(self, delta):
        if not self._items:
            return
        self._flush_caption()
        self._index = (self._index + delta) % len(self._items)
        self._load_current()

    def _load_current(self):
        c = self._current()
        if not c:
            return
        # image
        pm = QPixmap(c["image_path"])
        if pm.isNull():
            self._image_label.setText("No preview")
        else:
            self._image_label.setPixmap(
                pm.scaled(PREVIEW, PREVIEW, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        name = self._image_name()
        self._name_label.setText(name)
        self._counter.setText(f"{self._index + 1} / {len(self._items)}")
        self._prev_btn.setEnabled(len(self._items) > 1)
        self._next_btn.setEnabled(len(self._items) > 1)
        # caption (block save while loading)
        self._caption_edit.blockSignals(True)
        self._caption_edit.setPlainText(c.get("caption", ""))
        self._caption_edit.blockSignals(False)
        self._load_sidecars(c["image_path"])
        self._rebuild_cast()

    def _load_sidecars(self, image_path: str):
        stem = Path(image_path)
        for ext, view in ((".tags", self._tags_view), (".nl", self._nl_view)):
            side = stem.with_suffix(ext)
            try:
                text = side.read_text(encoding="utf-8", errors="ignore").strip() if side.exists() else ""
            except OSError:
                text = ""
            view.setText(text or "— none —")

    # ---- caption ----
    def _on_caption_changed(self):
        self._save_timer.start()

    def _flush_caption(self):
        self._save_timer.stop()
        c = self._current()
        if not c:
            return
        text = self._caption_edit.toPlainText()
        c["caption"] = text
        save_caption(c["txt_path"], text)
        self.caption_saved.emit(c["txt_path"], text)

    # ---- cast ----
    def _rebuild_cast(self):
        while self._cast_layout.count():
            item = self._cast_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        self._cast_checks = []
        name = self._image_name()
        entry = self._chars.assignments.get(name, {"present": [], "oneoffs": []})
        present = set(entry.get("present", []))
        oneoffs = entry.get("oneoffs", [])

        if self._chars.roster:
            for ch in self._chars.roster:
                cb = QCheckBox(f"{ch.token}  —  {ch.description}" if ch.description else ch.token)
                cb.setChecked(ch.token in present)
                cb.toggled.connect(self._commit_cast)
                self._cast_layout.addWidget(cb)
                self._cast_checks.append((ch.token, cb))
        else:
            self._cast_layout.addWidget(QLabel("(No roster yet — add characters in the Dataset tab.)"))

        self._oneoff_tok.blockSignals(True)
        self._oneoff_desc.blockSignals(True)
        self._oneoff_tok.setText(oneoffs[0]["token"] if oneoffs else "")
        self._oneoff_desc.setText(oneoffs[0]["description"] if oneoffs else "")
        self._oneoff_tok.blockSignals(False)
        self._oneoff_desc.blockSignals(False)

    def _commit_cast(self, *_):
        name = self._image_name()
        if not name:
            return
        present = [tok for tok, cb in self._cast_checks if cb.isChecked()]
        oneoffs = []
        if self._oneoff_tok.text().strip():
            oneoffs.append({"token": self._oneoff_tok.text().strip(),
                            "description": self._oneoff_desc.text().strip()})
        self._chars.assignments[name] = {"present": present, "oneoffs": oneoffs}
        self.cast_changed.emit(name)

    # ------------------------------------------------------------------
    def reject(self):
        self._flush_caption()
        super().reject()

    def accept(self):
        self._flush_caption()
        super().accept()
