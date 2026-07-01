"""Characters tab — read-only "Cast bundles".

The filename convention (NAME_SERIAL_CATEGORY, core/naming.py) is the source of truth for who
appears in each image. This tab no longer edits a roster: it groups the dataset's images into
per-character bundles (solo + combined) so you can confirm each name matches its pictures, shows
a "Needs naming" bundle for non-conforming files with a shortcut back to the validator, and keeps
a slim dataset-wide @style anchor. On load it rebuilds the per-folder animaforge_characters.json
roster/assignments from the filenames (preserving the style anchor) so what you see is exactly
what flows into captions.
"""
import io
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListView, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from core import characters as characters_mod
from core import naming
from core.dataset_manager import scan_folder

THUMB = 120


def _thumb_pixmap(path: str, size: int = THUMB) -> QPixmap:
    """Build a padded square thumbnail QPixmap via Pillow (robust across formats)."""
    try:
        from PIL import Image
        img = Image.open(path)
        img.thumbnail((size, size), Image.LANCZOS)
        padded = Image.new("RGB", (size, size), (12, 11, 10))
        ox, oy = (size - img.width) // 2, (size - img.height) // 2
        if img.mode == "RGBA":
            padded.paste(img, (ox, oy), img)
        else:
            padded.paste(img.convert("RGB"), (ox, oy))
        buf = io.BytesIO()
        padded.save(buf, format="PNG")
        buf.seek(0)
        return QPixmap.fromImage(QImage.fromData(buf.read()))
    except Exception:
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        return pm


class CharactersTab(QWidget):
    characters_changed = Signal()
    status_message = Signal(str)
    names_validated = Signal()        # Fix-names finished → Dataset tab re-combines the .txt

    def __init__(self, parent=None):
        super().__init__(parent)
        self._folder_path = ""
        self._build_ui()

    # ------------------------------------------------------------------
    # Wiring from MainWindow (preserved public surface)
    # ------------------------------------------------------------------
    def set_dataset(self, folder: str):
        """Load a folder: rebuild roster/assignments from filenames (keeping the style anchor)
        so captions match what we show, then render the bundles."""
        self._folder_path = folder or ""
        if self._folder_path:
            names = self._scan_names()
            data = (naming.write_characters_from_names(self._folder_path, names) if names
                    else characters_mod.load(self._folder_path))
            self._set_anchor_text(data.style_anchor)
        self._render()

    def reload_characters(self):
        """Re-render from disk after the validator or Dataset tab changed things."""
        if self._folder_path:
            self._set_anchor_text(characters_mod.load(self._folder_path).style_anchor)
        self._render()

    def set_style_anchor(self, text: str):
        """Set the @-style anchor from elsewhere (e.g. the Home cockpit); persists it."""
        text = (text or "").strip()
        if text != self._style_anchor_edit.text().strip():
            self._style_anchor_edit.setText(text)  # fires _on_style_anchor_changed

    def auto_detect_from_filenames(self) -> int:
        """Headless detection for the Home Quick Run pipeline. Rebuilds roster/assignments from
        the v2 filename convention and returns the number of distinct subjects (0 = none)."""
        if not self._folder_path:
            return 0
        names = self._scan_names()
        if not names:
            return 0
        tokens, _ = naming.assignments_from_names(names)
        naming.write_characters_from_names(self._folder_path, names)
        self._render()
        return len(tokens)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(10)

        title = QLabel("Characters")
        title.setObjectName("label_section")
        root.addWidget(title)

        anchor_row = QHBoxLayout()
        anchor_row.addWidget(QLabel("Style/artist anchor:"))
        self._style_anchor_edit = QLineEdit()
        self._style_anchor_edit.setPlaceholderText("@mystyle (optional)")
        self._style_anchor_edit.setFixedWidth(220)
        self._style_anchor_edit.textChanged.connect(self._on_style_anchor_changed)
        anchor_row.addWidget(self._style_anchor_edit)
        anchor_row.addStretch()
        root.addLayout(anchor_row)

        caption = QLabel("These character names come from your filenames and are written into "
                         "every caption automatically.")
        caption.setObjectName("label_field")
        caption.setWordWrap(True)
        root.addWidget(caption)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._bundles_host = QWidget()
        self._bundles_layout = QVBoxLayout(self._bundles_host)
        self._bundles_layout.setContentsMargins(0, 0, 0, 0)
        self._bundles_layout.setSpacing(12)
        self._bundles_layout.setAlignment(Qt.AlignTop)
        self._scroll.setWidget(self._bundles_host)
        root.addWidget(self._scroll, 1)

    def _set_anchor_text(self, text: str):
        self._style_anchor_edit.blockSignals(True)
        self._style_anchor_edit.setText(text or "")
        self._style_anchor_edit.blockSignals(False)

    def _on_style_anchor_changed(self, text: str):
        if not self._folder_path:
            return
        data = characters_mod.load(self._folder_path)
        data.style_anchor = text.strip()
        characters_mod.save(self._folder_path, data)
        self.characters_changed.emit()

    # ------------------------------------------------------------------
    # Bundles rendering
    # ------------------------------------------------------------------
    def _scan_names(self):
        return [Path(d["image_path"]).name for d in scan_folder(self._folder_path)]

    def _clear_bundles(self):
        while self._bundles_layout.count():
            item = self._bundles_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _make_bundle(self, title: str, image_paths, warn: bool = False,
                     show_fix: bool = False) -> QGroupBox:
        group = QGroupBox(title)
        if warn:
            group.setStyleSheet("QGroupBox { color: #d9534f; }")
        v = QVBoxLayout(group)
        if show_fix:
            bar = QHBoxLayout()
            fix = QPushButton("Fix names")
            fix.setObjectName("btn_primary")
            fix.clicked.connect(self._open_validator)
            bar.addWidget(fix)
            bar.addStretch()
            v.addLayout(bar)
        grid = QListWidget()
        grid.setViewMode(QListView.IconMode)
        grid.setIconSize(QSize(THUMB, THUMB))
        grid.setGridSize(QSize(THUMB + 16, THUMB + 16))
        grid.setResizeMode(QListView.Adjust)
        grid.setMovement(QListView.Static)
        grid.setSelectionMode(QListWidget.NoSelection)
        grid.setFocusPolicy(Qt.NoFocus)
        grid.setSpacing(8)
        grid.setUniformItemSizes(True)
        rows = max(1, (len(image_paths) + 4) // 5)
        grid.setFixedHeight(min(3, rows) * (THUMB + 16) + 24)
        for p in image_paths:
            item = QListWidgetItem(QIcon(_thumb_pixmap(p)), "")
            item.setToolTip(Path(p).name)
            grid.addItem(item)
        v.addWidget(grid)
        return group

    def _render(self):
        self._clear_bundles()
        if not self._folder_path:
            self._bundles_layout.addWidget(
                QLabel("Load a dataset folder on the Dataset tab first."))
            return
        data = scan_folder(self._folder_path)
        names = [Path(d["image_path"]).name for d in data]
        paths = {Path(d["image_path"]).name: d["image_path"] for d in data}
        b = naming.bundles_from_names(names)

        if b["needs_naming"]:
            self._bundles_layout.addWidget(self._make_bundle(
                f"⚠  Needs naming — {len(b['needs_naming'])} file(s)",
                [paths[n] for n in b["needs_naming"] if n in paths],
                warn=True, show_fix=True))
        for grp in b["solo"] + b["combined"]:
            self._bundles_layout.addWidget(self._make_bundle(
                f"{grp['name']} — {len(grp['images'])} image(s)",
                [paths[n] for n in grp["images"] if n in paths]))
        if not (b["needs_naming"] or b["solo"] or b["combined"]):
            self._bundles_layout.addWidget(QLabel("No images in this folder yet."))

    # ------------------------------------------------------------------
    # Fix names → re-open the validator
    # ------------------------------------------------------------------
    def _open_validator(self):
        if not self._folder_path:
            QMessageBox.warning(self, "No Dataset",
                                "Load a dataset folder on the Dataset tab first.")
            return
        from ui.name_validate_view import NameValidateView
        dlg = NameValidateView(self._folder_path, self)
        dlg.characters_changed.connect(self._on_validated)
        dlg.exec()

    def _on_validated(self):
        self.reload_characters()
        self.status_message.emit("Character names updated from filenames.")
        self.names_validated.emit()       # Dataset tab re-combines so names land in the prompt
        self.characters_changed.emit()
