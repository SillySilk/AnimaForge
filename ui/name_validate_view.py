"""Filename-convention validator (v2): scan a dataset, flag files that don't match the
NAME_SERIAL_CATEGORY convention, and one-click **Auto-format** them to conform (rename
image + caption sidecars on disk). A per-file rename box remains for stragglers. On Done,
write the caption roster from the filenames. Replaces the AI Name Cast screen."""
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from core import naming
from core.dataset_manager import scan_folder


class NameValidateView(QDialog):
    characters_changed = Signal()

    def __init__(self, folder, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Validate Names")
        self.setMinimumSize(760, 560)
        self._folder = folder or ""
        self._invalid_boxes = {}
        self._build_ui()
        self._refresh()

    def _image_names(self):
        return [Path(d["image_path"]).name for d in scan_folder(self._folder)]

    def _build_ui(self):
        root = QVBoxLayout(self)
        self._header = QLabel("")
        self._header.setStyleSheet("color:#d4af37; font-size:14px; font-weight:700;")
        root.addWidget(self._header)
        howto = QLabel(
            "Files must be named  NAME_SERIAL_CATEGORY  (e.g. Homer-Marge_004_Character.png; "
            "multiple subjects joined by '-'). Pick the category and hit Auto-format to fix "
            "everything, then Done. Training files are disposable, so renaming is safe.")
        howto.setWordWrap(True)
        howto.setStyleSheet("color:#9a9aa2; font-size:12px;")
        root.addWidget(howto)

        cat_row = QHBoxLayout()
        cat_row.addWidget(QLabel("Project category:"))
        self._cat_combo = QComboBox()
        self._cat_combo.addItems(list(naming.CATEGORIES))
        cat_row.addWidget(self._cat_combo)
        self._auto_btn = QPushButton("⚙  Auto-format all")
        self._auto_btn.setObjectName("btn_primary")
        self._auto_btn.clicked.connect(self._auto_format)
        cat_row.addWidget(self._auto_btn)
        cat_row.addStretch()
        root.addLayout(cat_row)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._host = QWidget()
        self._rows = QVBoxLayout(self._host)
        self._rows.setAlignment(Qt.AlignTop)
        self._scroll.setWidget(self._host)
        root.addWidget(self._scroll, 1)

        foot = QHBoxLayout()
        foot.addStretch()
        done = QPushButton("✓ Done")
        done.setObjectName("btn_primary")
        done.clicked.connect(self._done)
        foot.addWidget(done)
        root.addLayout(foot)

    def _clear_rows(self):
        while self._rows.count():
            item = self._rows.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _add_row(self, layout):
        w = QWidget()
        w.setLayout(layout)
        self._rows.addWidget(w)

    def _refresh(self):
        self._clear_rows()
        self._invalid_boxes = {}
        res = naming.validate_folder(self._image_names())
        cat = res["category"]
        if cat in naming.CATEGORIES:
            self._cat_combo.setCurrentText(cat)
        self._header.setText(
            f"Category: {cat or '—'} — {len(res['valid'])} valid, {len(res['invalid'])} to fix")
        for name in res["valid"]:
            row = QHBoxLayout()
            ok = QLabel(f"✓  {name}")
            ok.setStyleSheet("color:#7ed957;")
            row.addWidget(ok)
            row.addStretch()
            self._add_row(row)
        for d in res["invalid"]:
            row = QHBoxLayout()
            bad = QLabel(f"✗  {d['name']}   ({d['reason']})")
            bad.setStyleSheet("color:#d9534f;")
            row.addWidget(bad)
            row.addStretch()
            box = QLineEdit(d["name"])
            box.setMinimumWidth(240)
            self._invalid_boxes[d["name"]] = box
            row.addWidget(box)
            btn = QPushButton("Rename")
            btn.clicked.connect(lambda _c=False, n=d["name"]: self._do_rename(n))
            row.addWidget(btn)
            self._add_row(row)

    def _auto_format(self):
        try:
            naming.auto_format(self._folder, self._image_names(), self._cat_combo.currentText())
        except Exception as exc:  # noqa: BLE001 - surfaced to the user; files are disposable
            QMessageBox.warning(self, "Auto-format failed", str(exc))
        self._refresh()

    def _do_rename(self, old_name):
        box = self._invalid_boxes.get(old_name)
        if box is None:
            return
        new_name = box.text().strip()
        if not new_name or new_name == old_name:
            return
        try:
            naming.rename_image(self._folder, old_name, new_name)
        except (FileExistsError, FileNotFoundError) as exc:
            QMessageBox.warning(self, "Rename failed", f"{old_name}: {exc}")
            return
        self._refresh()

    def _done(self):
        naming.write_characters_from_names(self._folder, self._image_names())
        self.characters_changed.emit()
        self.accept()
