"""Advanced settings editor: every setting the model's engine will receive, editable.

A plain widget (hosted in a ForgeModal from Home). It shows the run's base config as
dotted paths, lets any value be changed or a new setting added, and reports the
differences from the base as the run's overrides (core/config_overrides.py).
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from core import config_overrides

_GOLD = QColor("#d4af37")
_ADDED = "(added)"
_SETTING, _VALUE, _PRESET = range(3)


class AdvancedSettingsPanel(QWidget):
    def __init__(self, base: dict, overrides: dict, locked=(), parent=None):
        super().__init__(parent)
        self._base = dict(base)
        self._locked = tuple(locked or ())
        self._filling = False

        self.search = QLineEdit()
        self.search.setPlaceholderText("Find a setting")
        self.search.textChanged.connect(self._filter)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Setting", "Value", "Preset value"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked
                                   | QAbstractItemView.EditKeyPressed
                                   | QAbstractItemView.AnyKeyPressed)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(_SETTING, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(_VALUE, QHeaderView.Stretch)
        header.setSectionResizeMode(_PRESET, QHeaderView.Interactive)
        self.table.setMinimumHeight(360)
        self.table.itemChanged.connect(self._on_item_changed)

        self.add_path = QLineEdit()
        self.add_path.setPlaceholderText("New setting, e.g. training_arguments.noise_offset")
        self.add_value = QLineEdit()
        self.add_value.setPlaceholderText("Value")
        self.add_button = QPushButton("Add setting")
        self.add_button.setObjectName("af_btn_ghost")
        self.add_button.clicked.connect(self._add)
        self.reset_selected_button = QPushButton("Reset selected")
        self.reset_selected_button.setObjectName("af_btn_ghost")
        self.reset_selected_button.clicked.connect(self._reset_selected)
        self.reset_all_button = QPushButton("Reset all")
        self.reset_all_button.setObjectName("af_btn_ghost")
        self.reset_all_button.clicked.connect(self._reset_all)

        add_row = QHBoxLayout()
        add_row.addWidget(self.add_path, 3)
        add_row.addWidget(self.add_value, 2)
        add_row.addWidget(self.add_button)
        self.add_error = QLabel("")
        self.add_error.setObjectName("af_eyebrow_mute")
        self.add_error.setWordWrap(True)
        self.add_error.setVisible(False)
        reset_row = QHBoxLayout()
        reset_row.addStretch(1)
        reset_row.addWidget(self.reset_selected_button)
        reset_row.addWidget(self.reset_all_button)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.search)
        lay.addWidget(self.table, 1)
        lay.addLayout(add_row)
        lay.addWidget(self.add_error)
        lay.addLayout(reset_row)

        self._fill(overrides or {})

    # ---- public ----

    def overrides(self) -> dict:
        """{path: value} for every row that differs from the base or was added.
        Raises ValueError listing every value that does not fit its setting."""
        out, problems = {}, []
        for r in range(self.table.rowCount()):
            path = self._path(r)
            try:
                value = self._parse_row(r)
            except ValueError as e:
                problems.append(f"{path}: {e}")
                continue
            if self._is_added(r) or value != self._base[path]:
                out[path] = value
        if problems:
            raise ValueError("\n".join(problems))
        return out

    def changed_count(self) -> int:
        return sum(1 for r in range(self.table.rowCount()) if self._row_changed(r))

    # ---- rows ----

    def _fill(self, overrides: dict):
        self._filling = True
        self.table.setRowCount(0)
        for path in sorted(self._base):
            shown = overrides.get(path, self._base[path])
            self._append(path, config_overrides.format_value(shown),
                         config_overrides.format_value(self._base[path]))
        for path in sorted(set(overrides) - set(self._base)):
            self._append(path, config_overrides.format_value(overrides[path]), _ADDED)
        self._filling = False
        for r in range(self.table.rowCount()):
            self._mark(r)
        self._filter(self.search.text())

    def _append(self, path: str, value_text: str, preset_text: str) -> int:
        r = self.table.rowCount()
        self.table.insertRow(r)
        fixed = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        setting = QTableWidgetItem(path)
        setting.setFlags(fixed)
        preset = QTableWidgetItem(preset_text)
        preset.setFlags(fixed)
        value = QTableWidgetItem(value_text)
        if config_overrides.is_locked(path, self._locked):
            value.setFlags(fixed)
            value.setToolTip("Set by the run itself (name, folders or step count) — "
                             "change those on Home.")
        else:
            value.setFlags(fixed | Qt.ItemIsEditable)
        self.table.setItem(r, _SETTING, setting)
        self.table.setItem(r, _VALUE, value)
        self.table.setItem(r, _PRESET, preset)
        return r

    def _path(self, r: int) -> str:
        return self.table.item(r, _SETTING).text()

    def _is_added(self, r: int) -> bool:
        return self._path(r) not in self._base

    def _parse_row(self, r: int):
        like = None if self._is_added(r) else self._base[self._path(r)]
        return config_overrides.parse(self.table.item(r, _VALUE).text(), like)

    def _row_changed(self, r: int) -> bool:
        if self._is_added(r):
            return True
        try:
            return self._parse_row(r) != self._base[self._path(r)]
        except ValueError:
            return True

    def _mark(self, r: int):
        changed = self._row_changed(r)
        self._filling = True
        for col in (_SETTING, _VALUE):
            item = self.table.item(r, col)
            font = item.font()
            font.setBold(changed)
            item.setFont(font)
            item.setForeground(QBrush(_GOLD) if changed else QBrush())
            if changed:
                item.setToolTip("Changed from the preset")
            elif not config_overrides.is_locked(self._path(r), self._locked):
                item.setToolTip("")
        self._filling = False

    def _on_item_changed(self, item: QTableWidgetItem):
        if self._filling or item.column() != _VALUE:
            return
        self._mark(item.row())

    # ---- actions ----

    def _filter(self, text: str):
        needle = (text or "").strip().lower()
        for r in range(self.table.rowCount()):
            self.table.setRowHidden(r, bool(needle) and needle not in self._path(r).lower())

    def _find(self, path: str) -> int:
        for r in range(self.table.rowCount()):
            if self._path(r) == path:
                return r
        return -1

    def _add(self):
        path = self.add_path.text().strip()
        problem = config_overrides.check_new_path(path, self._base)
        if not problem and config_overrides.is_locked(path, self._locked):
            problem = f"{path} is set by the run itself; change it on Home."
        self.add_error.setText(problem)
        self.add_error.setVisible(bool(problem))
        if problem:
            self.add_path.setFocus()
            return
        existing = self._find(path)
        if existing >= 0:
            self.search.clear()
            self.table.selectRow(existing)
            self.table.scrollToItem(self.table.item(existing, _SETTING))
            return
        self._filling = True
        r = self._append(path, self.add_value.text(), _ADDED)
        self._filling = False
        self._mark(r)
        self._filter(self.search.text())
        self.add_path.clear()
        self.add_value.clear()

    def _reset_selected(self):
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows:
            if self._is_added(r):
                self.table.removeRow(r)
                continue
            self.table.item(r, _VALUE).setText(
                config_overrides.format_value(self._base[self._path(r)]))

    def _reset_all(self):
        self._fill({})
