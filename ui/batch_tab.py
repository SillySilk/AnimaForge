from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.batch import BatchRunner, save_queue, load_queue, RUNNING, DONE, FAILED

QUEUE_PATH = str(Path(__file__).resolve().parents[1] / "batch_queue.json")

_STATUS_COLOR = {
    "queued": "#8a8a93",
    RUNNING: "#f4d160",
    DONE: "#d4af37",
    FAILED: "#d9534f",
}


class BatchTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._runs = load_queue(QUEUE_PATH)
        self._runner = BatchRunner(self)
        self._runner.run_started.connect(self._on_run_started)
        self._runner.run_finished.connect(self._on_run_finished)
        self._runner.batch_finished.connect(self._on_batch_finished)
        self._runner.log_line.connect(self._on_log)
        self._build_ui()
        self._refresh_table()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add_run(self, run_definition):
        self._runs.append(run_definition)
        self._persist()
        self._refresh_table()
        self._log.append(f"[Batch] Queued '{run_definition.lora_name}' "
                         f"({run_definition.image_count} images, {run_definition.target_steps} steps).")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(10)

        title = QLabel("Batch Queue")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #d4af37;")
        layout.addWidget(title)

        hint = QLabel("Queue runs from the front page ('Add to Batch' on Home), then Start — "
                      "they process one after another, unattended. The queue is saved and "
                      "survives a restart.")
        hint.setObjectName("label_field")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["LoRA Name", "Dataset", "Steps", "Status"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self._table)

        # Queue-management buttons
        mgmt = QHBoxLayout()
        for label, slot in (("↑ Up", self._move_up), ("↓ Down", self._move_down),
                            ("✖ Remove", self._remove), ("Clear", self._clear)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            mgmt.addWidget(b)
        mgmt.addStretch()
        layout.addLayout(mgmt)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #2a2a1e;")
        layout.addWidget(sep)

        # Run controls
        run_row = QHBoxLayout()
        self._start_btn = QPushButton("▶  Start Batch")
        self._start_btn.setObjectName("btn_start")
        self._start_btn.clicked.connect(self._start)
        self._stop_btn = QPushButton("■  Stop Batch")
        self._stop_btn.setObjectName("btn_stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        run_row.addWidget(self._start_btn)
        run_row.addWidget(self._stop_btn)
        self._overall_label = QLabel("")
        self._overall_label.setStyleSheet("color: #8a8a93; font-size: 12px;")
        run_row.addWidget(self._overall_label)
        run_row.addStretch()
        layout.addLayout(run_row)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setObjectName("log_output")
        self._log.setFixedHeight(140)
        self._log.setStyleSheet("font-family: Consolas, monospace; font-size: 12px; "
                                "background-color: #0c0b0a; color: #c6c6ce; "
                                "border: 1px solid #2a2a1e; border-radius: 5px;")
        layout.addWidget(self._log)

    def _refresh_table(self):
        self._table.setRowCount(len(self._runs))
        for i, r in enumerate(self._runs):
            self._table.setItem(i, 0, QTableWidgetItem(r.lora_name))
            self._table.setItem(i, 1, QTableWidgetItem(Path(r.dataset_folder).name or r.dataset_folder))
            self._table.setItem(i, 2, QTableWidgetItem(str(r.target_steps)))
            status_item = QTableWidgetItem(r.status)
            from PySide6.QtGui import QColor
            status_item.setForeground(QColor(_STATUS_COLOR.get(r.status, "#8a8a93")))
            self._table.setItem(i, 3, status_item)

    def _selected_row(self) -> int:
        rows = self._table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def _guard_running(self) -> bool:
        if self._runner.is_running():
            QMessageBox.information(self, "Batch Running", "Stop the batch before editing the queue.")
            return True
        return False

    def _move_up(self):
        if self._guard_running():
            return
        i = self._selected_row()
        if i > 0:
            self._runs[i - 1], self._runs[i] = self._runs[i], self._runs[i - 1]
            self._persist(); self._refresh_table()
            self._table.selectRow(i - 1)

    def _move_down(self):
        if self._guard_running():
            return
        i = self._selected_row()
        if 0 <= i < len(self._runs) - 1:
            self._runs[i + 1], self._runs[i] = self._runs[i], self._runs[i + 1]
            self._persist(); self._refresh_table()
            self._table.selectRow(i + 1)

    def _remove(self):
        if self._guard_running():
            return
        i = self._selected_row()
        if 0 <= i < len(self._runs):
            del self._runs[i]
            self._persist(); self._refresh_table()

    def _clear(self):
        if self._guard_running():
            return
        if self._runs and QMessageBox.question(self, "Clear Queue", "Remove all queued runs?") == QMessageBox.Yes:
            self._runs.clear()
            self._persist(); self._refresh_table()

    def _persist(self):
        try:
            save_queue(QUEUE_PATH, self._runs)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Run control
    # ------------------------------------------------------------------

    def _start(self):
        pending = [r for r in self._runs if r.status != DONE]
        if not pending:
            QMessageBox.information(self, "Nothing to Run", "The queue has no pending runs.")
            return
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._log.clear()
        self._log.append(f"[Batch] Starting {len(pending)} run(s)…")
        self._runner.start(self._runs, continue_on_error=True)

    def _stop(self):
        self._runner.stop()

    def _on_run_started(self, idx: int):
        self._overall_label.setText(f"Running {idx + 1} / {len(self._runs)}")
        self._refresh_table()

    def _on_run_finished(self, idx: int, success: bool):
        self._persist()
        self._refresh_table()

    def _on_batch_finished(self):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._overall_label.setText("Batch complete.")
        self._log.append("[Batch] Finished.")
        self._persist()
        self._refresh_table()

    def _on_log(self, line: str):
        self._log.append(line)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())
