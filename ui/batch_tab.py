from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
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
        self._refresh_queue()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add_run(self, run_definition):
        self._runs.append(run_definition)
        self._persist()
        self._refresh_queue()
        self._log.append(f"[Batch] Queued '{run_definition.lora_name}' "
                         f"({run_definition.image_count} images, {run_definition.target_steps} steps).")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(10)

        title = QLabel("The Line")
        title.setObjectName("af_screen_title")
        layout.addWidget(title)

        self._summary_label = QLabel("")
        self._summary_label.setObjectName("af_eyebrow_mute")
        layout.addWidget(self._summary_label)

        # Queue = a scrollable list of run rows (added from Home's 'Add to Batch').
        self._queue_scroll = QScrollArea()
        self._queue_scroll.setWidgetResizable(True)
        self._queue_scroll.setFrameShape(QFrame.NoFrame)
        self._queue_host = QWidget()
        self._queue_layout = QVBoxLayout(self._queue_host)
        self._queue_layout.setContentsMargins(0, 0, 0, 0)
        self._queue_layout.setSpacing(8)
        self._queue_layout.setAlignment(Qt.AlignTop)
        self._queue_scroll.setWidget(self._queue_host)
        layout.addWidget(self._queue_scroll, 1)

        footer = QHBoxLayout()
        foot = QLabel("Queue saved — survives a restart. Runs process top to bottom.")
        foot.setObjectName("af_eyebrow_mute")
        footer.addWidget(foot)
        footer.addStretch()
        clear_btn = QPushButton("Clear Queue")
        clear_btn.setObjectName("btn_danger")
        clear_btn.clicked.connect(self._clear)
        footer.addWidget(clear_btn)
        layout.addLayout(footer)

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

    def _refresh_queue(self):
        while self._queue_layout.count():
            item = self._queue_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if not self._runs:
            empty = QLabel("The line's empty. Add a set to get forging (Home → Add to Batch).")
            empty.setObjectName("af_marker")
            empty.setAlignment(Qt.AlignCenter)
            self._queue_layout.addWidget(empty)
        else:
            for i, r in enumerate(self._runs):
                self._queue_layout.addWidget(self._make_run_row(i, r))
        total = len(self._runs)
        done = sum(1 for r in self._runs if r.status == DONE)
        running = sum(1 for r in self._runs if r.status == RUNNING)
        queued = total - done - running
        self._summary_label.setText(
            f"{total} in line · {done} done · {running} running · {queued} queued")

    def _make_run_row(self, i: int, r) -> QFrame:
        color = _STATUS_COLOR.get(r.status, "#8a8a93")
        card = QFrame()
        card.setObjectName("af_card")
        row = QHBoxLayout(card)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(12)
        pos = QLabel(str(i + 1))
        pos.setObjectName("af_eyebrow_mute")
        pos.setFixedWidth(18)
        row.addWidget(pos)
        dot = QLabel("●")
        dot.setStyleSheet(f"color:{color}; font-size:12px;")
        row.addWidget(dot)
        name = QLabel(r.lora_name)
        name.setObjectName("af_display_gold")
        row.addWidget(name)
        chips = QLabel(f"{Path(r.dataset_folder).name} · {r.target_steps} steps")
        chips.setObjectName("af_eyebrow_mute")
        row.addWidget(chips)
        row.addStretch()
        pill = QLabel(r.status.upper())
        pill.setStyleSheet(
            f"color:{color}; border:1px solid {color}; border-radius:9px; "
            "padding:2px 10px; font-size:10px; font-weight:700;")
        row.addWidget(pill)
        for text, slot in (("↑", lambda _c=False, idx=i: self._move_up(idx)),
                           ("↓", lambda _c=False, idx=i: self._move_down(idx)),
                           ("✕", lambda _c=False, idx=i: self._remove(idx))):
            b = QPushButton(text)
            b.setObjectName("af_icon_btn")
            b.setFixedSize(26, 26)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(slot)
            row.addWidget(b)
        return card

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def _guard_running(self) -> bool:
        if self._runner.is_running():
            QMessageBox.information(self, "Batch Running", "Stop the batch before editing the queue.")
            return True
        return False

    def _move_up(self, i: int):
        if self._guard_running():
            return
        if i > 0:
            self._runs[i - 1], self._runs[i] = self._runs[i], self._runs[i - 1]
            self._persist(); self._refresh_queue()

    def _move_down(self, i: int):
        if self._guard_running():
            return
        if 0 <= i < len(self._runs) - 1:
            self._runs[i + 1], self._runs[i] = self._runs[i], self._runs[i + 1]
            self._persist(); self._refresh_queue()

    def _remove(self, i: int):
        if self._guard_running():
            return
        if 0 <= i < len(self._runs):
            del self._runs[i]
            self._persist(); self._refresh_queue()

    def _clear(self):
        if self._guard_running():
            return
        if self._runs and QMessageBox.question(self, "Clear Queue", "Remove all queued runs?") == QMessageBox.Yes:
            self._runs.clear()
            self._persist(); self._refresh_queue()

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
        self._refresh_queue()

    def _on_run_finished(self, idx: int, success: bool):
        self._persist()
        self._refresh_queue()

    def _on_batch_finished(self):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._overall_label.setText("Batch complete.")
        self._log.append("[Batch] Finished.")
        self._persist()
        self._refresh_queue()

    def _on_log(self, line: str):
        self._log.append(line)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())
