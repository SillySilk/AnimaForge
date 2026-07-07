"""Machine-readable batch status side output (batch_status.json).

Mirrors a BatchRunner's signals into an atomically-rewritten JSON file so
external tools (e.g. Comic Studio) can watch training progress. Nothing in
AnimaForge reads this file — it is write-only here, and both the GUI batch
tab and the --run-batch headless runner emit it, so the two modes are
indistinguishable to a watcher.

Schema: a JSON list, one entry per queued run:
    {"lora_name", "state": "queued"|"running"|"done"|"failed",
     "current_step", "target_steps", "started_at", "finished_at",
     "lora_path", "error_tail"}
"""
import json
import os
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject

from core.paths import run_output_dir
from core.settings import AppSettings
from core.step_calculator import calculate_training_params

_ERROR_TAIL_LINES = 15


def default_status_path() -> str:
    """`<output dir>/batch_status.json`, matching where finished runs land."""
    out = ""
    try:
        out = AppSettings().get("output_dir") or ""
    except Exception:
        pass
    return str(Path(out or "output") / "batch_status.json")


class StatusWriter(QObject):
    """Attach to a (Batch)Runner and persist every state change atomically."""

    def __init__(self, runner, runs, path: str, parent=None):
        super().__init__(parent)
        self._runs = runs
        self._path = Path(path)
        self._steps = {i: 0 for i in range(len(runs))}
        self._started = {}
        self._finished = {}
        self._tails = {}
        self._log_ring = deque(maxlen=_ERROR_TAIL_LINES)
        self._active_idx = None

        runner.run_started.connect(self._on_started)
        runner.progress_updated.connect(self._on_progress)
        runner.run_finished.connect(self._on_finished)
        runner.batch_finished.connect(self._write)
        runner.log_line.connect(self._on_log)
        self._write()

    # -- signal handlers -------------------------------------------------

    def _on_started(self, idx: int):
        self._active_idx = idx
        self._log_ring.clear()
        self._started[idx] = datetime.now().isoformat(timespec="seconds")
        self._write()

    def _on_progress(self, idx: int, step: int):
        self._steps[idx] = int(step)
        self._write()

    def _on_finished(self, idx: int, success: bool):
        self._finished[idx] = datetime.now().isoformat(timespec="seconds")
        if not success:
            self._tails[idx] = "\n".join(self._log_ring)
        self._active_idx = None
        self._write()

    def _on_log(self, line: str):
        self._log_ring.append(line)

    # -- persistence -----------------------------------------------------

    def _entry(self, idx: int, run) -> dict:
        try:
            total = calculate_training_params(
                run.image_count, target_steps=run.target_steps)["total_steps"]
        except Exception:
            total = run.target_steps
        lora_path = None
        if run.status == "done":
            lora_path = str(Path(run_output_dir(run.output_dir, run.lora_name))
                            / f"{run.lora_name}.safetensors")
        return {
            "lora_name": run.lora_name,
            "state": run.status,
            "current_step": self._steps.get(idx, 0),
            "target_steps": total,
            "started_at": self._started.get(idx),
            "finished_at": self._finished.get(idx),
            "lora_path": lora_path,
            "error_tail": self._tails.get(idx),
        }

    def _write(self, *args):
        data = [self._entry(i, r) for i, r in enumerate(self._runs)]
        text = json.dumps(data, indent=2)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, self._path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
