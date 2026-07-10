"""Headless batch runner (`main.py --run-batch`).

Processes the batch queue exactly as the GUI Batch tab does — same
BatchRunner, same one-GPU-job-at-a-time serialization — but under a
QCoreApplication with no windows, fonts, or ui.* imports. Progress is
mirrored to batch_status.json (core.batch_status) so external tools such
as Comic Studio can supervise the run; the queue file is rewritten with
final statuses so the GUI shows the results on next launch.
"""
import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication

from core.batch import BatchRunner, load_queue, save_queue, DONE
from core.batch_status import StatusWriter, default_status_path


def run_headless(queue_path: str | None = None, status_path: str | None = None) -> int:
    """Run every queued job sequentially. Returns 0 iff all runs finished DONE."""
    queue_path = queue_path or "batch_queue.json"
    runs = load_queue(queue_path)
    if not runs:
        print(f"[headless] no runs in queue '{queue_path}' — nothing to do.")
        return 1

    # 'ask' is a UI-only prompt; there is no GUI here to answer it. Degrade to
    # 'keep' before the runner starts so a queued 'ask' run never blocks and
    # never destroys captions that already exist on disk.
    from core import caption_policy as cp
    for r in runs:
        if r.caption_policy == cp.ASK:
            r.caption_policy = cp.KEEP
            print(f"[headless] '{r.lora_name}': policy 'ask' has no GUI here — "
                  f"using 'keep' (existing captions are never destroyed).", flush=True)

    app = QCoreApplication.instance() or QCoreApplication(sys.argv)
    runner = BatchRunner()
    writer = StatusWriter(runner, runs, status_path or default_status_path())

    def _safe_print(line: str) -> None:
        # sd-scripts logs Japanese; a cp1252 console must never crash the
        # passthrough (UnicodeEncodeError spam observed live 2026-07-07).
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(line.encode(enc, errors="replace").decode(enc), flush=True)

    runner.log_line.connect(_safe_print)
    runner.run_finished.connect(lambda _i, _ok: save_queue(queue_path, runs))

    done = {"flag": False}

    def _finish():
        done["flag"] = True
        app.quit()

    runner.batch_finished.connect(_finish)
    runner.start(runs, continue_on_error=True)
    if not done["flag"]:          # batch may finish synchronously on config errors
        app.exec()

    save_queue(queue_path, runs)
    del writer
    return 0 if all(r.status == DONE for r in runs) else 1
