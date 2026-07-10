"""A run manifest written beside the LoRA, so the output folder is self-describing.

`output/<lora>/run.json` carries the whole RunDefinition plus live progress. Copy the
folder anywhere, or start a second run on top of an old one, and resume still works --
unlike the single global sets/_last_run.json marker, which only ever remembers one.

A falsy `run_dir` (None or "") is guarded everywhere below: `Path("")` silently
normalizes to `Path(".")`, so an unguarded `_path()` would resolve into the process
CWD instead of failing loudly. Readers return their empty value (`{}`); writers
become no-ops. Never let a write escape into the CWD.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from core.batch import RunDefinition
from core.state_utils import find_saved_state

RUN_FILE = "run.json"
RUNNING, INTERRUPTED, DONE, FAILED = "running", "interrupted", "done", "failed"
VERSION = 1


def _path(run_dir: str):
    """The manifest path for `run_dir`, or None when `run_dir` is falsy.

    None must be checked explicitly by every caller below -- it means "no run
    directory was given", not "use the current directory".
    """
    if not run_dir:
        return None
    return Path(run_dir) / RUN_FILE


def load(run_dir: str) -> dict:
    """The manifest dict, or {} when absent, corrupt, or `run_dir` is falsy. Never raises."""
    p = _path(run_dir)
    if p is None:
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(run_dir: str, data: dict) -> None:
    """Write `data` to the manifest. No-op when `run_dir` is falsy."""
    p = _path(run_dir)
    if p is None:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_start(run_dir: str, rd: RunDefinition) -> None:
    """Start a fresh manifest for this run. No-op when `run_dir` is falsy."""
    _save(run_dir, {
        "version": VERSION,
        "run": rd.to_dict(),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": RUNNING,
        "epochs_done": 0,
        "global_step": 0,
        "last_state": "",
    })


def update(run_dir: str, **kw) -> None:
    """Merge `kw` into the existing manifest.

    No-op when there is no manifest to update -- either `run_dir` is falsy, or
    write_start() was never called for it. update() only ever amends an existing
    manifest; it never conjures one into existence.
    """
    d = load(run_dir)
    if not d:
        return
    d.update(kw)
    _save(run_dir, d)


def mark(run_dir: str, status: str) -> None:
    """Set the manifest's status: "running" | "interrupted" | "done" | "failed"."""
    update(run_dir, status=status)


def find_resumable(output_dir: str):
    """The newest run that stopped mid-flight and can still be resumed, or None.

    Resumable means: status running/interrupted, a saved sd-scripts state exists, and
    the final .safetensors was never written.

    `output_dir` is the PARENT directory holding one subfolder per LoRA
    (output/<lora>/run.json) -- pass it straight through to find_saved_state(), which
    calls run_output_dir(output_dir, lora_name) internally to relocate the per-run
    folder itself. Passing a run's own directory instead only "works" via
    find_saved_state's legacy flat-layout fallback -- an accident, not the contract.

    A falsy `output_dir` returns None outright rather than falling through to
    Path("").is_dir(), which is True (it normalizes to the process CWD) and would
    otherwise scan whatever happens to be in the working directory.
    """
    if not output_dir:
        return None
    base = Path(output_dir)
    if not base.is_dir():
        return None
    best = None
    for d in base.iterdir():
        if not d.is_dir():
            continue
        data = load(str(d))
        if data.get("status") not in (RUNNING, INTERRUPTED):
            continue
        try:
            rd = RunDefinition.from_dict(data["run"])
        except (KeyError, TypeError):
            continue
        if (d / f"{rd.lora_name}.safetensors").is_file():
            continue
        if find_saved_state(output_dir, rd.lora_name) is None:
            continue
        if best is None or data.get("started", "") > best[0]:
            best = (data.get("started", ""), rd)
    return best[1] if best else None
