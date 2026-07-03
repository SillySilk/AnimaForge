"""Named training "sets" (presets) + crash-recovery marker.

A set is one core.batch.RunDefinition serialized to JSON in <root>/sets/{name}.json.
A run-in-progress is recorded in <root>/sets/_last_run.json; interrupted_run() reports
it back when a saved state exists but the final {name}.safetensors does not.
"""
import json
import re
import shutil
from pathlib import Path

from core.batch import RunDefinition
from core.state_utils import find_saved_state

_MARKER = "_last_run"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sets_dir(root=None) -> Path:
    base = Path(root) if root else _project_root()
    d = base / "sets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9 ._-]", "_", (name or "").strip())
    return s or "set"


def set_to_markdown(rd: RunDefinition, name: str) -> str:
    """Human-readable summary of a set (glance-only; the JSON is the reload source)."""
    lines = [f"# {name}", ""]

    def add(label, value):
        if value not in (None, "", []):
            lines.append(f"- **{label}:** {value}")

    add("Dataset folder", rd.dataset_folder)
    add("Trigger word", rd.trigger_word)
    add("Image count", rd.image_count)
    add("Target steps", rd.target_steps)
    add("Network dim / alpha", f"{rd.network_dim} / {rd.network_alpha}")
    add("Optimizer", rd.optimizer)
    add("Train text encoder", rd.train_text_encoder)
    add("Aspect-ratio buckets", rd.enable_bucket)
    add("Save state", rd.save_state)
    add("Checkpoint every (steps)", rd.save_every_n_steps)
    add("Subject type", rd.subject_type)
    add("Sample previews", rd.sample_enabled)
    add("Resume from weights", rd.network_weights)
    add("Output dir", rd.output_dir)
    return "\n".join(lines) + "\n"


def set_save_decision(name: str, existing) -> str:
    """Decide what Save should do: 'empty' (no name), 'exists' (overwrite), or 'ok'."""
    n = (name or "").strip()
    if not n:
        return "empty"
    return "exists" if n in set(existing or []) else "ok"


def list_sets(root=None):
    names = []
    for p in sets_dir(root).glob("*.json"):
        if p.stem == _MARKER:
            continue
        names.append(p.stem)
    return sorted(names)


def save_set(name: str, rd: RunDefinition, root=None) -> Path:
    d = sets_dir(root)
    stem = _safe(name)
    json_path = d / f"{stem}.json"
    json_path.write_text(json.dumps(rd.to_dict(), indent=2), encoding="utf-8")
    (d / f"{stem}.md").write_text(set_to_markdown(rd, name), encoding="utf-8")
    return json_path


def _read_rd(p: Path):
    try:
        return RunDefinition.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def load_set(name: str, root=None):
    p = sets_dir(root) / f"{_safe(name)}.json"
    return _read_rd(p) if p.is_file() else None


def delete_set(name: str, root=None) -> None:
    d = sets_dir(root)
    stem = _safe(name)
    for ext in (".json", ".md"):
        p = d / f"{stem}{ext}"
        if p.is_file():
            p.unlink()
    snaps = d / stem
    if snaps.is_dir():
        shutil.rmtree(snaps, ignore_errors=True)


# ---- caption snapshots (project autosave) ----
# Autosave writes two restore points per run under sets/<name>/:
#   captions-captioned/  — raw caption passes done (tag/describe/refine)
#   captions-processed/  — combine done, final .txt built
CAPTION_EXTS = (".txt", ".tags", ".nl")
SNAPSHOT_STAGES = ("captioned", "processed")


def _snapshot_dir(name: str, stage: str, root=None) -> Path:
    return sets_dir(root) / _safe(name) / f"captions-{stage}"


def snapshot_captions(dataset_folder: str, name: str, stage: str, root=None) -> int:
    """Copy the dataset's caption files (.txt/.tags/.nl — never images) into the
    set's snapshot folder for `stage`, replacing any previous snapshot of that
    stage. Returns the number of files copied."""
    src = Path(dataset_folder or "")
    if not src.is_dir():
        return 0
    dst = _snapshot_dir(name, stage, root)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.iterdir()):
        if p.is_file() and p.suffix.lower() in CAPTION_EXTS:
            shutil.copy2(p, dst / p.name)
            n += 1
    return n


def list_caption_snapshots(name: str, root=None) -> dict:
    """{stage: file_count} for the stages that have a saved snapshot."""
    out = {}
    for stage in SNAPSHOT_STAGES:
        d = _snapshot_dir(name, stage, root)
        if d.is_dir():
            count = sum(1 for p in d.iterdir() if p.is_file())
            if count:
                out[stage] = count
    return out


def restore_captions(dataset_folder: str, name: str, stage: str, root=None) -> int:
    """Copy a snapshot's caption files back into the dataset folder (overwriting).
    Returns the number of files restored."""
    src = _snapshot_dir(name, stage, root)
    dst = Path(dataset_folder or "")
    if not src.is_dir() or not dst.is_dir():
        return 0
    n = 0
    for p in sorted(src.iterdir()):
        if p.is_file():
            shutil.copy2(p, dst / p.name)
            n += 1
    return n


def autosave_project(name: str, rd: RunDefinition, dataset_folder: str, stage: str,
                     root=None):
    """Save the Set under the LoRA name + snapshot its captions. Never raises —
    returns (ok, human message) so callers can drop it on the status bar."""
    try:
        save_set(name, rd, root)
        n = snapshot_captions(dataset_folder, name, stage, root)
        return True, f"Project autosaved as '{name}' ({stage} — {n} caption file(s))."
    except OSError as e:
        return False, f"Project autosave failed: {e}"


def mark_run_active(rd: RunDefinition, root=None) -> None:
    (sets_dir(root) / f"{_MARKER}.json").write_text(
        json.dumps(rd.to_dict(), indent=2), encoding="utf-8")


def clear_active_run(root=None) -> None:
    p = sets_dir(root) / f"{_MARKER}.json"
    if p.is_file():
        p.unlink()


def interrupted_run(root=None):
    p = sets_dir(root) / f"{_MARKER}.json"
    if not p.is_file():
        return None
    rd = _read_rd(p)
    if rd is None or not rd.output_dir or not rd.lora_name:
        return None
    if find_saved_state(rd.output_dir, rd.lora_name) is None:
        return None
    final = Path(rd.output_dir) / f"{rd.lora_name}.safetensors"
    if final.is_file():
        return None
    return rd
