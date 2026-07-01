"""Named training "sets" (presets) + crash-recovery marker.

A set is one core.batch.RunDefinition serialized to JSON in <root>/sets/{name}.json.
A run-in-progress is recorded in <root>/sets/_last_run.json; interrupted_run() reports
it back when a saved state exists but the final {name}.safetensors does not.
"""
import json
import re
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
