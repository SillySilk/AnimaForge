"""Every training run in the output folder, how far it got, and whether it can go on.

The Train tab lists these with "Train further" and a delete button. A run can train
further when its saved trainer state is still on disk and its dataset folder still
exists:

  Kohya       the newest `<name>-*-state` folder; its train_state.json says how many
              steps are behind it.
  OneTrainer  the newest workspace/backup/<...>/ folder; its meta.json says the same.
              OneTrainer writes one before it saves the final file
              (backup_before_save), so every finished run has one.

Deleting a run removes everything the trainer made in its folder -- epoch and step
saves, saved states, backups, caches, previews, logs, configs, run.json -- and keeps
only the finished LoRA and its trained-words sidecar. The dataset images and their
caption .txt files live in the dataset folder and are never touched.
"""
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from core import models, run_manifest
from core.batch import RunDefinition
from core.paths import sanitize_name
from core.state_utils import find_saved_state

TRAINED_WORDS_SUFFIX = ".trained-words.txt"
# The line metadata.stamp writes for the trigger in the trained-words sidecar.
_TRIGGER_MARK = "the set's trigger word"


@dataclass
class PastRun:
    name: str
    run_dir: str
    engine: str                # models.KOHYA | models.ONETRAINER
    steps_done: int            # steps behind the newest saved state (0 = none)
    epochs_done: int
    status: str                # run.json status, "" when the run has none
    rd: "RunDefinition | None"
    resume_from: str           # the state/backup folder training continues from
    has_lora: bool             # the finished <name>.safetensors is on disk
    problem: str = ""          # why it cannot train further; "" when it can

    @property
    def can_continue(self) -> bool:
        return not self.problem


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def kohya_progress(state_dir: str) -> tuple:
    """(steps, epochs) recorded in a Kohya state folder's train_state.json."""
    data = _read_json(Path(state_dir) / "train_state.json")
    return int(data.get("current_step") or 0), int(data.get("current_epoch") or 0)


def newest_backup(run_dir: str):
    """The newest OneTrainer backup folder under run_dir/workspace/backup, or None.
    Backup names start with a timestamp, so the last one by name is the newest --
    the same one OneTrainer's continue_last_backup picks up."""
    folder = Path(run_dir) / "workspace" / "backup"
    if not folder.is_dir():
        return None
    found = sorted(p for p in folder.iterdir()
                   if p.is_dir() and (p / "meta.json").is_file())
    return str(found[-1]) if found else None


def backup_progress(backup_dir: str) -> tuple:
    """(global_step, epoch) from a OneTrainer backup's meta.json."""
    prog = _read_json(Path(backup_dir) / "meta.json").get("train_progress") or {}
    return int(prog.get("global_step") or 0), int(prog.get("epoch") or 0)


def _trigger_from_sidecar(run_dir: Path, name: str) -> str:
    try:
        text = (run_dir / f"{name}{TRAINED_WORDS_SUFFIX}").read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        if _TRIGGER_MARK in line:
            return line.split("—")[0].strip()
    return ""


def _onetrainer_rd(run_dir: Path, onetrainer_root: str, output_dir: str):
    """A RunDefinition for a OneTrainer run folder written before runs carried a
    run.json: the dataset folder from its concepts.json, the trigger from its
    trained-words sidecar (the LoRA's name when there is none -- what the run
    itself falls back to), the rest at the Krea 2 defaults it trained with."""
    from core.dataset_manager import scan_folder
    from core.naming import trigger_from_lora_name
    concepts = _read_json_list(run_dir / "concepts.json")
    dataset = next((c.get("path") for c in concepts if c.get("path")), "")
    if not dataset:
        return None
    name = run_dir.name
    count = len(scan_folder(dataset)) if Path(dataset).is_dir() else 0
    return RunDefinition(
        lora_name=name, dataset_folder=dataset, image_count=count,
        trigger_word=_trigger_from_sidecar(run_dir, name) or trigger_from_lora_name(name),
        model="krea2", output_dir=output_dir, onetrainer_root=onetrainer_root)


def _read_json_list(path: Path) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _engine_of(rd) -> str:
    try:
        return models.get(rd.model).engine
    except KeyError:
        return ""


def _read_run(d: Path, output_dir: str, onetrainer_root: str):
    manifest = run_manifest.load(str(d))
    rd = None
    if manifest.get("run"):
        try:
            rd = RunDefinition.from_dict(manifest["run"])
        except (KeyError, TypeError, ValueError):
            rd = None
    onetrainer_folder = (d / "config.json").is_file() and (d / "workspace").is_dir()
    if rd is None and onetrainer_folder:
        rd = _onetrainer_rd(d, onetrainer_root, output_dir)
    if rd is None:
        return None
    engine = _engine_of(rd)
    name = rd.lora_name or d.name
    if engine == models.ONETRAINER:
        resume_from = newest_backup(str(d)) or ""
        steps, epochs = backup_progress(resume_from) if resume_from else (0, 0)
        missing = "no OneTrainer backup was saved"
    elif engine == models.KOHYA:
        resume_from = find_saved_state(output_dir, name) or ""
        steps, epochs = kohya_progress(resume_from) if resume_from else (0, 0)
        if not steps:
            steps = int(manifest.get("global_step") or 0)
        missing = "no saved training state"
    else:
        return None
    if not resume_from:
        problem = missing
    elif not rd.dataset_folder or not Path(rd.dataset_folder).is_dir():
        problem = "dataset folder is gone"
    else:
        problem = ""
    return PastRun(
        name=name, run_dir=str(d), engine=engine, steps_done=steps, epochs_done=epochs,
        status=manifest.get("status", ""), rd=rd, resume_from=resume_from,
        has_lora=(d / f"{sanitize_name(name)}.safetensors").is_file(), problem=problem)


def list_runs(output_dir: str, onetrainer_root: str = "") -> list:
    """Every run folder under output_dir, newest first. [] when the folder is unset
    or missing (a falsy path must never scan the process CWD)."""
    if not output_dir:
        return []
    base = Path(output_dir)
    if not base.is_dir():
        return []
    found = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        run = _read_run(d, output_dir, onetrainer_root)
        if run is not None:
            found.append((d.stat().st_mtime, run))
    found.sort(key=lambda t: t[0], reverse=True)
    return [run for _, run in found]


def kept_files(run: PastRun) -> set:
    """The file names a delete leaves in the run folder."""
    safe = sanitize_name(run.name)
    return {f"{safe}.safetensors", f"{safe}{TRAINED_WORDS_SUFFIX}"}


def delete_training_files(run: PastRun, output_dir: str) -> int:
    """Remove everything the trainer made in run.run_dir except kept_files(run).
    Returns how many entries were removed. The folder itself is removed when
    nothing is left in it. Refuses any folder that is not directly inside
    output_dir."""
    base = Path(output_dir).resolve()
    target = Path(run.run_dir).resolve()
    if not output_dir or target.parent != base or not target.is_dir():
        raise ValueError(f"{run.run_dir} is not a run folder inside {output_dir}")
    keep = kept_files(run)
    removed = 0
    for entry in target.iterdir():
        if entry.name in keep:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        removed += 1
    if not any(target.iterdir()):
        os.rmdir(target)
    return removed
