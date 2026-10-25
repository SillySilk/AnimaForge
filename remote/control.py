"""Remote dashboard logic: the batch queue, the headless runner process, and what a
run has produced. No widgets and no HTTP here — remote/server.py serves this.

The runner is the existing `main.py --run-batch`; this module only starts it, finds
it (by scanning processes, so a restarted dashboard still sees a run in flight),
stops it, and reads the files it already writes.
"""
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import psutil

from core import models, sets
from core.batch import DONE, FAILED, QUEUED, RUNNING, RunDefinition, load_queue, save_queue
from core.dataset_manager import count_images_in_folder
from core.engines.onetrainer.process import parse_progress_line
from core.paths import run_output_dir, sanitize_name
from core.step_calculator import suggest_target_steps
from core.train_metrics import parse_tqdm

APP_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
LOSS_POINTS = 200            # a loss chart never carries more points than this
LOG_TAIL_LINES = 30
_LOG_READ_BYTES = 4 << 20    # the newest 4 MB of a log is plenty for a run's curve
# Folders inside a run that hold engine working state, never previews.
_SKIP_DIRS = {"cache", "backup", "configs", "logs", "tensorboard"}


@dataclass
class Paths:
    root: Path          # the app folder: the runner's working directory, holds main.py
    queue: Path
    status: Path
    log: Path           # the runner's captured output
    sets_root: Path     # parent of sets/


def default_paths() -> Paths:
    from core.batch_status import default_status_path
    status = Path(default_status_path())
    if not status.is_absolute():
        status = APP_ROOT / status
    return Paths(APP_ROOT, APP_ROOT / "batch_queue.json", status,
                 APP_ROOT / "logs" / "remote_batch.log", APP_ROOT)


# ---- saved sets as templates -------------------------------------------------

def list_templates(paths: Paths) -> list:
    out = []
    for name in sets.list_sets(paths.sets_root):
        rd = sets.load_set(name, paths.sets_root)
        if rd is None:
            continue
        preset = models.MODELS.get(rd.model)
        out.append({"name": name, "model": rd.model,
                    "model_name": preset.name if preset else rd.model,
                    "subject_type": rd.subject_type or "character"})
    return out


def run_from_template(template: RunDefinition, folder: str, name: str,
                      trigger: str) -> RunDefinition:
    """A queued run on a new folder that keeps every setting of a saved set."""
    count = count_images_in_folder(folder)
    d = template.to_dict()
    d.update(dataset_folder=str(folder), image_count=count, lora_name=name,
             trigger_word=trigger, sample_prompts=[], style_anchor="",
             network_weights="", compare_group="", shipped_epoch=0, status=QUEUED)
    if models.get(template.model).engine == models.KOHYA:
        d["target_steps"] = suggest_target_steps(template.subject_type or "character", count,
                                              model=template.model)
    else:
        d["target_steps"] = 0        # Auto: the engine sizes the run from the dataset
    return RunDefinition.from_dict(d)


def run_dir(paths: Paths, run: RunDefinition) -> Path:
    base = Path(run.output_dir or "output")
    if not base.is_absolute():
        base = paths.root / base
    return Path(run_output_dir(str(base), run.lora_name))


def add_run(paths: Paths, folder: str, name: str, trigger: str, template_name: str):
    """(ok, message). Appends a run to the queue file."""
    name, trigger = (name or "").strip(), (trigger or "").strip()
    if not sanitize_name(name):
        return False, "Give the LoRA a name."
    if not folder or not Path(folder).is_dir():
        return False, "Pick a folder of training images."
    if count_images_in_folder(folder) == 0:
        return False, "That folder has no images in it."
    template = sets.load_set(template_name, paths.sets_root) if template_name else None
    if template is None:
        return False, "Pick a saved set to take the settings from."
    if template.model not in models.MODELS:
        return False, f"The saved set '{template_name}' is for an unknown model."
    runs = load_queue(paths.queue)
    if any(r.lora_name.lower() == name.lower() for r in runs):
        return False, f"'{name}' is already in the queue."
    run = run_from_template(template, folder, name, trigger)
    if run_dir(paths, run).exists():
        return False, f"A run named '{name}' already has an output folder. Pick another name."
    runs.append(run)
    save_queue(str(paths.queue), runs)
    return True, f"Queued '{name}' ({run.image_count} images, {models.get(run.model).name})."


def remove_run(paths: Paths, index: int):
    runs = load_queue(paths.queue)
    if not 0 <= index < len(runs):
        return False, "That run is no longer in the queue."
    gone = runs.pop(index)
    save_queue(str(paths.queue), runs)
    return True, f"Removed '{gone.lora_name}'."


# ---- processes -----------------------------------------------------------------

def _is_our_main(arg: str, cwd: str, main: str) -> bool:
    if not arg.lower().endswith("main.py"):
        return False
    full = arg if os.path.isabs(arg) else os.path.join(cwd or "", arg)
    return os.path.normcase(os.path.abspath(full)) == main


def find_app_processes(root: Path):
    """(desktop app processes, headless runner processes) for this app folder.

    A venv's python.exe re-launches the base interpreter with the same command line,
    so one app can show as two processes; both are returned."""
    main = os.path.normcase(str((Path(root) / "main.py").resolve()))
    gui, runners = [], []
    for p in psutil.process_iter(["cmdline"]):
        try:
            cmd = p.info["cmdline"] or []
            if not any(a.lower().endswith("main.py") for a in cmd):
                continue
            try:
                cwd = p.cwd()
            except (psutil.AccessDenied, psutil.ZombieProcess):
                cwd = ""
            if not any(_is_our_main(a, cwd, main) for a in cmd):
                continue
            (runners if "--run-batch" in cmd else gui).append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return gui, runners


def runner_command(paths: Paths, python: str = "") -> list:
    exe = Path(python or sys.executable)
    console = exe.with_name("python.exe")    # pythonw has no stdout to capture
    if exe.name.lower() == "pythonw.exe" and console.is_file():
        exe = console
    return [str(exe), "-u", str(paths.root / "main.py"), "--run-batch",
            "--queue", str(paths.queue), "--status", str(paths.status)]


def start_runner(paths: Paths, command=None):
    """Start the queue in the background; its output goes to paths.log."""
    paths.log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    with open(paths.log, "w", encoding="utf-8") as log:
        return subprocess.Popen(command or runner_command(paths), cwd=str(paths.root),
                                stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env, creationflags=flags)


def stop_runner(paths: Paths) -> int:
    """Kill every runner for this app and everything it started (the trainer).
    A run cut off this way goes back to queued, so the next Start runs it again."""
    _gui, runners = find_app_processes(paths.root)
    victims = []
    for p in runners:
        try:
            victims.extend(p.children(recursive=True))
        except psutil.NoSuchProcess:
            pass
        victims.append(p)
    for p in victims:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(victims, timeout=15)
    runs = load_queue(paths.queue)
    if any(r.status == RUNNING for r in runs):
        for r in runs:
            if r.status == RUNNING:
                r.status = QUEUED
        save_queue(str(paths.queue), runs)
    return len(runners)


# ---- what is happening -----------------------------------------------------------

def _read_status(path: Path) -> list:
    import json
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def run_rows(paths: Paths, running: bool) -> list:
    """One row per queued run: the queue file's state, with the runner's live status
    (phase, step) laid over it while it runs."""
    runs = load_queue(paths.queue)
    live = _read_status(paths.status)
    aligned = (len(live) == len(runs)
               and all(e.get("lora_name") == r.lora_name for e, r in zip(live, runs)))
    by_name = {e.get("lora_name"): e for e in live}
    rows = []
    for i, r in enumerate(runs):
        e = (live[i] if aligned else by_name.get(r.lora_name)) or {}
        state = r.status
        if running and e.get("state"):
            state = e["state"]
        elif not running and state == RUNNING:
            state = QUEUED
        preset = models.MODELS.get(r.model)
        rows.append({
            "index": i, "lora_name": r.lora_name, "model": r.model,
            "model_name": preset.name if preset else r.model,
            "dataset_folder": r.dataset_folder, "image_count": r.image_count,
            "trigger_word": r.trigger_word, "state": state,
            "phase": e.get("phase", "") if state == RUNNING else "",
            "current_step": e.get("current_step", 0), "target_steps": e.get("target_steps", 0),
            "started_at": e.get("started_at"), "finished_at": e.get("finished_at"),
            "error_tail": e.get("error_tail") if state == FAILED else None,
        })
    return rows


def _read_lines(path: Path) -> list:
    try:
        with open(path, "rb") as f:
            size = f.seek(0, 2)
            f.seek(max(0, size - _LOG_READ_BYTES))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return [ln for ln in text.replace("\r", "\n").split("\n") if ln.strip()]


def _section_for(lines: list, lora_name: str) -> list:
    """The runner log lines that belong to one run: from its '[Batch] ... starting'
    line to the next run's."""
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("[Batch] (") and "starting" in ln:
            if ln.rstrip().endswith(f"'{lora_name}'"):
                start = i
            elif start is not None:
                return lines[start:i]
    return lines[start:] if start is not None else []


def _thin(points: list, limit: int = LOSS_POINTS) -> list:
    if len(points) <= limit:
        return points
    stride = len(points) / limit
    return [points[int(i * stride)] for i in range(limit - 1)] + [points[-1]]


def loss_points(paths: Paths, run: RunDefinition) -> list:
    """The run's loss curve as [step, loss] pairs, oldest first."""
    preset = models.MODELS.get(run.model)
    by_step = {}
    if preset is not None and preset.engine == models.ONETRAINER:
        n = 0
        for ln in _read_lines(run_dir(paths, run) / "trainer.log"):
            d = parse_progress_line(ln)
            if d and d.get("loss") is not None:
                n += 1
                by_step[n] = d["loss"]
    else:
        for ln in _section_for(_read_lines(paths.log), run.lora_name):
            d = parse_tqdm(ln)
            if "loss" in d and "step" in d:
                by_step[d["step"]] = d["loss"]
    return _thin([[s, by_step[s]] for s in sorted(by_step)])


def log_tail(paths: Paths, run: RunDefinition | None = None) -> list:
    """The newest lines worth reading: the OneTrainer run's own log for a Krea 2 run
    (the runner only reports its step), the runner's output otherwise."""
    lines = []
    if run is not None and models.MODELS.get(run.model) is not None \
            and models.get(run.model).engine == models.ONETRAINER:
        lines = _read_lines(run_dir(paths, run) / "trainer.log")
    if not lines:
        lines = _read_lines(paths.log)
    out = []
    for ln in lines:                   # tqdm redraws: keep only the last of a run of bars
        if out and "|" in ln and "|" in out[-1] and ln.split("|")[0] == out[-1].split("|")[0]:
            out[-1] = ln
        else:
            out.append(ln)
    return out[-LOG_TAIL_LINES:]


def run_files(paths: Paths, run: RunDefinition) -> dict:
    """Preview images (newest first) and LoRA files (the run folder's top level)."""
    root = run_dir(paths, run)
    previews, loras = [], []
    if root.is_dir():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if not d.endswith("-state") and d.lower() not in _SKIP_DIRS]
            for fn in filenames:
                if Path(fn).suffix.lower() in IMAGE_EXTS:
                    p = Path(dirpath) / fn
                    previews.append((p.stat().st_mtime, p.relative_to(root).as_posix()))
        for p in root.glob("*.safetensors"):
            st = p.stat()
            loras.append({"name": p.name, "size": st.st_size, "mtime": st.st_mtime})
    previews.sort(reverse=True)
    loras.sort(key=lambda d: d["mtime"], reverse=True)
    return {"previews": [rel for _m, rel in previews], "loras": loras}


def run_file(paths: Paths, run: RunDefinition, rel: str):
    """The real path of a file inside the run's folder, or None."""
    root = run_dir(paths, run).resolve()
    target = (root / (rel or "")).resolve()
    if root not in target.parents or not target.is_file():
        return None
    return target


def queue_run(paths: Paths, index: int):
    runs = load_queue(paths.queue)
    return runs[index] if 0 <= index < len(runs) else None


def all_done(rows: list) -> bool:
    return bool(rows) and all(r["state"] == DONE for r in rows)
