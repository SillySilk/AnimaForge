"""Where OneTrainer lives, and whether a run may start.

Krea Forge never imports OneTrainer code — it needs exactly three artifacts of the
install: the root (subprocess cwd — train.py resolves imports and secrets.json
against it), its venv's python.exe, and scripts/train.py.
"""
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from core import sysmem

# Latent cache + workspace + 30-minute backups of a 12B LoRA run need headroom.
MIN_FREE_GB = 30

# The vendored preset trains the transformer as int W8A8 *compressed* (nvCOMP,
# effective 4-5 bit -- OneTrainer #1630, 2026-08-07). An older install rejects the
# data type at config load, minutes into a run nobody is watching; this file is
# what that commit added, so its absence is the cheapest "too old" there is.
COMPRESSION_MARKER = Path("modules") / "util" / "nvcomp_util.py"

# Commit charge (RAM + pagefile) a run has been seen to hold while it quantizes
# the 12B model to INT8, minutes before step 1: 51.6 GB, measured on the run the
# system commit limit stopped dead at 36% of quantization. Below this the run is
# warned, not refused. Refusing was tried, and it cost more runs than it saved:
# most runs under the line just load slower, and the one that does die is named
# by the crash hint afterwards (session._CRASH_HINT). A refusal here was a dialog
# nobody was in the room to answer -- the set was still untrained the next day.
MIN_FREE_COMMIT_GB = 55


@dataclass(frozen=True)
class TrainerInstall:
    root: str
    python: str
    train_script: str


def locate(root: str) -> "TrainerInstall | None":
    """The install at `root`, or None when any load-bearing piece is missing."""
    r = Path(root or "")
    python = r / "venv" / "Scripts" / "python.exe"
    script = r / "scripts" / "train.py"
    if not (root and r.is_dir() and python.is_file() and script.is_file()):
        return None
    return TrainerInstall(str(r), str(python), str(script))


def validate(root: str) -> tuple:
    """(ok, user-facing reason). Setup's status dot and preflight share this."""
    r = Path(root or "")
    if not root or not r.is_dir():
        return False, "OneTrainer folder not found."
    if not (r / "venv" / "Scripts" / "python.exe").is_file():
        return False, "OneTrainer venv is missing (venv\\Scripts\\python.exe)."
    if not (r / "scripts" / "train.py").is_file():
        return False, "scripts\\train.py is missing — is this an OneTrainer folder?"
    if not (r / COMPRESSION_MARKER).is_file():
        return False, ("This OneTrainer is older than the compressed-weights update "
                       "(2026-08-07) the preset relies on. Run its update.bat, then "
                       "check again.")
    return True, ""


def _no_captions_problem(dataset_dir: str) -> str:
    """Why this set cannot be trained at all, or "" when it can.

    OneTrainer reads each image's .txt sidecar (concept prompt_source "sample"). A
    folder with none of them trains every image against an empty prompt — an hour
    spent teaching the model nothing, which is a refusal, not an advisory. Its
    sibling below (a trigger the captions do not carry) is the softer case: that run
    still learns the subject, just not the word for it.
    """
    if not dataset_dir or not Path(dataset_dir).is_dir():
        return ""
    if sorted(Path(dataset_dir).glob("*.txt")):
        return ""
    return ("The dataset folder has no captions (.txt), so every image would train "
            "against an empty prompt. Caption the set first (Dataset → Describe, "
            "then Build → .txt).")


def _trigger_caption_problem(dataset_dir: str, trigger_word: str) -> str:
    """Why this set cannot teach `trigger_word`, or "" when it can.

    A trigger that appears in none of the captions is never trained: the finished
    LoRA does not answer to it and every sample renders a token the run never saw. A
    set captioned before the trigger was set — or by another tool — looks completely
    normal on disk, which is exactly why this is checked rather than assumed.

    An uncaptioned folder answers "" here; _no_captions_problem owns that case, and
    it refuses the run rather than merely reporting it.
    """
    folder = Path(dataset_dir)
    captions = sorted(folder.glob("*.txt"))
    if not captions:
        return ""
    pattern = re.compile(rf"\b{re.escape(trigger_word)}\b", re.IGNORECASE)
    for path in captions:
        try:
            if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
                return ""
        except OSError:
            continue
    return (f"None of the {len(captions)} captions contain the trigger word "
            f"'{trigger_word}', so the LoRA would never answer to it. Rebuild the "
            f"captions with the trigger (Dataset → Build → .txt) or change the trigger.")


def _commit_warning() -> str:
    """What the user should know about system memory, or "" when there is room.

    A probe that answers None means the OS would not report -- nothing to say.
    """
    free = sysmem.available_commit_gb()
    if free is None or free >= MIN_FREE_COMMIT_GB:
        return ""
    warning = (
        f"Only {free:.0f} GB of system memory is free to commit, and loading the "
        f"model has taken about {MIN_FREE_COMMIT_GB} GB (RAM plus pagefile). The "
        f"run is starting anyway: usually that just means it loads slower, but if "
        f"Windows runs out it will kill the trainer part-way through loading, "
        f"with no error message.")
    note = sysmem.biggest_consumer_note()
    if note:
        return warning + note + " Closing it would give the run more room."
    # Nothing big enough to close: the commit limit itself is the constraint.
    return warning + (" Nothing running is large enough to close, so raising the "
                      "Windows pagefile size is what would give it more room.")


def setup_problems(root, dataset_dir, image_count, trigger_word, output_base) -> list:
    """The subset of preflight() that is already knowable before a caption exists.

    The Forge It pipeline captions the set before it ever reaches a trainer, so an
    unset trigger or a missing OneTrainer folder would otherwise surface an hour
    later, at the end of a run the user was told to walk away from. These checks
    answer the same way then as now, so the pipeline asks them up front.

    Left out deliberately: whether the captions carry the trigger (they are not
    written yet — that is the whole point of the run), the commit-charge probe (an
    hour of captioning invalidates any reading taken before it), and the trigger
    itself. An unset trigger used to be listed here and it stopped the pipeline at
    a dialog nobody was in the room to answer; it is now filled in from the LoRA
    name before the run starts and reported through advisories().
    """
    problems = []
    ok, why = validate(root)
    if not ok:
        problems.append(f"{why} Set the OneTrainer folder in Setup.")
    if not dataset_dir or not Path(dataset_dir).is_dir():
        problems.append("Choose a dataset folder on Home first.")
    elif int(image_count or 0) <= 0:
        # Inherited hard rule: a zero-image run is never started, whatever the cause.
        problems.append("The dataset folder has no images — training refused.")
    if not output_base or not Path(output_base).is_dir():
        problems.append("Set the output folder in Setup first.")
    else:
        free_gb = shutil.disk_usage(output_base).free / 1e9
        if free_gb < MIN_FREE_GB:
            problems.append(
                f"Only {free_gb:.0f} GB free on the output drive — a run needs "
                f"~{MIN_FREE_GB} GB for the latent cache, workspace, and backups.")
    return problems


def preflight(root, dataset_dir, image_count, trigger_word, output_base) -> list:
    """Every failed check as a user-facing sentence; an empty list means go.
    Nothing is written to disk until this returns empty.

    Everything in here makes the run impossible or ruinous: no trainer, no images,
    no disk. Things that merely make the run *worse* than it could be — or that
    *might* end it, like a tight system-memory reading — belong in advisories().
    Refusing those stopped unattended runs at a dialog.
    """
    problems = setup_problems(root, dataset_dir, image_count, trigger_word, output_base)
    why = _no_captions_problem(dataset_dir)
    if why:
        problems.append(why)
    return problems


def advisories(dataset_dir, trigger_word) -> list:
    """What the user should know before an hour is spent, none of it a veto.

    Each entry is a finished user-facing sentence. The run is valid with any or all
    of them outstanding — they describe a LoRA that will be less than it could be,
    not one that cannot be made — so they are posted as notices and the run carries
    on. Blocking here is what made a walk-away run stop at a dialog.
    """
    out = []
    trigger = (trigger_word or "").strip()
    if trigger and dataset_dir and Path(dataset_dir).is_dir():
        why = _trigger_caption_problem(dataset_dir, trigger)
        if why:
            out.append(why)
    if dataset_dir and Path(dataset_dir).is_dir():
        from core.engines.onetrainer import buckets
        cropped = buckets.crop_report(dataset_dir)
        if cropped:
            out.append(cropped)
    # Independent of the dataset: the machine can be perfectly configured and still
    # be short of the memory the model load has needed. Said, not refused -- see
    # MIN_FREE_COMMIT_GB for why.
    why = _commit_warning()
    if why:
        out.append(why)
    return out
