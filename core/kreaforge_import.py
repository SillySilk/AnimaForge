"""One-time import of Krea Forge's settings and past runs into Anima Forge v3.

Krea Forge (the retired Krea 2 trainer) kept its own settings store and a run
folder per LoRA under output/. On the first launch that finds a Krea Forge folder
beside this app, v3 brings over the settings that still mean something here and
turns each past run into a Krea 2 set, with its caption snapshots. It never
overwrites a value or a set the user already has, and runs once (marker key).
"""
import json
import re
import shutil
from pathlib import Path

from core import sets
from core.batch import RunDefinition

MARKER = "kreaforge_imported"

# Settings with the same meaning in v3. Krea Forge's model paths, sample prompts and
# similar are leftovers from its Anima era and stale; they are never copied.
SETTING_KEYS = ("onetrainer_root", "debug_profiles", "ui_font_mode", "ui_font_family",
                "forge_lora_dir")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_TRIGGER_RE = re.compile(r"^\s*(.+?)\s+—\s+the set's trigger word")


def default_root() -> Path:
    """Krea Forge's folder in the Apps layout: beside this app."""
    return Path(__file__).resolve().parents[2] / "Krea Forge"


def _bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def import_settings(src, dst) -> list:
    """Copy SETTING_KEYS from Krea Forge's QSettings where Anima Forge has nothing."""
    copied = []
    for key in SETTING_KEYS:
        if not src.contains(key):
            continue
        value = src.value(key)
        if value in (None, ""):
            continue
        current = dst._s.value(key) if dst._s.contains(key) else None
        if current not in (None, ""):
            continue
        dst.set(key, _bool(value) if key == "debug_profiles" else value)
        copied.append(key)
    return copied


def _trigger(run_dir: Path, name: str) -> str:
    words = run_dir / f"{name}.trained-words.txt"
    try:
        text = words.read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        m = _TRIGGER_RE.match(line)
        if m:
            return m.group(1).strip()
    return ""


def _image_count(folder: str) -> int:
    p = Path(folder or "")
    try:
        if not p.is_dir():
            return 0
        return sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    except OSError:
        return 0


def _already_imported(candidate: str, dataset: str, sets_root: Path) -> bool:
    rd = sets.load_set(candidate, root=sets_root)
    return rd is not None and rd.model == "krea2" and rd.dataset_folder == dataset


def _free_name(name: str, dataset: str, sets_root: Path):
    """The set name to import under, or None when this run is already a set (an
    earlier, interrupted import) or both names are taken."""
    taken = set(sets.list_sets(root=sets_root))
    candidates = (name, f"{name} - Krea 2")
    if any(sets._safe(c) in taken and _already_imported(c, dataset, sets_root)
           for c in candidates):
        return None
    for candidate in candidates:
        if sets._safe(candidate) not in taken:
            return candidate
    return None


def import_runs(krea_root: Path, sets_root: Path) -> list:
    """Each Krea Forge run (output/<name>/ with concepts.json + config.json) becomes
    a Krea 2 set; caption snapshots come along. Returns the set names created."""
    out_dir = Path(krea_root) / "output"
    created = []
    for run_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        concepts, config = run_dir / "concepts.json", run_dir / "config.json"
        if not (concepts.is_file() and config.is_file()):
            continue
        try:
            entries = json.loads(concepts.read_text(encoding="utf-8"))
            dataset = str(entries[0]["path"])
        except (OSError, ValueError, LookupError, TypeError):
            continue
        try:
            set_name = _free_name(run_dir.name, dataset, sets_root)
            if set_name is None:
                continue
            # output_dir stays empty: the run trains into Setup's output folder, and
            # Krea Forge's own folder is read-only history that may be removed.
            rd = RunDefinition(model="krea2", lora_name=run_dir.name, dataset_folder=dataset,
                               image_count=_image_count(dataset),
                               trigger_word=_trigger(run_dir, run_dir.name),
                               output_dir="", subject_type="character")
            sets.save_set(set_name, rd, root=sets_root)
            snap_src = Path(krea_root) / "sets" / run_dir.name
            snap_dst = sets.sets_dir(sets_root) / sets._safe(set_name)
            if snap_src.is_dir() and not snap_dst.exists():
                for stage in snap_src.glob("captions-*"):
                    if stage.is_dir():
                        shutil.copytree(stage, snap_dst / stage.name)
        except OSError:
            continue        # one unreadable run never stops the rest
        created.append(set_name)
    return created


def run_once(app_settings, krea_root, sets_root, krea_settings) -> list:
    """Both imports, once. Returns report lines ([] when nothing ran)."""
    krea_root = Path(krea_root)
    if app_settings.get(MARKER) or not (krea_root / "output").is_dir():
        return []
    copied = import_settings(krea_settings, app_settings)
    created = import_runs(krea_root, Path(sets_root))
    app_settings.set(MARKER, True)
    lines = []
    if created:
        lines.append(f"Imported {len(created)} Krea Forge run(s) as Krea 2 sets: "
                     + ", ".join(created) + ".")
    if copied:
        lines.append("Imported Krea Forge settings: " + ", ".join(copied) + ".")
    return lines
