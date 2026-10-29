"""Pictures added to a trained set later: a new character or element, trained in
with extra weight when the run trains further.

Each addition is a sub-folder of the set's dataset folder (`<dataset>/<Name>/`),
listed by name in RunDefinition.added_folders. A sub-folder because both trainers
weight per folder and never per file: Kohya gives each folder its own subset with
its own num_repeats, OneTrainer its own concept with its own balancing. Every
addition is shown EMPHASIS times as often as the original pictures (Setup →
"New character emphasis", default 3x), so a new character catches up with a cast
that has already had a whole run.

The dataset folder's own scan does not recurse, so the original set's picture count
and everything built from it stay as they were; the step math adds the weighted
extra pictures on top (weighted_count).
"""
import shutil
from pathlib import Path

from core.dataset_manager import NL_EXT, SUPPORTED_EXTENSIONS, TAGS_EXT
from core.paths import sanitize_name

SETTING = "added_set_emphasis"
# Sidecars that travel with a picture when it is added: its caption and the two
# sources Combine builds a caption from.
_SIDECARS = (".txt", TAGS_EXT, NL_EXT)


def emphasis(settings=None) -> float:
    """How many times as often an added picture is shown as an original one."""
    if settings is None:
        from core.settings import AppSettings
        settings = AppSettings()
    return max(1.0, float(settings.get(SETTING)))


def _images(folder: Path) -> list:
    if not folder.is_dir():
        return []
    return sorted(f for f in folder.iterdir()
                  if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS)


def folders(dataset_folder: str, names) -> list:
    """[(absolute path, picture count)] for each added folder that exists and holds
    pictures, in the order they were added."""
    out = []
    for name in names or []:
        path = Path(dataset_folder) / name
        count = len(_images(path))
        if count:
            out.append((str(path), count))
    return out


def weighted_count(image_count: int, dataset_folder: str, names, weight: float) -> int:
    """The picture count the step math should use: the originals plus each added
    picture counted `weight` times."""
    extra = sum(count for _path, count in folders(dataset_folder, names))
    return int(image_count) + round(extra * float(weight))


def kohya_repeats(base_repeats: int, weight: float) -> int:
    return max(1, round(int(base_repeats) * float(weight)))


def folder_name(name: str) -> str:
    safe = sanitize_name(name)
    if not safe:
        raise ValueError("Give the new character or element a name.")
    return safe


def add_pictures(dataset_folder: str, name: str, files) -> tuple:
    """Copy `files` (and any caption sidecars beside them) into
    `<dataset_folder>/<name>/`. Returns (folder name, pictures copied). The
    originals are left where they are; a picture already in the folder is kept."""
    base = Path(dataset_folder)
    if not base.is_dir():
        raise ValueError(f"The set's dataset folder is gone: {dataset_folder}")
    sub = folder_name(name)
    dest = base / sub
    dest.mkdir(exist_ok=True)
    copied = 0
    for f in files:
        src = Path(f)
        if not (src.is_file() and src.suffix.lower() in SUPPORTED_EXTENSIONS):
            continue
        if src.parent.resolve() == dest.resolve():
            copied += 1                      # picked from the folder itself
            continue
        target = dest / src.name
        if target.exists():
            continue
        shutil.copy2(src, target)
        for ext in _SIDECARS:
            side = src.with_suffix(ext)
            if side.is_file() and not target.with_suffix(ext).exists():
                shutil.copy2(side, target.with_suffix(ext))
        copied += 1
    return sub, copied


def uncaptioned(folder: str) -> int:
    """Pictures in `folder` with no caption .txt (or an empty one)."""
    n = 0
    for img in _images(Path(folder)):
        try:
            if img.with_suffix(".txt").read_text(encoding="utf-8").strip():
                continue
        except OSError:
            pass
        n += 1
    return n
