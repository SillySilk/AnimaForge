"""Pictures added to a trained set later: a new character or element, trained in
with extra weight when the run trains further.

Each addition is a sub-folder of the set's dataset folder (`<dataset>/<Name>/`),
listed by name in RunDefinition.added_folders. A sub-folder because both trainers
weight per folder and never per file: Kohya gives each folder its own subset with
its own num_repeats, OneTrainer its own concept with its own balancing. Every
addition is shown EMPHASIS times as often as the original pictures (Setup →
"New character emphasis", default 3x), so a new character catches up with a cast
that has already had a whole run.

Boosting a character already in the set (RunDefinition.boosted) works the same
way: its pictures -- every file whose NAME_SERIAL_CATEGORY name includes it, group
shots too -- are copied into `<dataset>/_boost_<Name>/` and that folder is weighted
EMPHASIS - 1, so with the originals still in the set they are shown EMPHASIS times
as often in all. The copies are rebuilt from the set at every launch
(prepare_boosts), so a renamed or re-captioned picture never trains stale.

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


BOOST_PREFIX = "_boost_"


def boost_folder(subject: str) -> str:
    return BOOST_PREFIX + sanitize_name(subject)


def _subjects_of(image: Path) -> list:
    from core.naming import parse_name
    parsed = parse_name(image.stem)
    return [x.lower() for x in parsed["subjects"]] if parsed else []


def subject_pictures(dataset_folder: str, subject: str) -> list:
    """The set's own pictures whose file name includes `subject` (any case)."""
    want = (subject or "").strip().lower()
    return [img for img in _images(Path(dataset_folder)) if want in _subjects_of(img)]


def subjects(dataset_folder: str) -> list:
    """[(subject, picture count)] across the set, most pictures first."""
    counts = {}
    names = {}
    for img in _images(Path(dataset_folder)):
        from core.naming import parse_name
        parsed = parse_name(img.stem)
        for subj in (parsed["subjects"] if parsed else []):
            key = subj.lower()
            names.setdefault(key, subj)
            counts[key] = counts.get(key, 0) + 1
    return sorted(((names[k], n) for k, n in counts.items()), key=lambda t: (-t[1], t[0]))


def training_folders(dataset_folder: str, added, boosted, weight: float) -> list:
    """[(folder, picture count, weight)] trained beside the set itself: each added
    folder at `weight`, each boost folder at `weight - 1` (its pictures are also in
    the set at 1). Boost counts come from the set, so this never touches the disk
    beyond reading it."""
    out = [(path, count, float(weight)) for path, count in folders(dataset_folder, added)]
    extra = float(weight) - 1.0
    if extra > 0:
        for subject in boosted or []:
            count = len(subject_pictures(dataset_folder, subject))
            if count:
                out.append((str(Path(dataset_folder) / boost_folder(subject)), count, extra))
    return out


def weighted_count(image_count: int, dataset_folder: str, names, weight: float,
                   boosted=()) -> int:
    """The picture count the step math should use: the originals plus each added or
    boosted picture counted at its folder's weight."""
    extra = sum(count * w for _p, count, w in
                training_folders(dataset_folder, names, boosted, weight))
    return int(image_count) + round(extra)


def prepare_boosts(dataset_folder: str, boosted) -> None:
    """Rebuild every boost folder from the set as it is now, and drop the ones no
    longer boosted. Called at launch, before the trainer reads the folders."""
    base = Path(dataset_folder)
    if not base.is_dir():
        return
    wanted = {boost_folder(subj): subj for subj in boosted or []}
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith(BOOST_PREFIX):
            shutil.rmtree(d)
    for folder, subject in wanted.items():
        dest = base / folder
        dest.mkdir()
        for img in subject_pictures(dataset_folder, subject):
            # The picture, its caption sidecars, and the trainer's cache files for it
            # (`<stem>_<size>_<model>.npz`), so the copies need no fresh caching.
            for f in base.iterdir():
                if f.is_file() and (f.stem == img.stem or f.name.startswith(img.stem + "_")):
                    shutil.copy2(f, dest / f.name)


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
