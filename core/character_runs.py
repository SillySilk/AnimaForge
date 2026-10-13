"""Small per-character LoRAs trained after a cast/style LoRA.

Each named character with enough solo images gets its own short run, trained on just
those images under their name, to load beside the main LoRA and lock that person in.

Where a character's images come from:
- Its own set beside the main one, `<set>_characters/<name>/` (Lora Builder exports one
  per cast member: half the cast set's solo shots, half fresh ones). The run trains on
  that folder like any set, captioning it first.
- Otherwise the main set's solo images. That folder is built from the main set's
  captions when the run starts, so a run queued before captioning still trains on the
  finished captions.
Pure stdlib apart from the RunDefinition it copies.
"""
import shutil
from dataclasses import replace
from pathlib import Path

from core import characters
from core.dataset_manager import SUPPORTED_EXTENSIONS
from core.paths import run_output_dir, sanitize_name

DEFAULT_STEPS = 500
# Fewer solo images than this teaches a likeness poorly; the character is skipped.
MIN_IMAGES = 5


def solo_images(data: characters.DatasetCharacters, image_names) -> dict:
    """{token: [image names]} for images where that character is the only one named.

    Group shots are left out: the run teaches one face under one name. Order follows
    the roster, then the image order given."""
    by_token = {c.token: [] for c in data.roster if c.token.strip()}
    for name in image_names:
        entry = data.assignments.get(name) or {}
        # Count everyone named, on the roster or not: an off-roster companion still
        # makes it a group shot.
        people = [t for t in entry.get("present", []) if t.strip()]
        people += [o.get("token", "") for o in entry.get("oneoffs", []) if o.get("token", "").strip()]
        if len(people) == 1 and people[0] in by_token:
            by_token[people[0]].append(name)
    return by_token


def image_names(folder: str) -> list:
    return sorted((p.name for p in Path(folder).iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
                  key=str.lower)


def run_name(main_name: str, token: str) -> str:
    return sanitize_name(f"{main_name}_{token}".replace(" ", "_"))


def own_set(main_folder: str, token: str) -> Path:
    """Where a character's own set sits beside the main set."""
    main = Path(main_folder)
    return main.with_name(main.name + "_characters") / token


def plan(main_run, data, names, steps: int = DEFAULT_STEPS):
    """The character runs to queue after `main_run`, and who was skipped.

    Returns (runs, skipped) where skipped is [(token, image count)] for characters
    under MIN_IMAGES."""
    runs, skipped = [], []
    for token, solo in solo_images(data, names).items():
        name = run_name(main_run.lora_name, token)
        folder = own_set(main_run.dataset_folder, token)
        if folder.is_dir():
            count, dataset, source = len(image_names(str(folder))), str(folder), ""
        else:
            count, source = len(solo), main_run.dataset_folder
            dataset = str(Path(run_output_dir(main_run.output_dir, name)) / "dataset")
        if count < MIN_IMAGES:
            skipped.append((token, count))
            continue
        runs.append(replace(
            main_run,
            lora_name=name,
            dataset_folder=dataset,
            image_count=count,
            trigger_word=token,
            target_steps=int(steps),
            # Previews draw from this character's own captions, not the main set's.
            sample_prompts=[],
            network_weights="",
            compare_group="",
            shipped_epoch=0,
            character_source=source,
            character_token=token,
            status="queued",
        ))
    return runs, skipped


def build_folder(run) -> int:
    """(Re)build a character run's folder from its source set: every solo image of the
    character with its caption. Returns the image count.

    Raises ValueError when the set no longer has enough captioned solo images."""
    source = run.character_source
    data = characters.load(source)
    solo = solo_images(data, image_names(source)).get(run.character_token, [])
    missing = [n for n in solo if not _caption(Path(source) / n)]
    if missing:
        raise ValueError(f"{len(missing)} of {run.character_token}'s images have no caption "
                         f"(first: {missing[0]}) — caption the set first")
    if len(solo) < MIN_IMAGES:
        raise ValueError(f"{run.character_token} has only {len(solo)} solo image(s); "
                         f"needs {MIN_IMAGES}")
    dest = Path(run.dataset_folder)
    if dest.name != "dataset" or dest.resolve() == Path(source).resolve():
        raise ValueError(f"refusing to rebuild {dest}: not a character run folder")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for n in solo:
        shutil.copy2(Path(source) / n, dest / n)
        txt = Path(n).with_suffix(".txt").name
        shutil.copy2(Path(source) / txt, dest / txt)
    return len(solo)


def _caption(image: Path) -> str:
    txt = image.with_suffix(".txt")
    return txt.read_text(encoding="utf-8").strip() if txt.is_file() else ""
