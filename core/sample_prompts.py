"""Draw sample/preview prompts from a dataset's own captions.

`grab_caption_blocks` returns random, verbatim `.txt` caption blocks — the exact
captions the trainer reads — to seed the Train tab's sample-prompt box. The trigger
word is NOT stripped here; config_generator/prepare_sample_args handles it at write time.
"""
import random


def grab_caption_blocks(folder, n: int, rng=None):
    """Return up to `n` random, verbatim caption blocks from a captioned dataset.

    Each block is the full merged `.txt` caption the trainer reads (natural language
    + tags together). Empty/whitespace-only captions are skipped. When fewer than `n`
    captions exist, all of them are returned (shuffled). `n <= 0`, a missing folder, or
    no captions yields []. Pass a seeded `random.Random` for deterministic selection.
    """
    from core.dataset_manager import scan_folder

    if n <= 0:
        return []
    items = scan_folder(folder) if folder else []
    blocks = [cap for cap in (d.get("caption", "").strip() for d in items) if cap]
    if not blocks:
        return []
    rng = rng or random
    if len(blocks) <= n:
        out = list(blocks)
        rng.shuffle(out)
        return out
    return rng.sample(blocks, n)


def character_caption_blocks(folder, rng=None):
    """One verbatim caption block per named subject, for a set with a character roster.

    Each subject gets a caption that names them, from the picture where they appear
    with the fewest others (solo when one exists), so every preview checks a
    different person's likeness by name. No picture is used twice. Subjects with no
    named caption are skipped. A set with no roster subjects yields [] (the caller
    falls back to random blocks). Pass a seeded `random.Random` for determinism.
    """
    import re
    from pathlib import Path
    from core import characters as ch
    from core.dataset_manager import scan_folder

    if not folder:
        return []
    data = ch.load(folder)
    subjects = [c.token for c in data.roster if c.role == "subject" and c.token.strip()]
    if not subjects:
        return []
    rng = rng or random
    present = {name: a.get("present", []) for name, a in data.assignments.items()}
    captions = {Path(d["image_path"]).name: d.get("caption", "").strip()
                for d in scan_folder(folder)}
    used, out = set(), []
    for token in subjects:
        named = re.compile(rf"(?<!\w){re.escape(token)}(?!\w)", re.IGNORECASE)
        pool = [(len(present.get(name, [])) if token in present.get(name, []) else 99, name)
                for name, cap in captions.items()
                if cap and name not in used and named.search(cap)]
        if not pool:
            continue
        fewest = min(n for n, _ in pool)
        name = rng.choice(sorted(name for n, name in pool if n == fewest))
        used.add(name)
        out.append(captions[name])
    return out
