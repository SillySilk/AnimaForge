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
