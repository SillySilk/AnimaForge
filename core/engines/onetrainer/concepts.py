"""Concept and sample payloads for a run — mirrors the proven 2026-08-07 run.

OneTrainer fills unset sample fields from its per-model defaults, and its trainer
proceeds without sampling if the file can't be parsed — so the sample entries here
stay minimal. The manual end-to-end (Task 14) verifies samples actually render.

Sample prompts come from the set's own captions; see the note above build_samples
for why a written-in scene is the wrong thing to render.
"""
import re
from pathlib import Path

# Fixed concept seed → stable shuffling/latent-cache keys across reruns of a set.
_CONCEPT_SEED = 743829201


def build_concepts(dataset_dir: str) -> list:
    folder = Path(dataset_dir)
    return [{
        "__version": 2,
        "image": {
            "__version": 0,
            "enable_crop_jitter": True,
            "enable_random_flip": False, "enable_fixed_flip": False,
            "enable_random_rotate": False, "enable_fixed_rotate": False,
            "random_rotate_max_angle": 0.0,
            "enable_random_brightness": False, "enable_fixed_brightness": False,
            "random_brightness_max_strength": 0.0,
            "enable_random_contrast": False, "enable_fixed_contrast": False,
            "random_contrast_max_strength": 0.0,
            "enable_random_saturation": False, "enable_fixed_saturation": False,
            "random_saturation_max_strength": 0.0,
            "enable_random_hue": False, "enable_fixed_hue": False,
            "random_hue_max_strength": 0.0,
            "enable_resolution_override": False, "resolution_override": "512",
            "enable_random_circular_mask_shrink": False,
            "enable_random_mask_rotate_crop": False,
        },
        "text": {
            "__version": 0,
            "prompt_source": "sample",
            "prompt_path": "",
            "enable_tag_shuffling": False,
            "tag_delimiter": ",",
            "keep_tags_count": 1,
            "tag_dropout_enable": False,
            "tag_dropout_mode": "FULL",
            "tag_dropout_probability": 0.0,
            "tag_dropout_special_tags_mode": "NONE",
            "tag_dropout_special_tags": "",
            "tag_dropout_special_tags_regex": False,
            "caps_randomize_enable": False,
            "caps_randomize_probability": 0.0,
            "caps_randomize_mode": "capslock, title, first, random",
            "caps_randomize_lowercase": False,
        },
        "name": folder.name,
        "path": str(folder),
        "seed": _CONCEPT_SEED,
        "enabled": True,
        "type": "STANDARD",
        "include_subdirectories": False,
        "image_variations": 1,
        "text_variations": 1,
        "balancing": 1.0,
        "balancing_strategy": "REPEATS",
        "loss_weight": 1.0,
        "concept_stats": {},
    }]


# Sample prompts are the run's real captions, not a written-in scene. A preview is
# only worth rendering if it previews what the run is being taught: Bev_and_Billy
# (2026-08-17) trained on two-character comic panels and sampled "a portrait
# photograph of a person standing indoors", which could not have shown the
# characters even from a healthy LoRA. Krea 2's text encoder is Qwen3-VL, which
# reads sentences, and the captions are already prose in exactly the shape the
# model is being taught (`trigger, prose` — see dataset_manager.build_caption),
# so the caption IS the correct prompt.

# How many captions to preview per interval. OneTrainer renders every enabled entry
# at each sampling interval (the preset's is 10 minutes), so this is a direct
# wall-clock cost on the run. Three is enough to tell "learning the subject" from
# "memorising one frame" without crowding out training time.
SAMPLE_COUNT = 3

# Krea 2's text encoder window is 512 tokens (the preset's
# text_encoder_sequence_length). At ~4 characters per token this stays well inside
# it, so a long caption is cut here — deliberately, on a sentence — rather than
# clipped mid-word by the tokenizer.
_MAX_PROMPT_CHARS = 1600

# Fixed so one interval is comparable to the next; consecutive so the previews are
# not near-duplicates of each other.
_FIRST_SEED = 42


def _captions(dataset_dir: str) -> list:
    """Every non-blank .txt caption in the folder, whitespace normalised, by name.

    .txt is the file OneTrainer reads (concept prompt_source "sample"), so these
    are literally the strings the run trains on.
    """
    out = []
    try:
        paths = sorted(Path(dataset_dir).glob("*.txt"))
    except OSError:
        return out
    for path in paths:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = " ".join(raw.split())      # captions wrap across lines; prompts do not
        if text:
            out.append(text)
    return out


def _trim(text: str) -> str:
    if len(text) <= _MAX_PROMPT_CHARS:
        return text
    head = text[:_MAX_PROMPT_CHARS]
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    return head[:cut + 1] if cut > 0 else head.rstrip()


def _spread(items: list, count: int) -> list:
    """`count` items spaced evenly across the list, or all of them if there are
    fewer. Evenly spaced rather than the first N: the first N captions of a sorted
    set are often near-identical frames, which previews one pose three times.
    """
    if len(items) <= count:
        return items
    return [items[i * len(items) // count] for i in range(count)]


def build_samples(dataset_dir: str, trigger_word: str,
                  count: int = SAMPLE_COUNT) -> list:
    """512×512 samples drawn from the set's own captions, rendered every sampling
    interval so the user can watch the subject appear.

    Only captions carrying the trigger are previewed — a caption without it never
    taught the trigger, so rendering it measures nothing about whether the LoRA
    answers to it. This matches engine._trigger_caption_problem's word-boundary
    test, which already refuses a run whose captions all lack the trigger; if that
    guard is somehow bypassed this returns no samples rather than faking a preview
    of an untrained token.
    """
    captions = _captions(dataset_dir)
    trigger = (trigger_word or "").strip().strip(",").strip()
    if trigger:
        pattern = re.compile(rf"\b{re.escape(trigger)}\b", re.IGNORECASE)
        captions = [c for c in captions if pattern.search(c)]
    chosen = _spread(captions, max(1, int(count)))
    return [{
        "enabled": True,
        "prompt": _trim(caption),
        "negative_prompt": "",
        "height": 512,
        "width": 512,
        "frames": 1,
        "seed": _FIRST_SEED + i,
        "random_seed": False,
    } for i, caption in enumerate(chosen)]
