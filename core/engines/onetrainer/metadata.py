"""Trained words on the finished LoRA — written into the file, and beside it.

OneTrainer saves the .safetensors with no `__metadata__` at all: no trigger, no
tags, no title. That used not to matter, because Civitai read the tags out of the
file itself and filled the "Trained words" field in for you. It no longer does, so
an hour-old LoRA arrives at an upload form that cannot say what word calls it, and
the one person who knows is the one being asked.

So the run says it, twice, because the two readers are different:

  in the file   `ss_tag_frequency`, `ss_output_name` and the `modelspec.*` keys are
                what ComfyUI's LoRA info panels and A1111/Forge's activation-text
                box read. This is the copy that travels with the file.
  beside it     a `.trained-words.txt` sidecar, because a metadata key nobody
                surfaces is not an answer to "what do I paste in this box".

The words themselves come from the captions the run actually trained on, not from
a guess: the trigger, the quality prefix, and every comma-separated fragment that
recurs across the set. A caption here is prose (`trigger, prefix, sentence…` — see
dataset_manager.build_caption), so the recurring short fragments are exactly the
lead tokens, and the one-off prose sentences fall below MIN_SHARE and drop out.

Stdlib only, matching core/engines/onetrainer/verify.py: requirements.txt is PySide6/Pillow/
toml, so this cannot assume the `safetensors` package. The format does not need
it — 8 bytes of length, then a JSON header, then the tensor buffer, and every
`data_offsets` pair is relative to the end of that header. The header can be
rewritten to any length without a single weight moving.
"""
import json
import os
import shutil
import struct
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from core.version import __version__ as APP_VERSION

SIDECAR_SUFFIX = ".trained-words.txt"

_MAX_HEADER_BYTES = 100 * 1024 * 1024
_ALIGN = 8                       # safetensors pads its header to 8 bytes, with spaces
_COPY_CHUNK = 8 * 1024 * 1024

# A caption fragment longer than this is a sentence about one image, not a word the
# LoRA answers to. The lead tokens this is meant to catch (a trigger, a character
# name, "masterpiece, best quality") are all well inside it.
MAX_WORDS_PER_TAG = 5

# ...and a fragment carried by fewer than this share of the set describes that
# corner of the set, not the set. One image in four is the floor for "recurring".
MIN_SHARE = 0.25

# Enough to name the subject, the style and the prompt scaffolding; past this the
# list stops being something a human pastes into a box.
MAX_TRAINED_WORDS = 24


class MetadataError(RuntimeError):
    """The finished file could not be read or rewritten."""


# ----------------------------------------------------------------------------
# Reading the captions
# ----------------------------------------------------------------------------

def _fragments(text: str):
    """The comma-separated pieces of one caption, whitespace normalised."""
    for piece in text.split(","):
        piece = " ".join(piece.split())
        if piece:
            yield piece


def caption_tags(dataset_dir: str) -> "OrderedDict[str, int]":
    """Every comma fragment across the set's .txt captions → how many carry it.

    Counted case-insensitively (a trigger written 'Aria' in one caption and 'aria'
    in the next is one word) but reported in the casing it was first written in,
    which is the casing the user will prompt with. Ordered most-carried first,
    ties broken toward first appearance so the order is stable across runs.

    Counted per caption, not per occurrence: a word repeated three times in one
    sentence is still one image's worth of evidence.
    """
    counts, display, order = {}, {}, []
    try:
        paths = sorted(Path(dataset_dir).glob("*.txt"))
    except OSError:
        return OrderedDict()
    for path in paths:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        seen = set()
        for frag in _fragments(raw):
            key = frag.casefold()
            if key in seen:
                continue
            seen.add(key)
            if key not in counts:
                counts[key] = 0
                display[key] = frag
                order.append(key)
            counts[key] += 1
    ranked = sorted(order, key=lambda k: (-counts[k], order.index(k)))
    return OrderedDict((display[k], counts[k]) for k in ranked)


def caption_count(dataset_dir: str) -> int:
    try:
        return len(list(Path(dataset_dir).glob("*.txt")))
    except OSError:
        return 0


def trained_words(dataset_dir: str, trigger_word: str = "",
                  quality_prefix: str = "") -> list:
    """The words to prompt this LoRA with, best first.

    The trigger leads whether or not the captions carry it — it is what the user
    set out to teach, and a list that omits it is wrong even when the set is. The
    quality prefix follows (it was prepended to every caption, so it is trained
    scaffolding, not decoration), then whatever else recurs across the set.
    """
    words, seen = [], set()

    def add(word):
        word = " ".join((word or "").split()).strip()
        key = word.casefold()
        if word and key not in seen:
            seen.add(key)
            words.append(word)

    add(trigger_word)
    for piece in _fragments(quality_prefix or ""):
        add(piece)

    total = caption_count(dataset_dir)
    if total:
        floor = max(2, int(total * MIN_SHARE))
        for tag, count in caption_tags(dataset_dir).items():
            if count >= floor and len(tag.split()) <= MAX_WORDS_PER_TAG:
                add(tag)
    return words[:MAX_TRAINED_WORDS]


def tag_frequency(dataset_dir: str) -> dict:
    """kohya's `ss_tag_frequency` shape: {concept name: {tag: count}}.

    Only the fragments short enough to be words — the whole table would otherwise
    carry every prose caption in the set into the file header verbatim, which is
    both enormous and useless to the readers that parse this key.
    """
    tags = OrderedDict(
        (tag, count) for tag, count in caption_tags(dataset_dir).items()
        if len(tag.split()) <= MAX_WORDS_PER_TAG)
    return {Path(dataset_dir).name: tags}


# ----------------------------------------------------------------------------
# Building the metadata block
# ----------------------------------------------------------------------------

def build_metadata(lora_name: str, dataset_dir: str, trigger_word: str = "",
                   quality_prefix: str = "", plan: dict | None = None,
                   image_count: int = 0, when: datetime | None = None) -> dict:
    """The `__metadata__` block for the finished file. Every value is a string —
    that is the safetensors contract, and a reader that hits an int gives up on
    the whole header."""
    plan = plan or {}
    words = trained_words(dataset_dir, trigger_word, quality_prefix)
    stamp = (when or datetime.now()).replace(microsecond=0).isoformat()
    meta = {
        # SAI model spec — the block ComfyUI and Forge look for first.
        "modelspec.sai_model_spec": "1.0.0",
        "modelspec.architecture": "krea-2/lora",
        "modelspec.implementation": "https://github.com/Nerogar/OneTrainer",
        "modelspec.title": lora_name,
        "modelspec.date": stamp,
        "modelspec.resolution": "512x512",
        # kohya's names — what the upload forms and LoRA info panels parse.
        "ss_output_name": lora_name,
        "ss_base_model_version": "krea/Krea-2-Raw",
        "ss_network_module": "networks.lora",
        "ss_network_dim": "16",
        "ss_network_alpha": "16.0",
        "ss_tag_frequency": json.dumps(tag_frequency(dataset_dir),
                                       ensure_ascii=False),
        "ss_num_train_images": str(int(image_count or 0)),
        "animaforge_version": APP_VERSION,
    }
    if words:
        # Both spellings are in the wild: Forge/A1111 read the first as activation
        # text, the SAI-spec readers the second. They carry the same list.
        joined = ", ".join(words)
        meta["ss_trained_words"] = joined
        meta["modelspec.trigger_phrase"] = joined
    if trigger_word:
        meta["animaforge_trigger"] = trigger_word
    for key, field in (("ss_num_epochs", "epochs"),
                       ("ss_max_train_steps", "total_steps"),
                       ("ss_steps_per_epoch", "steps_per_epoch")):
        if plan.get(field):
            meta[key] = str(int(plan[field]))
    return meta


# ----------------------------------------------------------------------------
# Writing it into the file
# ----------------------------------------------------------------------------

def read_metadata(path: str) -> dict:
    """The file's existing `__metadata__`, or {} when it has none."""
    with open(path, "rb") as f:
        header, _ = _read_header(f)
    meta = header.get("__metadata__") or {}
    return meta if isinstance(meta, dict) else {}


def _read_header(f):
    raw_len = f.read(8)
    if len(raw_len) < 8:
        raise MetadataError("file is too short to be a safetensors file")
    (length,) = struct.unpack("<Q", raw_len)
    if length <= 0 or length > _MAX_HEADER_BYTES:
        raise MetadataError(f"header length {length} is not plausible")
    raw = f.read(length)
    if len(raw) < length:
        raise MetadataError("header is truncated — the file did not finish writing")
    try:
        header = json.loads(raw)
    except ValueError as e:
        raise MetadataError(f"header is not JSON ({e})") from e
    if not isinstance(header, dict):
        raise MetadataError("header is not a JSON object")
    return header, length


def write_metadata(path: str, meta: dict) -> None:
    """Merge `meta` into the file's `__metadata__`, in place.

    Every tensor's `data_offsets` is relative to the end of the header, so growing
    the header moves the whole buffer and invalidates nothing. The new file is
    built beside the old one and swapped in with os.replace, because the input is
    an hour of GPU time and a half-written header would destroy it.
    """
    src = Path(path)
    tmp = src.with_name(src.name + ".krea-meta.tmp")
    try:
        with open(src, "rb") as f:
            header, _ = _read_header(f)
            existing = header.get("__metadata__")
            merged = dict(existing) if isinstance(existing, dict) else {}
            merged.update({k: str(v) for k, v in meta.items()})
            header["__metadata__"] = merged
            raw = json.dumps(header, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
            raw += b" " * (-len(raw) % _ALIGN)     # safetensors pads with spaces
            with open(tmp, "wb") as out:
                out.write(struct.pack("<Q", len(raw)))
                out.write(raw)
                shutil.copyfileobj(f, out, _COPY_CHUNK)   # f is at the tensor buffer
        os.replace(tmp, src)
    except OSError as e:
        raise MetadataError(f"could not rewrite {src.name}: {e}") from e
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ----------------------------------------------------------------------------
# Writing it beside the file
# ----------------------------------------------------------------------------

def sidecar_path(lora_path: str) -> str:
    p = Path(lora_path)
    return str(p.with_name(p.stem + SIDECAR_SUFFIX))


def sidecar_text(lora_name: str, words: list, dataset_dir: str,
                 trigger_word: str = "") -> str:
    """The paste-into-the-upload-form file, and where each word came from."""
    total = caption_count(dataset_dir)
    counts = caption_tags(dataset_dir)
    lines = [
        f"Trained words — {lora_name}",
        "",
        "Paste this into the Trained words / activation text box:",
        "",
        "    " + ", ".join(words) if words else "    (none — the captions carry no recurring words)",
        "",
        "Where each one comes from:",
        "",
    ]
    lookup = {k.casefold(): v for k, v in counts.items()}
    for word in words:
        seen = lookup.get(word.casefold(), 0)
        if trigger_word and word.casefold() == trigger_word.casefold():
            note = f"the set's trigger word — in {seen} of {total} captions"
        elif total:
            note = f"in {seen} of {total} captions"
        else:
            note = "set on Home"
        lines.append(f"    {word}  —  {note}")
    lines += [
        "",
        "The same list is written into the .safetensors metadata "
        "(ss_trained_words, ss_tag_frequency, modelspec.trigger_phrase); this file "
        "is here because not every upload form reads it back out any more.",
        "",
    ]
    return "\n".join(lines)


def stamp(lora_path: str, lora_name: str, dataset_dir: str, trigger_word: str = "",
          quality_prefix: str = "", plan: dict | None = None,
          image_count: int = 0) -> dict:
    """Write both copies for a finished run. Returns what was written.

    Raises MetadataError if the .safetensors could not be rewritten — the caller
    decides what that is worth, and it is never worth failing a finished run over.
    """
    meta = build_metadata(lora_name, dataset_dir, trigger_word, quality_prefix,
                          plan, image_count)
    words = trained_words(dataset_dir, trigger_word, quality_prefix)
    write_metadata(lora_path, meta)
    side = sidecar_path(lora_path)
    try:
        Path(side).write_text(
            sidecar_text(lora_name, words, dataset_dir, trigger_word),
            encoding="utf-8")
    except OSError as e:
        raise MetadataError(f"could not write {Path(side).name}: {e}") from e
    return {"metadata": meta, "words": words, "sidecar": side}
