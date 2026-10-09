"""Which images OneTrainer will actually train on, and which it will silently skip.

OneTrainer sorts images into aspect-ratio buckets, then drops
`len(bucket) % batch_size` of them from every bucket, every epoch — mgds
AspectBatchSorting.__shuffle, "drop images for full buckets". The drop rotates
(AspectBatchSorting is a SingleVariationRandomAccessPipelineModule, so
current_variation = current_epoch and the shuffle is re-seeded each epoch), which
makes an odd bucket harmless: a different image sits out each time.

A bucket holding FEWER images than the batch size is a different matter. It yields
`int(len(bucket) / batch_size)` = 0 batches, so every image in it is dropped every
epoch, forever, with nothing in the log to say so. Bev and Billy (2026-08-17) lost
2 of 47 that way across a full 57-epoch run: one shot at aspect 1.600 and one at
0.687, each the only image of its shape in the set.

The bucket maths here mirrors mgds AspectBucketing.__create_automatic_buckets and
__get_bucket, with the quantization Krea2BaseDataLoader passes (64). Stdlib +
Pillow only — Pillow is already a declared dependency; numpy is not, so the
nearest-bucket search is a plain min() rather than np.argmin.
"""
import math
from pathlib import Path

from PIL import Image

# Krea2BaseDataLoader.py: aspect_bucketing_quantization=64
QUANTIZATION = 64

# Matches the locked preset (assets/krea2_preset.json).
RESOLUTION = 512
BATCH_SIZE = 2

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# mgds AspectBucketing.all_possible_input_aspects
_INPUT_ASPECTS = [
    (1.0, 1.0), (1.0, 1.25), (1.0, 1.5), (1.0, 1.75),
    (1.0, 2.0), (1.0, 2.5), (1.0, 3.0), (1.0, 3.5), (1.0, 4.0),
]


def _quantize(value: float) -> int:
    return round(value / QUANTIZATION) * QUANTIZATION


def bucket_resolutions(resolution: int = RESOLUTION) -> list:
    """Every (height, width) bucket available at this target resolution.

    Sorted, unlike mgds' set(), so ties in the nearest-bucket search resolve the
    same way every call. Exact aspect ties are vanishingly rare in real sets.
    """
    normalized = [(h / math.sqrt(h * w) * resolution, w / math.sqrt(h * w) * resolution)
                  for (h, w) in _INPUT_ASPECTS]
    normalized += [(w, h) for (h, w) in normalized]
    return sorted({(_quantize(h), _quantize(w)) for h, w in normalized})


def _bucket_for(h: int, w: int, resolutions: list) -> tuple:
    aspect = h / w
    return min(resolutions, key=lambda r: abs(r[0] / r[1] - aspect))


def bucket_plan(dataset_dir: str, resolution: int = RESOLUTION,
                batch_size: int = BATCH_SIZE) -> dict:
    """How this set actually batches.

    `stranded` is the finding that matters: images in a bucket too small to fill
    one batch, which are therefore never trained at all. An odd bucket is not
    stranded — its dropout rotates.
    """
    resolutions = bucket_resolutions(resolution)
    grouped = {}
    try:
        paths = sorted(p for p in Path(dataset_dir).iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    except OSError:
        paths = []

    for path in paths:
        try:
            with Image.open(path) as im:
                w, h = im.size
        except (OSError, ValueError):
            continue                      # unreadable: OneTrainer will not see it either
        if not w or not h:
            continue
        grouped.setdefault(_bucket_for(h, w, resolutions), []).append(path.name)

    stranded, steps, dropped, used = [], 0, 0, 0
    for names in grouped.values():
        batches = len(names) // batch_size
        steps += batches
        if batches == 0:
            stranded.extend(names)        # never forms a batch: never trained
            dropped += len(names)
        else:
            dropped += len(names) % batch_size
            used += batches * batch_size

    return {
        "buckets": grouped,
        "steps_per_epoch": steps,
        "stranded": sorted(stranded),
        "dropped_per_epoch": dropped,
        "used": used,
        "total": sum(len(v) for v in grouped.values()),
    }


def crop_stranded(dataset_dir: str, resolution: int = RESOLUTION,
                  batch_size: int = BATCH_SIZE) -> list:
    """Crop every stranded image into the nearest bucket that already trains.

    A stranded image's own bucket is too thin to ever form a batch; the fix is
    to make it belong to one that isn't, by trimming it to that bucket's aspect.
    Only the longer-relative side gives anything up, so this only ever removes
    pixels — never upscales, never touches the side that already fits.

    Bucket resolutions are a quantization at `resolution`, not the image's own
    scale, so only the aspect ratio they imply is used; the crop itself is
    computed against the image's real, full-resolution dimensions.

    Files are overwritten in place — the dataset folder IS what OneTrainer
    reads, there is no separate copy to redirect it at. Skipped, not fatal, if
    a file can't be read or nothing healthy exists to aim it at.
    """
    resolutions = bucket_resolutions(resolution)
    plan = bucket_plan(dataset_dir, resolution, batch_size)
    grouped = plan["buckets"]
    healthy = [r for r in resolutions if len(grouped.get(r, [])) >= batch_size]
    if not healthy:
        return []

    fixed = []
    for key, names in grouped.items():
        if len(names) >= batch_size:
            continue                      # already healthy, nothing to do
        target = min(healthy, key=lambda r: abs(r[0] / r[1] - key[0] / key[1]))
        target_aspect = target[0] / target[1]
        for name in names:
            path = Path(dataset_dir) / name
            try:
                with Image.open(path) as im:
                    w, h = im.size
                    fmt = im.format
                    aspect = h / w
                    if aspect > target_aspect:
                        new_w, new_h = w, max(1, round(w * target_aspect))
                    elif aspect < target_aspect:
                        new_w, new_h = max(1, round(h / target_aspect)), h
                    else:
                        continue           # already at the target aspect
                    left = (w - new_w) // 2
                    top = (h - new_h) // 2
                    cropped = im.crop((left, top, left + new_w, top + new_h))
                    cropped.load()
            except (OSError, ValueError):
                continue                  # unreadable: leave it for bucket_plan to skip
            cropped.save(path, format=fmt)
            fixed.append({"name": name, "old_size": (w, h), "new_size": (new_w, new_h)})
    return fixed


def crop_report(dataset_dir: str, resolution: int = RESOLUTION,
                batch_size: int = BATCH_SIZE) -> str:
    """Fix what can be fixed, then say what happened. Never "". Never a question.

    Cropping already-healthy images is no-cost (crop_stranded skips them), so
    this always resolves what it can before reporting. Anything left stranded
    afterward had nothing close enough in the set to crop toward.
    """
    fixed = crop_stranded(dataset_dir, resolution, batch_size)
    parts = []
    if fixed:
        names = "\n".join(f"    • {f['name']}" for f in fixed)
        parts.append(
            f"{len(fixed)} image(s) were alone at their aspect ratio and would "
            f"never have been trained, so they were cropped to match a shape "
            f"the rest of the set already trains at:\n\n{names}"
        )
    plan = bucket_plan(dataset_dir, resolution, batch_size)
    lost = plan["stranded"]
    if lost:
        names = "\n".join(f"    • {n}" for n in lost)
        parts.append(
            f"{len(lost)} of {plan['total']} images will never be trained — "
            f"nothing else in the set is close enough in aspect to crop them "
            f"toward:\n\n{names}"
        )
    return "\n\n".join(parts)
