"""Read-only folder browsing on this PC, for picking a run's training images, plus
small cached thumbnails so a slow link can still see what is in a folder."""
import hashlib
import os
import string
import threading
import uuid
from pathlib import Path

from remote.control import IMAGE_EXTS

SAMPLE_IMAGES = 12
THUMB_SIZE = 256
_BUILD_SLOTS = threading.BoundedSemaphore(3)
_KEY_LOCKS: dict = {}
_KEY_LOCKS_GUARD = threading.Lock()


def drives() -> list:
    return [f"{c}:\\" for c in string.ascii_uppercase if os.path.isdir(f"{c}:\\")]


def browse(path: str):
    """{path, parent, folders: [{name, path}], image_count, images: [paths]} for a
    folder, the list of drives for an empty path, or None for anything else."""
    if not path:
        return {"path": "", "parent": None, "image_count": 0, "images": [],
                "folders": [{"name": d, "path": d} for d in drives()]}
    p = Path(path)
    if not p.is_absolute() or not p.is_dir():
        return None
    folders, images = [], []
    try:
        entries = list(os.scandir(p))
    except OSError:
        entries = []
    for e in entries:
        if e.name.startswith((".", "$")):
            continue
        try:
            if e.is_dir():
                folders.append({"name": e.name, "path": e.path})
            elif Path(e.name).suffix.lower() in IMAGE_EXTS:
                images.append(e.path)
        except OSError:
            continue
    folders.sort(key=lambda f: f["name"].lower())
    images.sort(key=lambda s: Path(s).name.lower())
    parent = str(p.parent) if p.parent != p else ""
    return {"path": str(p), "parent": parent, "folders": folders,
            "image_count": len(images), "images": images[:SAMPLE_IMAGES]}


def _key_lock(key: str) -> threading.Lock:
    with _KEY_LOCKS_GUARD:
        return _KEY_LOCKS.setdefault(key, threading.Lock())


def thumbnail(image: Path, cache_dir: Path) -> Path:
    """A cached small WebP of `image`, rebuilt when the image changes."""
    from PIL import Image
    st = image.stat()
    key = hashlib.sha1(f"{image}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()
    out = Path(cache_dir) / f"{key}.webp"
    if out.exists():
        return out
    with _key_lock(key), _BUILD_SLOTS:
        if out.exists():
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{key}.{uuid.uuid4().hex}.tmp")
        try:
            with Image.open(image) as im:
                im.thumbnail((THUMB_SIZE, THUMB_SIZE))
                im.convert("RGB").save(tmp, "WEBP", quality=70, method=4)
            os.replace(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)
    return out
