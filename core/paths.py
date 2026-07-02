"""Per-run output path helpers.

Every training run gets its own folder so the LoRA, its `sample/` previews, logs,
configs, and resumable state live together and never bleed across runs (sd-scripts
hardcodes samples to `{output_dir}/sample`, so isolating previews means isolating the
output dir). See docs/superpowers/specs/2026-06-24-train-workflow-and-name-cast-polish-design.md.
"""
import os
import re
from pathlib import Path

# Characters illegal in Windows filenames (plus path separators), collapsed to "_".
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Files that mark a directory as an AnimaForge install root. Used to recognize
# settings that leaked from ANOTHER copy of the app (QSettings is machine-global,
# so running two installs side by side shares one store — user feedback: a new
# copy silently trained with the old copy's sd-scripts and output folder).
_INSTALL_MARKERS = ("main.py", "launch.bat")


def app_root() -> Path:
    """This install's root directory (the folder holding main.py)."""
    return Path(__file__).resolve().parents[1]


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def _is_under(path: str, root: str) -> bool:
    np, nr = _norm(path), _norm(root)
    return np == nr or np.startswith(nr + os.sep)


def to_portable(path: str, root: str | None = None) -> str:
    """Make an install-internal path relative to the app root for storage.

    Paths outside the install are returned unchanged (deliberate external
    locations stay absolute). Case-insensitive on purpose — Windows paths.
    """
    path = (path or "").strip()
    if not path:
        return ""
    root = str(root or app_root())
    if not _is_under(path, root):
        return path
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    return "" if rel == "." else rel


def from_portable(value: str, root: str | None = None) -> str:
    """Resolve a stored path: relative values live under *this* install's root."""
    value = (value or "").strip()
    if not value:
        return ""
    if os.path.isabs(value):
        return value
    return str(Path(root or app_root()) / value)


def in_other_install(path: str, root: str | None = None) -> bool:
    """True when `path` sits inside a different AnimaForge install than `root`.

    Detected by walking the path's ancestors for the install marker files. Paths
    under this install, or anywhere that isn't an AnimaForge install (a user's
    own kohya clone, an external output drive), return False.
    """
    path = (path or "").strip()
    if not path:
        return False
    root = str(root or app_root())
    if _is_under(path, root):
        return False
    p = Path(os.path.abspath(path))
    for anc in (p, *p.parents):
        try:
            if all((anc / m).is_file() for m in _INSTALL_MARKERS):
                return True
        except OSError:
            continue
    return False


def sanitize_name(lora_name: str) -> str:
    """Return a filesystem-safe folder name for a LoRA name (trims trailing dots/spaces)."""
    safe = _ILLEGAL.sub("_", (lora_name or "").strip())
    return safe.rstrip(". ")


def delivery_filename(lora_name: str, trigger_word: str = "") -> str:
    """Filename for a *delivered* LoRA copy: ``{name}_{trigger}.safetensors``.

    The trigger rides in the filename so end users can read the activation word
    or phrase straight off the file. Multi-word phrases are underscored (spaces
    don't survive filing systems / prompt tags reliably). Skipped when no trigger
    is set, or when the name already carries it (name == trigger or already
    suffixed). The training output keeps the plain name — this is only for the
    copies handed to Forge / ComfyUI.
    """
    name = sanitize_name(lora_name)
    trig = "_".join(sanitize_name((trigger_word or "").strip()).split())
    if trig and trig.lower() != name.lower() and not name.lower().endswith("_" + trig.lower()):
        name = f"{name}_{trig}"
    return f"{name}.safetensors"


def run_output_dir(base_dir: str, lora_name: str) -> str:
    """Per-run output folder `{base_dir}/{sanitized lora_name}`.

    Falls back to `base_dir` when either input is empty so callers stay safe before a
    name/output is set.
    """
    safe = sanitize_name(lora_name)
    if not base_dir or not safe:
        return base_dir or ""
    return str(Path(base_dir) / safe)
