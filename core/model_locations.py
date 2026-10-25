"""Finding the three Anima model files: the first-run scan-folder guess, and the
role-by-role detection Setup's Auto-detect runs inside that folder.

Scans common Stable-Diffusion install layouts (Forge, ComfyUI, A1111, etc.) for the
Anima files, so a fresh user often gets the model-scan folder prefilled instead of a
path baked to one machine. Returns "" when nothing is found — never guesses wrong.
"""
from pathlib import Path

from core.model_check import (
    ROLE_DIT, ROLE_FLUX2_VAE, ROLE_MOSS_DIT, ROLE_MOSS_TE, ROLE_TEXT_ENCODER,
    ROLE_VAE, identify,
)

# Folder-name fragments that commonly hold a 'models' tree.
_APP_HINTS = ("forge", "comfyui", "stable-diffusion", "sd", "automatic", "webui", "a1111")
_QWEN3 = "qwen_3_06b_base.safetensors"
_QWEN_VAE = "qwen_image_vae.safetensors"


def _default_roots():
    roots = []
    for drive in ("C:/", "D:/", "E:/", "F:/"):
        if Path(drive).is_dir():
            roots.append(drive)
    try:
        roots.append(str(Path.home()))
    except Exception:
        pass
    return roots


def _looks_like_app(dirname: str) -> bool:
    low = dirname.lower()
    return any(h in low for h in _APP_HINTS)


def _has_anima_files(models_dir: Path) -> bool:
    """True if this 'models' folder contains the Anima files (shallow recursive check)."""
    try:
        names = {p.name.lower() for p in models_dir.rglob("*.safetensors")}
    except (OSError, ValueError):
        return False
    if _QWEN3 in names and _QWEN_VAE in names:
        return True
    return any(n.startswith("anima") for n in names)


def _candidate_model_dirs(root: Path):
    """Yield plausible '<app>/models' (and '<app>/<sub>/models') dirs under root."""
    try:
        entries = [p for p in root.iterdir() if p.is_dir()]
    except (OSError, PermissionError):
        return
    for app in entries:
        if not _looks_like_app(app.name):
            continue
        direct = app / "models"
        if direct.is_dir():
            yield direct
        # one level deeper, e.g. <root>/Forge_neo/forge-neo/models
        try:
            for sub in app.iterdir():
                if sub.is_dir() and (sub / "models").is_dir():
                    yield sub / "models"
        except (OSError, PermissionError):
            continue


def guess_model_scan_dir(roots=None) -> str:
    """Return the first common 'models' folder that holds the Anima files, or ''."""
    roots = roots if roots is not None else _default_roots()
    for r in roots:
        root = Path(r)
        if not root.is_dir():
            continue
        for models_dir in _candidate_model_dirs(root):
            if _has_anima_files(models_dir):
                return str(models_dir)
    return ""


# ----------------------------------------------------------------------------
# Per-role detection inside the scan folder
# ----------------------------------------------------------------------------
#
# This used to be three hardcoded lookups in ui/setup_tab.py --
# <scan>/text_encoder/qwen_3_06b_base.safetensors, <scan>/VAE/qwen_image_vae.safetensors,
# <scan>/Stable-diffusion/anima*.safetensors -- i.e. the A1111/Forge tree with exact
# filenames. But circlestone-labs/Anima, the download our own README links to, publishes
# the ComfyUI tree: split_files/diffusion_models/, split_files/text_encoders/ (plural),
# split_files/vae/ (lowercase). Two of the three lookups missed outright; the VAE matched
# only because Windows compares paths case-insensitively. Users saw "auto detect can only
# see the vae", set the other two by hand, and a wrong pick then died inside the trainer.
#
# So role is decided by what a file *contains* (core.model_check reads the safetensors
# header — cheap, no tensor data), not by where it sits or what it is called. Directory
# and filename hints only choose the order headers get read in.

_ROLES = (ROLE_DIT, ROLE_TEXT_ENCODER, ROLE_VAE)

# Directory-name fragments that suggest a role. Order within a tuple is irrelevant;
# a hit only promotes a file up the queue.
_DIR_HINTS = {
    ROLE_DIT: ("diffusion_model", "stable-diffusion", "unet", "checkpoint"),
    ROLE_TEXT_ENCODER: ("text_encoder", "clip"),
    ROLE_VAE: ("vae",),
}

# MOSS-Image. Forge Neo names its folders diffusion_models / text_encoder / VAE, ComfyUI
# uses diffusion_models / text_encoders / vae; both are covered by these fragments.
# `_MOSS_PREFER` breaks ties among files that all identify correctly: Base DiT over SFT,
# the int8 text encoder over any other Qwen3.5 file, the exact flux2-vae name over a
# renamed copy. Earlier entry wins.
_MOSS_DIR_HINTS = {
    ROLE_DIT: ("diffusion_model", "unet", "checkpoint", "stable-diffusion"),
    ROLE_TEXT_ENCODER: ("text_encoder", "clip"),
    ROLE_VAE: ("vae",),
}
_MOSS_NAME_HINTS = {
    ROLE_DIT: ("moss",),
    ROLE_TEXT_ENCODER: ("qwen3.5", "qwen_3.5", "qwen35", "convrot", "int8"),
    ROLE_VAE: ("flux2", "vae"),
}
_MOSS_PREFER = {
    ROLE_DIT: ("moss_image_base_bf16", "base", "moss_image"),
    ROLE_TEXT_ENCODER: ("qwen3.5_9b_int8_convrot", "int8", "qwen3.5"),
    ROLE_VAE: ("flux2-vae", "flux2", "vae"),
}
_MOSS_ROLES = {ROLE_DIT: ROLE_MOSS_DIT, ROLE_TEXT_ENCODER: ROLE_MOSS_TE, ROLE_VAE: ROLE_FLUX2_VAE}

_NAME_HINTS = {
    ROLE_DIT: ("anima",),
    ROLE_TEXT_ENCODER: ("qwen_3", "qwen3", "06b", "0.6b", "text_encoder"),
    ROLE_VAE: ("vae",),
}

# Folders that never hold a base model but can hold thousands of safetensors.
_SKIP_DIRS = ("lora", "embedding", "controlnet", "ipadapter", "adetailer",
              "esrgan", "reactor", "insightface", "clip_vision", "hypernetwork")

MAX_FILES = 2000      # how many .safetensors to consider at all
MAX_HEADER_READS = 60  # how many headers to actually open per role


def _walk_safetensors(root: Path, limit: int = MAX_FILES):
    """.safetensors under `root`, skipping folders that can't hold a base model."""
    found = []
    stack = [root]
    while stack and len(found) < limit:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except (OSError, PermissionError):
            continue
        for entry in entries:
            try:
                if entry.is_dir():
                    if not any(s in entry.name.lower() for s in _SKIP_DIRS):
                        stack.append(entry)
                elif entry.suffix.lower() == ".safetensors":
                    found.append(entry)
                    if len(found) >= limit:
                        break
            except OSError:
                continue
    return found


def _rank_for(role: str, files, dir_hints=None, name_hints=None, prefer=()):
    """`files` ordered by how likely each is to be `role` — best guess first."""
    dir_hints = dir_hints or _DIR_HINTS
    name_hints = name_hints or _NAME_HINTS

    def pref(path: Path) -> int:
        name = path.name.lower()
        for i, frag in enumerate(prefer):
            if frag in name:
                return i
        return len(prefer)

    def score(path: Path) -> int:
        parent = path.parent.name.lower()
        name = path.name.lower()
        s = 0
        if any(h in parent for h in dir_hints[role]):
            s -= 2
        if any(h in name for h in name_hints[role]):
            s -= 1
        # 'vae' as a name hint is weak enough to drag the VAE's own folder-mates
        # along; a DiT sitting in a vae/ folder is not a thing worth ranking high.
        return s
    ranked = [f for f in files if score(f) < 0]
    ranked.sort(key=lambda p: (score(p), pref(p), str(p).lower()))
    # Files with no hint at all still get a look, after every hinted candidate --
    # that is what finds a wholly renamed file.
    rest = sorted((f for f in files if score(f) >= 0), key=lambda p: (pref(p), str(p).lower()))
    return ranked + rest


def detect_model_files(scan_root, model: str = "anima",
                       max_header_reads: int = MAX_HEADER_READS) -> dict:
    """{'dit':…, 'text_encoder':…, 'vae':…} of paths found under `scan_root`.

    A role maps to "" when nothing under `scan_root` actually contains it. Never
    returns a file whose header says it is something else -- in particular a
    ComfyUI-wrapped Anima merge is never offered as the DiT, because the trainer
    cannot load one.

    `model="moss"` looks for the MOSS-Image trio instead (same dict keys): the Base DiT,
    the Qwen3.5-9B text encoder and the FLUX.2 VAE. The W4A8 text encoder is never
    returned, because sd-scripts cannot load it.
    """
    moss = model == "moss"
    if model not in ("anima", "moss"):
        raise ValueError(f"unknown model {model!r}")
    empty = {role: "" for role in _ROLES}
    root = Path(scan_root) if scan_root else None
    if root is None or not root.is_dir():
        return empty

    files = _walk_safetensors(root)
    if not files:
        return empty

    out = dict(empty)
    identified = {}  # path -> role, so a header is never read twice
    for role in _ROLES:
        reads = 0
        want = _MOSS_ROLES[role] if moss else role
        if moss:
            ranked = _rank_for(role, files, _MOSS_DIR_HINTS, _MOSS_NAME_HINTS, _MOSS_PREFER[role])
        else:
            ranked = _rank_for(role, files)
        for path in ranked:
            if path in identified:
                if identified[path] == want and not out[role]:
                    out[role] = str(path)
                    break
                continue
            if reads >= max_header_reads:
                break
            reads += 1
            identified[path] = identify(path)
            if identified[path] == want:
                out[role] = str(path)
                break
    return out
