"""Best-effort first-run guess of where the Anima model files live.

Scans common Stable-Diffusion install layouts (Forge, ComfyUI, A1111, etc.) for the
Anima files, so a fresh user often gets the model-scan folder prefilled instead of a
path baked to one machine. Returns "" when nothing is found — never guesses wrong.
"""
from pathlib import Path

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
