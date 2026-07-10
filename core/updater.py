"""Self-update from the public GitHub repo (zip download — no git required).

Works for both install styles (git clone and the Civitai zip): fetch the raw
core/version.py from `main` to see if a newer release exists, then download the
repo zipball and overlay its files onto the install. The zipball contains only
repo-tracked files, so user data (sets/, .venv/, sd-scripts/, settings,
datasets) is never in it and can't be touched. Overlay-only: files removed
upstream are left behind, which is harmless.

Overwriting .py files while the app runs is safe on Windows (source files
aren't locked; the running app keeps its in-memory code) — a restart picks up
the new version.
"""
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

REPO = "SillySilk/AnimaForge"
BRANCH = "main"
RAW_VERSION_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/core/version.py"
ZIPBALL_URL = f"https://github.com/{REPO}/archive/refs/heads/{BRANCH}.zip"
TIMEOUT_S = 15

_VERSION_RE = re.compile(r"__version__\s*=\s*[\"']([^\"']+)[\"']")

STAMP_NAME = "build_stamp.json"


def write_build_stamp(root, commit: str, built: str) -> bool:
    """Write build_stamp.json into `root`. Returns False on any OS error."""
    try:
        (Path(root) / STAMP_NAME).write_text(
            json.dumps({"commit": commit, "built": built}), encoding="utf-8")
        return True
    except OSError:
        return False


def read_build_stamp(root):
    """Return the commit SHA from root/build_stamp.json, or None if absent/malformed."""
    p = Path(root) / STAMP_NAME
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        sha = data.get("commit")
        return sha if isinstance(sha, str) and sha else None
    except (OSError, ValueError):
        return None


def parse_version(s: str) -> tuple:
    """'2.1.0' -> (2, 1, 0). Non-numeric segments count as 0."""
    parts = []
    for seg in (s or "").strip().split("."):
        try:
            parts.append(int(seg))
        except ValueError:
            parts.append(0)
    return tuple(parts) or (0,)


def is_newer(remote: str, local: str) -> bool:
    return parse_version(remote) > parse_version(local)


def extract_version(version_py_text: str):
    """Pull __version__ out of a core/version.py body. None if absent."""
    m = _VERSION_RE.search(version_py_text or "")
    return m.group(1) if m else None


def fetch_remote_version():
    """Latest version on GitHub main, or None on any network/parse failure."""
    try:
        with urllib.request.urlopen(RAW_VERSION_URL, timeout=TIMEOUT_S) as resp:
            return extract_version(resp.read().decode("utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


def download_and_extract(tmpdir: str) -> Path:
    """Download the repo zipball into tmpdir and extract it. Returns the inner
    repo root (e.g. tmpdir/AnimaForge-main). Raises on any failure — nothing in
    the install is touched until this has fully succeeded."""
    tmp = Path(tmpdir)
    zip_path = tmp / "update.zip"
    with urllib.request.urlopen(ZIPBALL_URL, timeout=TIMEOUT_S) as resp:
        zip_path.write_bytes(resp.read())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    roots = [p for p in tmp.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise ValueError(f"unexpected zip layout: {[p.name for p in roots]}")
    return roots[0]


def requirements_changed(new_root, app_root) -> bool:
    """True when the update ships a different requirements.txt (user should
    re-run install.bat before the next launch)."""
    old = Path(app_root) / "requirements.txt"
    new = Path(new_root) / "requirements.txt"
    old_text = old.read_text(encoding="utf-8", errors="replace") if old.is_file() else ""
    new_text = new.read_text(encoding="utf-8", errors="replace") if new.is_file() else ""
    return old_text.strip() != new_text.strip()


def apply_update(new_root, app_root) -> int:
    """Overlay-copy every file from the extracted repo onto the install.
    Returns the number of files written."""
    new_root = Path(new_root)
    app_root = Path(app_root)
    n = 0
    for src in new_root.rglob("*"):
        if not src.is_file():
            continue
        dst = app_root / src.relative_to(new_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    return n
