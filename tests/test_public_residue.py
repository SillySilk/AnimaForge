"""This repo is public: tracked files must not carry the maintainer's machine layout."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"(?i)(C:\\{1,2}Users\\|C:\\{1,2}AI\\|C:/AI/|/c/AI/|G:\\{1,2}LORAS|littledale)")
# The built-in output folder is published on purpose (maintainer decision 2026-10-13);
# only that exact path is let through, in its plain and escaped spellings.
ALLOWED = re.compile(r"(?i)C:\\{1,2}AI\\{1,2}Apps\\{1,2}Output\\{1,2}Lora Forge")
TEXT = {".py", ".bat", ".json", ".toml", ".txt", ".md", ".yml", ".yaml", ".cfg", ".ini", ".ps1"}


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [ROOT / p for p in out.stdout.splitlines() if Path(p).suffix.lower() in TEXT]


def test_no_tracked_file_names_the_maintainers_paths():
    hits = []
    for f in tracked_files():
        if f.name == "test_public_residue.py" or not f.is_file():
            continue
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if FORBIDDEN.search(ALLOWED.sub("", line)):
                hits.append(f"{f.relative_to(ROOT)}:{n}: {line.strip()[:100]}")
    assert hits == []


def test_psutil_is_a_declared_dependency():
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"(?m)^psutil\b", reqs)
