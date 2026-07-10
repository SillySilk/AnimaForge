import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import updater


# ---- version parsing / comparison ----

def test_parse_version():
    assert updater.parse_version("2.1.0") == (2, 1, 0)
    assert updater.parse_version("2.10") == (2, 10)
    assert updater.parse_version("") == (0,)
    assert updater.parse_version("2.x.1") == (2, 0, 1)


def test_is_newer():
    assert updater.is_newer("2.1.0", "2.0.0")
    assert updater.is_newer("2.10.0", "2.9.9")   # numeric, not lexicographic
    assert not updater.is_newer("2.1.0", "2.1.0")
    assert not updater.is_newer("2.0.9", "2.1.0")


def test_extract_version_from_version_py():
    body = '"""docstring"""\n\n__version__ = "2.1.0"\n'
    assert updater.extract_version(body) == "2.1.0"
    assert updater.extract_version("no version here") is None
    assert updater.extract_version("__version__ = '3.0'") == "3.0"


# ---- overlay apply ----

def _fake_repo(root: Path) -> Path:
    repo = root / "AnimaForge-main"
    (repo / "core").mkdir(parents=True)
    (repo / "core" / "version.py").write_text('__version__ = "9.9.9"\n', encoding="utf-8")
    (repo / "main.py").write_text("print('new')\n", encoding="utf-8")
    (repo / "requirements.txt").write_text("PySide6\nnewdep\n", encoding="utf-8")
    return repo


def test_apply_update_overlays_files_and_preserves_user_data(tmp_path: Path):
    new_root = _fake_repo(tmp_path / "extracted")
    app = tmp_path / "app"
    (app / "core").mkdir(parents=True)
    (app / "core" / "version.py").write_text('__version__ = "2.1.0"\n', encoding="utf-8")
    (app / "main.py").write_text("print('old')\n", encoding="utf-8")
    # User data that must survive (not present in the repo zip)
    (app / "sets").mkdir()
    (app / "sets" / "Eve.json").write_text("{}", encoding="utf-8")

    n = updater.apply_update(new_root, app)
    assert n == 3
    assert (app / "main.py").read_text(encoding="utf-8") == "print('new')\n"
    assert '9.9.9' in (app / "core" / "version.py").read_text(encoding="utf-8")
    assert (app / "sets" / "Eve.json").is_file()  # untouched


def test_apply_update_creates_new_directories(tmp_path: Path):
    new_root = _fake_repo(tmp_path / "extracted")
    (new_root / "ui").mkdir()
    (new_root / "ui" / "new_tab.py").write_text("x = 1\n", encoding="utf-8")
    app = tmp_path / "app"
    app.mkdir()
    updater.apply_update(new_root, app)
    assert (app / "ui" / "new_tab.py").is_file()


def test_requirements_changed(tmp_path: Path):
    new_root = _fake_repo(tmp_path / "extracted")
    app = tmp_path / "app"
    app.mkdir()
    (app / "requirements.txt").write_text("PySide6\n", encoding="utf-8")
    assert updater.requirements_changed(new_root, app)
    (app / "requirements.txt").write_text("PySide6\nnewdep\n", encoding="utf-8")
    assert not updater.requirements_changed(new_root, app)


def test_requirements_changed_ignores_whitespace(tmp_path: Path):
    new_root = _fake_repo(tmp_path / "extracted")
    app = tmp_path / "app"
    app.mkdir()
    (app / "requirements.txt").write_text("PySide6\nnewdep", encoding="utf-8")
    assert not updater.requirements_changed(new_root, app)


# ---- zip layout validation (extraction half of download_and_extract) ----

def test_build_stamp_roundtrip(tmp_path):
    assert updater.read_build_stamp(tmp_path) is None
    assert updater.write_build_stamp(tmp_path, "a" * 40, "2026-07-10") is True
    assert updater.read_build_stamp(tmp_path) == "a" * 40


def test_read_build_stamp_ignores_garbage(tmp_path):
    (tmp_path / "build_stamp.json").write_text("not json", encoding="utf-8")
    assert updater.read_build_stamp(tmp_path) is None


def test_read_build_stamp_ignores_non_dict_json(tmp_path):
    (tmp_path / "build_stamp.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert updater.read_build_stamp(tmp_path) is None


def test_zip_extract_layout(tmp_path: Path):
    # Build a GitHub-style zipball: everything under one AnimaForge-main/ root.
    src = _fake_repo(tmp_path / "build")
    zip_path = tmp_path / "update.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, f"AnimaForge-main/{p.relative_to(src)}")
    out = tmp_path / "out"
    out.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)
    roots = [p for p in out.iterdir() if p.is_dir()]
    assert len(roots) == 1 and roots[0].name == "AnimaForge-main"


# ---- local_commit resolution ----

def test_local_commit_prefers_git(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(updater, "_git_head", lambda root: "b" * 40)
    assert updater.local_commit(tmp_path) == "b" * 40


def test_local_commit_falls_back_to_stamp(tmp_path):
    # no .git dir
    updater.write_build_stamp(tmp_path, "c" * 40, "2026-07-10")
    assert updater.local_commit(tmp_path) == "c" * 40


def test_local_commit_none_when_nothing(tmp_path):
    assert updater.local_commit(tmp_path) is None


# ---- on_main_branch guard ----

def test_on_main_branch_true_without_git(tmp_path):
    assert updater.on_main_branch(tmp_path) is True


def test_on_main_branch_true_when_head_is_main(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(updater, "_git_rev", lambda root, ref: "d" * 40)
    assert updater.on_main_branch(tmp_path) is True


def test_on_main_branch_false_when_head_differs(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(updater, "_git_rev",
                        lambda root, ref: ("d" * 40) if ref == "HEAD" else ("e" * 40))
    assert updater.on_main_branch(tmp_path) is False
