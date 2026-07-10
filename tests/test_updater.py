import io
import json as _json
import sys
import urllib.error
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


def test_apply_update_writes_stamp_when_commit_given(tmp_path):
    new_root = tmp_path / "new"; new_root.mkdir()
    (new_root / "a.py").write_text("x", encoding="utf-8")
    app_root = tmp_path / "app"; app_root.mkdir()
    n = updater.apply_update(new_root, app_root, commit="a" * 40, built="2026-07-10")
    assert n == 1
    assert updater.read_build_stamp(app_root) == "a" * 40


def test_apply_update_without_commit_writes_no_stamp(tmp_path):
    new_root = tmp_path / "new"; new_root.mkdir()
    (new_root / "a.py").write_text("x", encoding="utf-8")
    app_root = tmp_path / "app"; app_root.mkdir()
    updater.apply_update(new_root, app_root)
    assert updater.read_build_stamp(app_root) is None


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


# ---- compare_to_main ----

def _fake_opener(payload=None, *, http_error=None):
    def opener(url, timeout=None):
        if http_error is not None:
            raise urllib.error.HTTPError(url, http_error, "err", None, None)
        return io.BytesIO(_json.dumps(payload).encode("utf-8"))
    return opener


def test_compare_ahead_extracts_fields():
    payload = {"status": "ahead", "ahead_by": 12,
               "commits": [{"sha": "f" * 40,
                            "commit": {"message": "fix: batch icons\n\nbody"}}]}
    out = updater.compare_to_main("a" * 40, opener=_fake_opener(payload))
    assert out == {"status": "ahead", "ahead_by": 12,
                   "head_sha": "f" * 40, "latest_subject": "fix: batch icons"}


def test_compare_identical():
    payload = {"status": "identical", "ahead_by": 0, "commits": []}
    assert updater.compare_to_main("a" * 40,
                                   opener=_fake_opener(payload)) == {"status": "identical"}


def test_compare_404_is_not_found():
    out = updater.compare_to_main("a" * 40, opener=_fake_opener(http_error=404))
    assert out == {"status": "not_found"}


def test_compare_network_error_is_error():
    def boom(url, timeout=None):
        raise OSError("no net")
    assert updater.compare_to_main("a" * 40, opener=boom) == {"status": "error"}


def test_compare_non_integer_ahead_by_is_error():
    payload = {"status": "ahead", "ahead_by": "lots",
               "commits": [{"sha": "f" * 40, "commit": {"message": "x"}}]}
    assert updater.compare_to_main("a" * 40,
                                   opener=_fake_opener(payload)) == {"status": "error"}


def test_compare_non_404_http_error_is_error():
    out = updater.compare_to_main("a" * 40, opener=_fake_opener(http_error=500))
    assert out == {"status": "error"}


# ---- fetch_remote_head ----

def test_fetch_remote_head_ok():
    assert updater.fetch_remote_head(opener=_fake_opener({"sha": "a" * 40})) == "a" * 40


def test_fetch_remote_head_non_dict_is_none():
    assert updater.fetch_remote_head(opener=_fake_opener([1, 2, 3])) is None


def test_fetch_remote_head_network_error_is_none():
    def boom(url, timeout=None):
        raise OSError("no net")
    assert updater.fetch_remote_head(opener=boom) is None


# ---- build_update_decision ----

def _ahead(head="f" * 40, n=12, subj="fix: x"):
    return {"status": "ahead", "ahead_by": n, "head_sha": head, "latest_subject": subj}


def test_decision_ahead_prompts():
    d = updater.build_update_decision(
        _ahead(), remote_version="2.6.0", local_version="2.5.0", skipped_commit="")
    assert d["head_sha"] == "f" * 40 and d["ahead_by"] == 12
    assert d["latest_subject"] == "fix: x"
    assert d["remote_version"] == "2.6.0" and d["local_version"] == "2.5.0"


def test_decision_ahead_but_version_not_bumped_hides_version_line():
    d = updater.build_update_decision(
        _ahead(), remote_version="2.5.0", local_version="2.5.0", skipped_commit="")
    assert d is not None and d["remote_version"] is None   # commits behind, no bump


def test_decision_skipped_commit_silences():
    d = updater.build_update_decision(
        _ahead(head="f" * 40), remote_version="2.6.0", local_version="2.5.0",
        skipped_commit="f" * 40)
    assert d is None


def test_decision_identical_no_prompt():
    assert updater.build_update_decision(
        {"status": "identical"}, remote_version="2.5.0", local_version="2.5.0",
        skipped_commit="") is None


def test_decision_error_status_silent():
    assert updater.build_update_decision(
        {"status": "error"}, remote_version="9.9.9", local_version="2.5.0",
        skipped_commit="") is None


def test_decision_not_found_falls_back_to_version():
    d = updater.build_update_decision(
        {"status": "not_found"}, remote_version="2.6.0", local_version="2.5.0",
        skipped_commit="")
    assert d is not None and d["ahead_by"] is None and d["head_sha"] is None
    assert d["remote_version"] == "2.6.0"


def test_decision_not_found_no_newer_version_silent():
    assert updater.build_update_decision(
        {"status": "not_found"}, remote_version="2.5.0", local_version="2.5.0",
        skipped_commit="") is None
