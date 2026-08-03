import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import af_log


def test_append_creates_file_and_writes_line(tmp_path):
    p = tmp_path / "caption.log"
    af_log.append("hello", path=p)
    assert "hello" in p.read_text(encoding="utf-8")


def test_append_creates_missing_parent_dir(tmp_path):
    p = tmp_path / "logs" / "caption.log"
    af_log.append("hello", path=p)
    assert p.is_file()


def test_append_stamps_each_line(tmp_path):
    p = tmp_path / "caption.log"
    af_log.append("hello", path=p)
    line = p.read_text(encoding="utf-8").strip()
    # "2026-08-02 14:03:11  hello"
    assert line.endswith("hello")
    assert line[:4].isdigit()


def test_append_is_utf8_safe(tmp_path):
    p = tmp_path / "caption.log"
    af_log.append("caption — 日本語 ✓", path=p)
    assert "日本語" in p.read_text(encoding="utf-8")


def test_append_accumulates_lines(tmp_path):
    p = tmp_path / "caption.log"
    for i in range(3):
        af_log.append(f"line {i}", path=p)
    assert len(p.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_rotation_moves_current_to_previous(tmp_path):
    p = tmp_path / "caption.log"
    af_log.append("x" * 200, path=p, max_bytes=100)
    af_log.append("fresh", path=p, max_bytes=100)
    assert p.with_suffix(".log.1").is_file()
    assert p.read_text(encoding="utf-8").strip().endswith("fresh")


def test_rotation_replaces_an_existing_previous(tmp_path):
    p = tmp_path / "caption.log"
    p.with_suffix(".log.1").write_text("old", encoding="utf-8")
    af_log.append("y" * 200, path=p, max_bytes=100)
    af_log.append("fresh", path=p, max_bytes=100)
    assert "old" not in p.with_suffix(".log.1").read_text(encoding="utf-8")


def test_append_never_raises_on_a_bad_path(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    af_log.append("hello", path=blocker / "nested" / "caption.log")  # must not raise
