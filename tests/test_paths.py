import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.paths import run_output_dir, sanitize_name


def test_composes_run_dir():
    got = run_output_dir("/out", "my_char_v1")
    assert Path(got) == Path("/out") / "my_char_v1"


def test_empty_name_falls_back_to_base():
    assert run_output_dir("/out", "") == "/out"
    assert run_output_dir("/out", "   ") == "/out"


def test_empty_base_returns_empty():
    assert run_output_dir("", "name") == ""


def test_sanitizes_illegal_chars():
    assert sanitize_name('a/b:c*?') == "a_b_c__"
    assert sanitize_name("  trailing.  ") == "trailing"
    # composed path uses the sanitized leaf
    assert Path(run_output_dir("/out", "a/b")).name == "a_b"
