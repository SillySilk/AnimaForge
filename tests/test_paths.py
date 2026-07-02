import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.paths import (
    delivery_filename, from_portable, in_other_install, run_output_dir,
    sanitize_name, to_portable,
)


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


# ---- delivered-copy filename (trigger rides in the name) ----

def test_delivery_filename_appends_trigger():
    assert delivery_filename("MyChar_v1", "mychar") == "MyChar_v1_mychar.safetensors"


def test_delivery_filename_underscores_phrase():
    # multi-word trigger phrases are underscored — spaces don't survive filing
    assert delivery_filename("Style01", "my cool char") == "Style01_my_cool_char.safetensors"
    assert delivery_filename("Style01", "  my   cool char ") == "Style01_my_cool_char.safetensors"


def test_delivery_filename_no_trigger_plain():
    assert delivery_filename("MyChar_v1", "") == "MyChar_v1.safetensors"
    assert delivery_filename("MyChar_v1", "   ") == "MyChar_v1.safetensors"


def test_delivery_filename_skips_redundant_trigger():
    # name == trigger, or already suffixed: don't stutter
    assert delivery_filename("mychar", "MyChar") == "mychar.safetensors"
    assert delivery_filename("MyChar_v1_mychar", "mychar") == "MyChar_v1_mychar.safetensors"


def test_delivery_filename_sanitizes():
    assert delivery_filename("a/b", 'tr:ig') == "a_b_tr_ig.safetensors"


# ---- portable (per-install) path storage ----

def test_to_portable_relativizes_inside_root(tmp_path):
    root = tmp_path / "AnimaForge"
    inside = root / "sd-scripts"
    assert to_portable(str(inside), str(root)) == "sd-scripts"
    # case-insensitive on Windows-style stores
    assert to_portable(str(inside), str(root).upper()) == "sd-scripts"


def test_to_portable_keeps_external_absolute(tmp_path):
    root = tmp_path / "AnimaForge"
    external = tmp_path / "models" / "anima.safetensors"
    assert to_portable(str(external), str(root)) == str(external)
    assert to_portable("", str(root)) == ""


def test_from_portable_resolves_under_root(tmp_path):
    root = tmp_path / "AnimaForge"
    assert Path(from_portable("sd-scripts", str(root))) == root / "sd-scripts"
    ext = str(tmp_path / "elsewhere")
    assert from_portable(ext, str(root)) == ext
    assert from_portable("", str(root)) == ""


def test_portable_round_trip(tmp_path):
    root_a = tmp_path / "copyA"
    root_b = tmp_path / "copyB"
    stored = to_portable(str(root_a / "output"), str(root_a))
    # the same stored value resolves inside whichever install loads it
    assert Path(from_portable(stored, str(root_b))) == root_b / "output"


def test_in_other_install_detects_foreign_copy(tmp_path):
    ours = tmp_path / "ours"
    theirs = tmp_path / "theirs"
    for r in (ours, theirs):
        (r / "sd-scripts").mkdir(parents=True)
        (r / "main.py").write_text("", encoding="utf-8")
        (r / "launch.bat").write_text("", encoding="utf-8")
    assert in_other_install(str(theirs / "sd-scripts"), str(ours))
    assert in_other_install(str(theirs / "output"), str(ours))
    # our own paths and non-install locations are not foreign
    assert not in_other_install(str(ours / "sd-scripts"), str(ours))
    kohya = tmp_path / "kohya" / "sd-scripts"
    kohya.mkdir(parents=True)
    assert not in_other_install(str(kohya), str(ours))
    assert not in_other_install("", str(ours))
