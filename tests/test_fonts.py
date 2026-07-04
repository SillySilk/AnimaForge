import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from utils import fonts
from utils.styles import build_stylesheet

_app = QApplication.instance() or QApplication([])

ROLES = {"display", "marker", "type", "body"}


def _reset():
    fonts.ACTIVE_FAMILIES.clear()


def test_forge_mode_is_byte_identical_to_loaded_families():
    loaded = fonts.load_app_fonts()
    assert fonts.resolve_families("forge") == dict(loaded)
    assert build_stylesheet(fonts.resolve_families("forge")) == build_stylesheet(loaded)


def test_system_mode_collapses_all_roles_to_ui_stack():
    fam = fonts.resolve_families("system")
    assert set(fam) == ROLES
    assert all(stack == fonts.UI_STACK for stack in fam.values())


def test_custom_mode_injects_family_and_keeps_fallbacks():
    fam = fonts.resolve_families("custom", "Arial")
    assert set(fam) == ROLES
    for stack in fam.values():
        assert stack.startswith('"Arial"')
        assert '"Segoe UI Symbol"' in stack
        assert '"Malgun Gothic"' in stack


def test_custom_without_family_falls_back_to_forge():
    fonts.load_app_fonts()
    assert fonts.resolve_families("custom", "") == fonts.resolve_families("forge")
    assert fonts.resolve_families("custom", None) == fonts.resolve_families("forge")


def test_unknown_mode_falls_back_to_forge():
    fonts.load_app_fonts()
    assert fonts.resolve_families("bogus") == fonts.resolve_families("forge")


def test_family_prefers_active_families():
    _reset()
    fonts.load_app_fonts()
    fonts.ACTIVE_FAMILIES.update(fonts.resolve_families("custom", "Arial"))
    assert fonts.family("display").startswith('"Arial"')
    _reset()
    assert "Arial" not in fonts.family("display")


def test_primary_family_extracts_first_name():
    _reset()
    fonts.load_app_fonts()
    assert fonts.primary_family("type") == "Special Elite"
    fonts.ACTIVE_FAMILIES.update(fonts.resolve_families("custom", "Arial"))
    assert fonts.primary_family("type") == "Arial"
    _reset()


def test_apply_app_font_sets_stylesheet_and_cache():
    _reset()
    fonts.load_app_fonts()
    fams = fonts.apply_app_font(_app)
    assert fams == fonts.ACTIVE_FAMILIES
    assert _app.styleSheet() == build_stylesheet(fams)
    _app.setStyleSheet("")
    _reset()
