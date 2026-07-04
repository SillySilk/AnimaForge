import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from ui.setup_tab import SetupTab

_app = QApplication.instance() or QApplication([])


def test_fine_tuning_groups_are_stashed_for_modals():
    s = SetupTab()
    # App Defaults + Advanced Training are hidden (shown in Fine-Tuning modals, not inline).
    assert s._advanced_group.isHidden() and s._defaults_group.isHidden()
    assert hasattr(s, "_open_defaults_modal") and hasattr(s, "_open_advanced_modal")


def test_setting_modal_reparents_and_restashes():
    s = SetupTab()
    grp = s._defaults_group
    s._open_defaults_modal()
    assert grp.isHidden() is False          # shown inside the modal
    s._restash_setting(grp)                  # what modal.closed does
    assert grp.isHidden() is True and grp.parent() is s


def test_font_mode_combo_saves_and_toggles_family_combo():
    s = SetupTab()
    prev_mode = s._app.get("ui_font_mode")
    prev_family = s._app.get("ui_font_family")
    try:
        s._font_mode_combo.setCurrentIndex(2)   # Custom…
        assert s._app.get("ui_font_mode") == "custom"
        assert not s._font_family_combo.isHidden()
        # picking a font in custom mode persists the family name
        assert s._app.get("ui_font_family") != ""
        s._font_mode_combo.setCurrentIndex(1)   # System font
        assert s._app.get("ui_font_mode") == "system"
        assert s._font_family_combo.isHidden()
        s._font_mode_combo.setCurrentIndex(0)   # Forge fonts (default)
        assert s._app.get("ui_font_mode") == "forge"
    finally:
        s._app.set("ui_font_mode", prev_mode)
        s._app.set("ui_font_family", prev_family)
