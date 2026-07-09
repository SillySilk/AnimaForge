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


def test_policy_radio_reflects_and_writes_the_setting():
    from core.caption_policy import ASK, OVERWRITE
    s = SetupTab()
    prev_policy = s._app.get("caption_existing_policy")
    try:
        assert s._policy_buttons[ASK].isChecked()
        s._policy_buttons[OVERWRITE].setChecked(True)
        assert s.get_app_settings().get("caption_existing_policy") == OVERWRITE
    finally:
        s._app.set("caption_existing_policy", prev_policy)


def test_setup_tab_survives_an_unrecognized_policy_in_the_store():
    """SetupTab is built unconditionally at app start; a corrupted preference must not
    take the whole app down with a KeyError."""
    from core.caption_policy import ASK
    app = SetupTab().get_app_settings()
    prev = app.get("caption_existing_policy")
    try:
        app.set("caption_existing_policy", "nonsense-value")
        t = SetupTab()                       # must not raise
        assert t._policy_buttons[ASK].isChecked()
    finally:
        app.set("caption_existing_policy", prev)
