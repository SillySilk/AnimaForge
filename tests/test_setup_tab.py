import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication, QMessageBox
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


def test_caption_rules_table_loads_without_write_back_and_saves_on_edit():
    """The rules table is populated in _bind_app_widgets BEFORE itemChanged is connected,
    so constructing SetupTab from a seeded store must not alter that store. Editing a cell
    afterwards (connection now live) must persist."""
    from core.caption_rules import dump_rules, parse_rules
    from core.settings import AppSettings
    app_probe = AppSettings()
    prev = app_probe.get("caption_rules_json")
    try:
        seeded = dump_rules([("man", "woman")])
        app_probe.set("caption_rules_json", seeded)
        s = SetupTab()
        assert s._app.get("caption_rules_json") == seeded      # load did not write back
        assert s._rules_table.rowCount() == 1
        assert s._rules_table.item(0, 0).text() == "man"
        assert s._rules_table.item(0, 1).text() == "woman"
        s._rules_table.item(0, 1).setText("woman!!")           # edit after load
        assert parse_rules(s._app.get("caption_rules_json")) == [("man", "woman!!")]
    finally:
        app_probe.set("caption_rules_json", prev)


def test_add_and_remove_rule_row():
    from core.caption_rules import parse_rules
    from core.settings import AppSettings
    app_probe = AppSettings()
    prev = app_probe.get("caption_rules_json")
    try:
        s = SetupTab()
        s._rules_table.setRowCount(0)
        s._add_rule_row()
        assert s._rules_table.rowCount() == 1
        s._rules_table.item(0, 0).setText("1boy")
        assert parse_rules(s._app.get("caption_rules_json")) == [("1boy", "")]
        s._rules_table.selectRow(0)
        s._remove_selected_rule()
        assert s._rules_table.rowCount() == 0
        assert parse_rules(s._app.get("caption_rules_json")) == []
    finally:
        app_probe.set("caption_rules_json", prev)


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


class _FakeProc:
    """Stands in for the QProcess the version probe reads from."""
    def __init__(self, text):
        self._text = text.encode("utf-8")

    def readAllStandardOutput(self):
        return self._text


def test_parse_torch_probe_reads_the_version_line():
    from ui.setup_tab import parse_torch_probe
    assert parse_torch_probe("torch 2.5.1+cu121 cuda 12.1") == (2, 5)
    assert parse_torch_probe("torch 2.10.0+cu128 cuda 12.8") == (2, 10)
    assert parse_torch_probe("ModuleNotFoundError: No module named 'torch'") is None
    assert parse_torch_probe("") is None


def test_detected_pytorch_survives_a_restart():
    """Home's health row read an in-memory flag that reset to False on every launch, so
    PyTorch showed as missing until the user clicked Check PyTorch Version — every time.
    A successful probe is now remembered."""
    s = SetupTab()
    prev = s._settings.value("torch_version", "", type=str)
    try:
        s._handle_torch_check_output(_FakeProc("torch 2.5.1+cu121 cuda 12.1"))
        assert s.is_pytorch_ok()
        assert SetupTab().is_pytorch_ok()        # a fresh instance == an app restart
    finally:
        s._settings.setValue("torch_version", prev)
        s._settings.sync()


def test_a_too_old_torch_is_remembered_as_not_ok():
    s = SetupTab()
    prev = s._settings.value("torch_version", "", type=str)
    try:
        s._handle_torch_check_output(_FakeProc("torch 2.1.0+cu118 cuda 11.8"))
        assert not s.is_pytorch_ok()
        assert not SetupTab().is_pytorch_ok()
    finally:
        s._settings.setValue("torch_version", prev)
        s._settings.sync()


def test_unparseable_output_does_not_erase_a_known_good_result():
    """A probe that fails to run (interpreter missing, import error) must not flip a
    previously-detected PyTorch to 'missing' on the Home health row."""
    s = SetupTab()
    prev = s._settings.value("torch_version", "", type=str)
    try:
        s._handle_torch_check_output(_FakeProc("torch 2.5.1+cu121 cuda 12.1"))
        s._handle_torch_check_output(_FakeProc("Traceback (most recent call last):"))
        assert s.is_pytorch_ok()
    finally:
        s._settings.setValue("torch_version", prev)
        s._settings.sync()


def test_probe_emits_status_change_so_home_can_refresh():
    s = SetupTab()
    prev = s._settings.value("torch_version", "", type=str)
    seen = []
    s.pytorch_status_changed.connect(lambda: seen.append(True))
    try:
        s._handle_torch_check_output(_FakeProc("torch 2.5.1+cu121 cuda 12.1"))
        assert seen
    finally:
        s._settings.setValue("torch_version", prev)
        s._settings.sync()


def test_startup_update_toggle_roundtrips():
    s = SetupTab()
    prev = s._app.get("startup_update_check")
    try:
        s._startup_update_check.setChecked(False)
        assert s._app.get("startup_update_check") is False
        s._startup_update_check.setChecked(True)
        assert s._app.get("startup_update_check") is True
    finally:
        s._app.set("startup_update_check", prev)


def test_force_libuv_checkbox_roundtrips():
    s = SetupTab()
    prev = s._app.get("force_use_libuv_off")
    try:
        s._force_libuv_check.setChecked(True)
        assert s._app.get("force_use_libuv_off") is True
        s._force_libuv_check.setChecked(False)
        assert s._app.get("force_use_libuv_off") is False
    finally:
        s._app.set("force_use_libuv_off", prev)


def test_libuv_status_label_reflects_the_registry_probe(monkeypatch):
    monkeypatch.setattr("ui.setup_tab.windows_user_env_libuv", lambda: "1")
    monkeypatch.setattr("ui.setup_tab.windows_system_env_libuv", lambda: None)
    s = SetupTab()
    s._refresh_libuv_status()
    assert s._clear_libuv_btn.isEnabled()
    monkeypatch.setattr("ui.setup_tab.windows_user_env_libuv", lambda: None)
    s._refresh_libuv_status()
    assert not s._clear_libuv_btn.isEnabled()


def test_libuv_status_label_explains_a_system_scope_only_value(monkeypatch):
    monkeypatch.setattr("ui.setup_tab.windows_user_env_libuv", lambda: None)
    monkeypatch.setattr("ui.setup_tab.windows_system_env_libuv", lambda: "1")
    s = SetupTab()
    s._refresh_libuv_status()
    assert not s._clear_libuv_btn.isEnabled()
    assert "System scope" in s._libuv_status_label.text()


def test_clear_libuv_confirms_then_clears(monkeypatch):
    monkeypatch.setattr("ui.setup_tab.windows_user_env_libuv", lambda: "1")
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    calls = []
    monkeypatch.setattr("ui.setup_tab.clear_windows_user_env_libuv", lambda: calls.append(True))
    s = SetupTab()
    s._clear_libuv_from_windows()
    assert calls == [True]


def test_clear_libuv_declines_without_confirmation(monkeypatch):
    monkeypatch.setattr("ui.setup_tab.windows_user_env_libuv", lambda: "1")
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    calls = []
    monkeypatch.setattr("ui.setup_tab.clear_windows_user_env_libuv", lambda: calls.append(True))
    s = SetupTab()
    s._clear_libuv_from_windows()
    assert calls == []


def test_clear_libuv_is_a_no_op_when_nothing_is_set(monkeypatch):
    monkeypatch.setattr("ui.setup_tab.windows_user_env_libuv", lambda: None)
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                         staticmethod(lambda *a, **k: asked.append(True) or QMessageBox.Yes))
    calls = []
    monkeypatch.setattr("ui.setup_tab.clear_windows_user_env_libuv", lambda: calls.append(True))
    s = SetupTab()
    s._clear_libuv_from_windows()
    assert asked == []   # no dialog when there's nothing to clear
    assert calls == []   # and the real clear function is never reached either
