import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

BASE = {"epochs": 10, "learning_rate": 1.0, "optimizer.use_speed": True, "name": "x", "lst": [1]}


def _panel(overrides=None):
    from ui.advanced_settings import AdvancedSettingsPanel
    return AdvancedSettingsPanel(BASE, overrides or {})


def _row(p, path):
    for r in range(p.table.rowCount()):
        if p.table.item(r, 0).text() == path:
            return r
    raise AssertionError(path)


def test_lists_every_setting_with_its_preset_value():
    p = _panel()
    assert p.table.rowCount() == len(BASE)
    r = _row(p, "learning_rate")
    assert p.table.item(r, 1).text() == "1.0" and p.table.item(r, 2).text() == "1.0"
    assert p.overrides() == {} and p.changed_count() == 0


def test_only_the_value_column_is_editable():
    from PySide6.QtCore import Qt
    p = _panel()
    r = _row(p, "epochs")
    assert p.table.item(r, 1).flags() & Qt.ItemIsEditable
    assert not p.table.item(r, 0).flags() & Qt.ItemIsEditable
    assert not p.table.item(r, 2).flags() & Qt.ItemIsEditable


def test_editing_a_value_makes_an_override_and_marks_the_row():
    p = _panel()
    r = _row(p, "epochs")
    p.table.item(r, 1).setText("25")
    assert p.overrides() == {"epochs": 25} and p.changed_count() == 1
    assert p.table.item(r, 0).font().bold()
    p.table.item(r, 1).setText("10")
    assert p.overrides() == {} and not p.table.item(r, 0).font().bold()


def test_existing_overrides_start_marked():
    p = _panel({"optimizer.use_speed": False, "noise_offset": 0.05})
    assert p.overrides() == {"optimizer.use_speed": False, "noise_offset": 0.05}
    assert p.table.rowCount() == len(BASE) + 1
    assert p.table.item(_row(p, "optimizer.use_speed"), 0).font().bold()


def test_bad_value_for_a_typed_setting_is_refused():
    p = _panel()
    p.table.item(_row(p, "learning_rate"), 1).setText("12x")
    p.table.item(_row(p, "optimizer.use_speed"), 1).setText("maybe")
    with pytest.raises(ValueError) as e:
        p.overrides()
    assert "learning_rate" in str(e.value) and "optimizer.use_speed" in str(e.value)


def test_search_hides_non_matching_rows():
    p = _panel()
    p.search.setText("OPTIM")
    visible = [p.table.item(r, 0).text() for r in range(p.table.rowCount()) if not p.table.isRowHidden(r)]
    assert visible == ["optimizer.use_speed"]


def test_add_a_setting_the_preset_does_not_have():
    p = _panel()
    p.add_path.setText("training_arguments.noise_offset")
    p.add_value.setText("0.05")
    p.add_button.click()
    assert p.overrides() == {"training_arguments.noise_offset": 0.05}
    assert p.add_path.text() == "" and p.add_value.text() == ""


def test_adding_an_existing_path_does_not_duplicate_it():
    p = _panel()
    p.add_path.setText("epochs")
    p.add_value.setText("3")
    p.add_button.click()
    assert p.table.rowCount() == len(BASE)


def test_adding_needs_a_path():
    p = _panel()
    p.add_value.setText("3")
    p.add_button.click()
    assert p.table.rowCount() == len(BASE)


def test_reset_selected_restores_that_row():
    p = _panel({"epochs": 3, "extra": 1})
    p.table.selectRow(_row(p, "epochs"))
    p.reset_selected_button.click()
    assert p.overrides() == {"extra": 1}
    p.table.selectRow(_row(p, "extra"))
    p.reset_selected_button.click()
    assert p.overrides() == {} and p.table.rowCount() == len(BASE)


def test_reset_all_restores_the_preset():
    p = _panel({"epochs": 3, "extra": 1})
    p.reset_all_button.click()
    assert p.overrides() == {} and p.table.rowCount() == len(BASE)


def _locked_panel(overrides=None):
    from ui.advanced_settings import AdvancedSettingsPanel
    return AdvancedSettingsPanel(BASE, overrides or {}, locked=("name", "opt*"))


def test_locked_settings_are_shown_but_not_editable():
    from PySide6.QtCore import Qt
    p = _locked_panel()
    r = _row(p, "name")
    assert not p.table.item(r, 1).flags() & Qt.ItemIsEditable
    assert "run" in p.table.item(r, 1).toolTip().lower()


def test_adding_a_locked_or_broken_path_is_refused_with_a_reason():
    p = _locked_panel()
    for path in ("optimizer.extra", "epochs.x", "a..b"):
        p.add_path.setText(path)
        p.add_value.setText("1")
        p.add_button.click()
        assert p.table.rowCount() == len(BASE), path
        assert p.add_error.text()


def test_an_added_group_name_cannot_replace_a_group():
    p = _panel()
    p.add_path.setText("optimizer")
    p.add_value.setText("5")
    p.add_button.click()
    assert p.table.rowCount() == len(BASE)
