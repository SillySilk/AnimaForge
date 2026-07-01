import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from ui.home_tab import HomeTab

_app = QApplication.instance() or QApplication([])


def test_home_is_first_screen_and_six_nav():
    w = MainWindow()
    assert w._stack.count() == 6
    assert isinstance(w._stack.widget(0), HomeTab)
    assert len(w._nav_buttons) == 6


def test_settings_reached_via_gear_not_top_nav():
    w = MainWindow()
    # Exactly one nav button targets the Settings stack page (index 1): the bottom gear.
    targets = [idx for _, idx in w._nav_buttons]
    assert targets.count(1) == 1
    gear_btn, gear_idx = w._nav_buttons[-1]   # pinned last
    assert gear_idx == 1 and "Setup" in gear_btn.text()
    # No primary (non-gear) button points at Setup/Settings.
    assert all(idx != 1 for _, idx in w._nav_buttons[:-1])
    w._switch_tab(1)
    assert w._stack.currentIndex() == 1
    assert gear_btn._selected is True


def test_home_navigate_switches_tab():
    w = MainWindow()
    w._home_tab.navigate.emit(4)
    assert w._stack.currentIndex() == 4


def test_collect_home_context_keys():
    w = MainWindow()
    ctx = w._collect_home_context()
    for k in ("sdscripts", "dit", "qwen3", "vae", "output", "torch_ok",
              "dataset_folder", "image_count", "lms_url", "lms_ok"):
        assert k in ctx


def test_open_characters_request_switches_to_characters_tab():
    w = MainWindow()
    w._switch_tab(2)                       # start on Dataset
    w._dataset_tab.open_characters_requested.emit()
    assert w._stack.currentIndex() == 3    # jumped to Characters


def test_home_run_split_wired():
    # MainWindow constructs and the split handlers + batch entry exist.
    w = MainWindow()
    assert hasattr(w, "_on_home_caption") and hasattr(w, "_on_home_train")
    assert hasattr(w._train_tab, "add_current_to_batch")


def test_trigger_and_prefix_single_source_on_home():
    w = MainWindow()
    # The Dataset tab's trigger + quality-prefix row is hidden — Home is the single source.
    assert w._dataset_tab._identity_row_widget.isHidden() is True
    # Editing on Home drives the (hidden) Dataset widgets the combine engine reads.
    w._home_tab.trigger_changed.emit("mychar")
    w._home_tab.prefix_changed.emit("masterpiece")
    assert w._dataset_tab.get_trigger_word() == "mychar"
    assert w._dataset_tab.get_prefix() == "masterpiece"
