import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from utils import styles

_app = QApplication.instance() or QApplication([])


def test_palette_and_selectors_present():
    s = styles.DARK_STYLESHEET
    assert "#d4af37" in s          # gold
    assert "#0a0a0b" in s          # base black
    assert "#c6c6ce" in s          # silver text
    # old purple accent fully gone
    assert "#c084fc" not in s
    assert "#533483" not in s
    for sel in ("#btn_primary", "#btn_start", "#btn_stop", "#image_card",
                "#label_section", "#nav_button", "#app_title"):
        assert sel in s, sel


def test_stylesheet_applies_without_error():
    _app.setStyleSheet(styles.DARK_STYLESHEET)
    assert _app.styleSheet() == styles.DARK_STYLESHEET
