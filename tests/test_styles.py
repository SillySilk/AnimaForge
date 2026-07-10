import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from PySide6.QtWidgets import (
    QApplication,
    QPushButton,
    QStyle,
    QStyleOptionButton,
)
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


def _content_width(object_name: str, w: int, h: int) -> int:
    """Width the style leaves for the button's label, after border + padding.

    Measured through QStyle rather than by counting painted pixels: a pixel
    count depends on which fonts happen to be installed and on whichever
    stylesheet an earlier test left applied, so it flakes. The content rect is
    pure geometry.
    """
    _app.setStyleSheet(styles.build_stylesheet())
    b = QPushButton("X")
    b.setObjectName(object_name)
    b.setFixedSize(w, h)
    b.show()
    opt = QStyleOptionButton()
    opt.initFrom(b)
    opt.rect = b.rect()
    return b.style().subElementRect(QStyle.SE_PushButtonContents, opt, b).width()


@pytest.mark.parametrize(
    "object_name, w, h",
    [
        ("af_collapse_btn", 26, 26),   # ‹ › sidebar collapse / expand
        ("af_icon_btn", 26, 26),       # ↑ ↓ ✕ batch row: move up, move down, remove
        ("af_icon_btn", 28, 28),       # ✕ train recovery dismiss
        ("af_icon_btn", 34, 34),       # ⚙ home options gear
        ("af_icon_btn", 36, 36),       # i home info
        ("af_icon_btn", 48, 48),       # 📜 home config preview
        ("af_modal_close", 32, 32),    # ✕ modal close
    ],
)
def test_square_icon_buttons_leave_room_for_their_glyph(object_name, w, h):
    """A square icon button must not inherit QPushButton's 14px side padding.

    At 26px wide that padding leaves a *negative* content rect, so Qt clips the
    glyph away entirely and the button ships as an empty box — which is exactly
    what happened to the Batch row's up/down/remove buttons, the Train recovery
    dismiss, and the sidebar collapse chevron.
    """
    assert _content_width(object_name, w, h) >= 16
