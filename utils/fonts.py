"""Bundle-and-load the four Thing-o-Matic / forge display faces.

The redesign leans on four hand-picked Google faces (Pirata One, Permanent
Marker, Special Elite, Crimson Pro). We ship the .ttf files in ``assets/fonts``
and register them at startup via ``QFontDatabase`` so the app looks right on any
machine with no system-install dependency — matching the "local-only desktop
app" goal.

Call :func:`load_app_fonts` once, after the ``QApplication`` exists but before
the stylesheet is applied. It returns the map of logical role -> resolved family
name so the QSS layer can reference exactly what Qt registered (font files do
not always advertise the family name you expect).
"""

from __future__ import annotations

import os
import re

from PySide6.QtGui import QFontDatabase

_FONT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts"
)

# role -> (filename, expected family, fallback stack used if the file is missing)
# "Segoe UI Symbol" is kept in every stack so Qt's per-glyph fallback can render
# the UI's chevrons/arrows/anvil (‹ › → ⚒ ✓ ♻ ⚒) that the display faces lack;
# "Malgun Gothic" follows so CJK text (Korean Windows) renders cleanly instead of
# falling into whatever Qt picks last.
_SYM = '"Segoe UI Symbol", "Malgun Gothic"'
_FONTS = {
    "display": ("PirataOne-Regular.ttf", "Pirata One", f'"Times New Roman", {_SYM}, serif'),
    "marker": ("PermanentMarker-Regular.ttf", "Permanent Marker", f'"Comic Sans MS", {_SYM}, cursive'),
    "type": ("SpecialElite-Regular.ttf", "Special Elite", f'"Courier New", {_SYM}, monospace'),
    "body": ("CrimsonPro-Variable.ttf", "Crimson Pro", f'Georgia, {_SYM}, serif'),
}

# Populated by load_app_fonts(); role -> a ready-to-use QSS font-family stack.
FAMILIES: dict[str, str] = {}


def load_app_fonts() -> dict[str, str]:
    """Register the bundled fonts and return role -> QSS family stack.

    Idempotent: safe to call more than once (Qt de-dupes by content).
    """
    FAMILIES.clear()
    for role, (filename, expected, fallback) in _FONTS.items():
        path = os.path.join(_FONT_DIR, filename)
        resolved = None
        if os.path.exists(path):
            font_id = QFontDatabase.addApplicationFont(path)
            if font_id != -1:
                fams = QFontDatabase.applicationFontFamilies(font_id)
                if fams:
                    resolved = fams[0]
        primary = resolved or expected
        FAMILIES[role] = f'"{primary}", {fallback}'
    return FAMILIES


# System-UI stack: what functional text always uses (see utils.styles._F_UI) and
# what "system" mode collapses the decorative roles to.
UI_STACK = '"Segoe UI", "Malgun Gothic", "Segoe UI Symbol", sans-serif'

# The role -> stack map currently IN EFFECT after mode resolution. Updated by
# apply_app_font(); empty means "forge" (fall through to FAMILIES).
ACTIVE_FAMILIES: dict[str, str] = {}


def resolve_families(mode: str | None = None, custom_family: str | None = None) -> dict[str, str]:
    """Role -> QSS stack honoring the user's font preference.

    Pass ``mode``/``custom_family`` explicitly (tests), or leave ``None`` to read
    the persisted setting. Unknown modes and custom-without-a-family resolve to
    forge (the bundled faces) so a corrupt setting can never blank the UI.
    """
    if mode is None:
        from core.settings import AppSettings
        s = AppSettings()
        mode = s.get("ui_font_mode")
        custom_family = s.get("ui_font_family")
    if mode == "system":
        return {role: UI_STACK for role in _FONTS}
    if mode == "custom" and (custom_family or "").strip():
        fam = custom_family.strip().replace('"', "")
        stack = f'"{fam}", "Segoe UI Symbol", "Malgun Gothic", sans-serif'
        return {role: stack for role in _FONTS}
    return dict(FAMILIES)


def apply_app_font(app=None) -> dict[str, str]:
    """Re-resolve the font preference and re-apply the app stylesheet, live."""
    from PySide6.QtWidgets import QApplication

    from utils.styles import build_stylesheet

    ACTIVE_FAMILIES.clear()
    ACTIVE_FAMILIES.update(resolve_families())
    target = app or QApplication.instance()
    if target is not None:
        target.setStyleSheet(build_stylesheet(ACTIVE_FAMILIES))
    return ACTIVE_FAMILIES


def family(role: str) -> str:
    """QSS font-family stack for a role ('display'/'marker'/'type'/'body')."""
    stack = ACTIVE_FAMILIES.get(role) or FAMILIES.get(role)
    return stack or "sans-serif"


def primary_family(role: str) -> str:
    """First family name (unquoted) of the active stack — for QFont() consumers."""
    stack = family(role)
    m = re.match(r'\s*"([^"]+)"', stack)
    return m.group(1) if m else stack.split(",")[0].strip()
