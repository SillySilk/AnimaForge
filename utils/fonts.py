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


def family(role: str) -> str:
    """QSS font-family stack for a role ('display'/'marker'/'type'/'body')."""
    return FAMILIES.get(role, "sans-serif")
