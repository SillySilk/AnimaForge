import os

_ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
).replace("\\", "/")


def asset_url(name: str) -> str:
    return f"{_ASSETS}/{name}"


# ----------------------------------------------------------------------------
# Font families
# ----------------------------------------------------------------------------
# The forge redesign uses four bundled faces (see utils/fonts.py). QSS has no
# variables, so we inject the resolved family stacks into the sheet at build
# time. Fallback stacks keep the app legible before fonts load / if a file is
# missing.
_FALLBACK_FAMILIES = {
    "display": '"Pirata One", "Times New Roman", "Segoe UI Symbol", serif',
    "marker": '"Permanent Marker", "Comic Sans MS", "Segoe UI Symbol", cursive',
    "type": '"Special Elite", "Courier New", "Segoe UI Symbol", monospace',
    "body": '"Crimson Pro", Georgia, "Segoe UI Symbol", serif',
}


def build_stylesheet(families: dict | None = None) -> str:
    """Assemble the global QSS, injecting the resolved font families.

    Pass the map returned by ``utils.fonts.load_app_fonts()`` once fonts are
    registered; omit it (fallbacks used) for import-time construction and tests.
    """
    fam = dict(_FALLBACK_FAMILIES)
    if families:
        fam.update(families)
    F_DISPLAY = fam["display"]  # Pirata One — blackletter headlines, big numbers
    F_MARKER = fam["marker"]    # Permanent Marker — scrawled taglines
    F_TYPE = fam["type"]        # Special Elite — eyebrows, labels, nav, buttons
    F_BODY = fam["body"]        # Crimson Pro — body copy, inputs, captions

    return f"""
QMainWindow {{ background-color: #0a0a0b; color: #c6c6ce; }}

QWidget {{
    background-color: #0a0a0b;
    color: #c6c6ce;
    font-family: {F_BODY};
    font-size: 14px;
}}

/* window/content backdrop carries the ember vignette (scoped, not universal) */
#app_bg {{
    background-color: #0a0a0b;
    border-image: url({_ASSETS}/bg_embers.png) 0 0 0 0 stretch stretch;
}}

/* ============================================================
   FORGE REDESIGN SHELL — collapsible sidebar
   ============================================================ */
#af_sidebar {{ background-color: #08080a; border-right: 1px solid #2a2a1e; }}
#af_wordmark {{ font-family: {F_DISPLAY}; color: #d4af37; font-size: 30px;
    letter-spacing: 1px; background-color: transparent; }}
#af_eyebrow {{ font-family: {F_TYPE}; color: #8a8a93; font-size: 10px;
    letter-spacing: 3px; background-color: transparent; }}
#af_rule {{ background-color: #8a5a12; max-height: 2px; min-height: 2px; border: none; }}
#af_collapse_btn {{ background-color: #141312; border: 1px solid #3a3a1f;
    border-radius: 5px; color: #f4d160; font-size: 15px; font-weight: 700; }}
#af_collapse_btn:hover {{ border: 1px solid #8a5a12; background-color: #201d14; }}

#af_nav {{
    background-color: transparent; color: #8a8a93; border: none;
    border-left: 3px solid transparent; padding: 0 14px; text-align: left;
    font-family: {F_TYPE}; font-size: 13px; letter-spacing: 1px; text-transform: uppercase;
}}
#af_nav:hover {{ background-color: rgba(212,151,43,0.06); color: #e8e0c8; }}
#af_nav[selected="true"] {{ background-color: #161208; color: #f4d160;
    border-left: 3px solid #d4af37; }}

#af_decor_quote {{ font-family: {F_MARKER}; color: #a89c7e; font-size: 13px;
    background-color: transparent; }}
#af_decor_meta {{ font-family: {F_TYPE}; color: #4a4a44; font-size: 10px;
    letter-spacing: 2px; background-color: transparent; }}
#af_ver {{ color: #4a4a44; font-family: {F_TYPE}; font-size: 10px;
    letter-spacing: 2px; background-color: transparent; }}

/* ---- legacy sidebar/nav (kept for un-migrated screens + tests) ---- */
#sidebar {{ background-color: #08080a; border-right: 1px solid #2a2a1e; }}
#nav_button {{
    background-color: transparent; color: #8a8a93; border: none;
    border-left: 3px solid transparent; padding: 14px 18px; text-align: left;
    font-family: {F_TYPE}; font-size: 13px; letter-spacing: 1px;
}}
#nav_button:hover {{ background-color: #14130f; color: #e8e0c8; border-left: 3px solid #8a5a12; }}
#nav_button[selected="true"] {{
    background-color: #161208; color: #f4d160; border-left: 3px solid #d4af37;
}}
#app_title {{ color: #d4af37; font-family: {F_DISPLAY}; font-size: 24px; letter-spacing: 1px;
    padding: 12px 16px 2px 16px; background-color: #08080a; }}
#app_subtitle {{ color: #8a8a93; font-family: {F_TYPE}; font-size: 10px; letter-spacing: 1px;
    padding: 0 16px 16px 16px; background-color: #08080a; }}

/* ---- App header bar ---- */
#app_header {{ background-color: #0d0c0a; border-bottom: 1px solid #2a2a1e; }}
#header_title {{ color: #d4af37; font-family: {F_DISPLAY}; font-size: 22px; letter-spacing: 1px;
    padding: 10px 16px; }}
#header_status {{ color: #8a8a93; font-family: {F_TYPE}; font-size: 12px; padding: 10px 16px; }}
#af_screen_title {{ color: #d4af37; font-family: {F_DISPLAY}; font-size: 26px;
    background-color: transparent; }}
#af_screen_eyebrow {{ color: #8a8a93; font-family: {F_TYPE}; font-size: 11px;
    letter-spacing: 2px; background-color: transparent; }}

/* ============================================================
   BUTTONS
   ============================================================ */
QPushButton {{
    background-color: #161512; color: #e8e0c8; border: 1px solid #3a3a1f;
    border-radius: 6px; padding: 8px 14px; font-family: {F_TYPE}; font-size: 11px;
    letter-spacing: 0; min-height: 22px;
}}
QPushButton:hover {{ background-color: #201d14; border: 1px solid #d4af37; color: #f4d160; }}
QPushButton:pressed {{ background-color: #2a2415; border: 1px solid #f4d160; }}
QPushButton:disabled {{ background-color: #121110; color: #4a4a44; border: 1px solid #222; }}

#btn_primary, #quick_action {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f6c453, stop:0.5 #d4972b, stop:1 #8a5a12);
    color: #1a1206; border: 1px solid #f4d160; padding: 10px 20px; font-size: 13px;
    letter-spacing: 1px;
}}
#btn_primary:hover, #quick_action:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffd874, stop:1 #a8701a); border: 1px solid #fff0c0;
}}
#btn_primary:pressed, #quick_action:pressed {{ background: #8a5a12; }}
#btn_primary:disabled {{ background: #2a2415; color: #6a5a30; border: 1px solid #3a3a1f; }}

/* Compact gold button for the rarely-used individual caption steps (4-up row). */
#btn_step_compact {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f6c453, stop:0.5 #d4972b, stop:1 #8a5a12);
    color: #1a1206; border: 1px solid #f4d160; padding: 4px 6px; font-size: 11px;
    border-radius: 5px;
}}
#btn_step_compact:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffd874, stop:1 #a8701a); border: 1px solid #fff0c0;
}}
#btn_step_compact:pressed {{ background: #8a5a12; }}
#btn_step_compact:disabled {{ background: #2a2415; color: #6a5a30; border: 1px solid #3a3a1f; }}

/* Dataset gallery card — cast pill (over the thumbnail) + 2-line caption preview. */
#btn_cast_pill {{
    background: rgba(12,11,10,0.82); color: #f4d160; border: 1px solid #3a3a1f;
    border-radius: 9px; padding: 2px 8px; font-size: 10px; font-weight: 700;
}}
#btn_cast_pill:hover {{ border: 1px solid #d4af37; color: #ffe085; }}
#image_caption_preview {{ color: #c6c6ce; font-size: 11px; }}
#image_caption_preview[empty="true"] {{ color: #6a6a72; font-style: italic; }}

/* Dataset filter row — search box + segmented All/Captioned/Needs Work. */
#af_search {{
    background: #100f0d; border: 1px solid #3a3a3f; border-radius: 6px;
    padding: 7px 10px; color: #c6c6ce;
}}
#af_search:focus {{ border: 1px solid #d4af37; }}
#af_segment {{
    background: #100f0d; color: #8a8a93; border: 1px solid #3a3a1f;
    padding: 7px 14px; font-size: 12px; font-weight: 600;
}}
#af_segment:hover {{ color: #e8e0c8; }}
#af_segment:checked {{ background: #161208; color: #f4d160; border: 1px solid #8a5a12; }}

#btn_start {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f6c453, stop:1 #b8860b);
    color: #1a1206; border: 1px solid #f4d160; padding: 12px 24px; font-size: 14px;
    letter-spacing: 1px; border-radius: 8px;
}}
#btn_start:hover {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffe085, stop:1 #d4972b); }}
#btn_start:pressed {{ background: #8a5a12; }}
#btn_start:disabled {{ background: #211d12; color: #6a5a30; border: 1px solid #3a3a1f; }}

#btn_stop {{
    background-color: #2a1410; color: #ff9a5c; border: 1px solid #ff7a18;
    padding: 12px 24px; font-size: 14px; letter-spacing: 1px; border-radius: 8px;
}}
#btn_stop:hover {{ background-color: #3a1c12; border: 1px solid #ffa050; color: #ffc090; }}
#btn_stop:pressed {{ background-color: #1f0f0a; }}
#btn_stop:disabled {{ background-color: #1a120f; color: #6a4030; border: 1px solid #3a2418; }}

#btn_ponify {{ background-color: #201d14; color: #f4d160; border: 1px solid #8a5a12; padding: 8px 16px;
    font-size: 12px; }}
#btn_ponify:hover {{ background-color: #2a2415; border: 1px solid #d4af37; color: #ffe085; }}
#btn_ponify:pressed {{ background-color: #161208; }}

#btn_danger {{ background-color: #2a1410; color: #ff9a5c; border: 1px solid #8a2a18; }}
#btn_danger:hover {{ background-color: #3a1c12; color: #ffc090; }}

/* ---- forge redesign buttons ---- */
#af_btn_ghost {{ background-color: transparent; border: 1px solid #3a3a1f; color: #f4d160;
    border-radius: 6px; padding: 0 14px; font-family: {F_TYPE}; font-size: 11px; letter-spacing: 1px; }}
#af_btn_ghost:hover {{ background-color: rgba(212,151,43,0.10); border-color: #8a5a12; color: #ffe085; }}
#af_btn_forge {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f6c453, stop:1 #b8860b);
    color: #1a1206; border: 1px solid #f4d160; border-radius: 10px; padding: 0 22px;
    font-family: {F_TYPE}; font-size: 14px; letter-spacing: 2px; }}
#af_btn_forge:hover {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffe085, stop:1 #d4972b); border: 1px solid #fff0c0; }}
#af_btn_forge:pressed {{ background: #8a5a12; }}
#af_pill_ok {{ background-color: rgba(143,168,107,0.10); border: 1px solid rgba(143,168,107,0.4);
    color: #8fa86b; border-radius: 18px; padding: 0 14px; font-family: {F_TYPE}; font-size: 11px; letter-spacing: 1px; }}
#af_pill_ok:hover {{ background-color: rgba(143,168,107,0.18); }}
#af_icon_btn {{ background-color: #141312; border: 1px solid #3a3a1f; border-radius: 6px; color: #8a8a93; }}
#af_icon_btn:hover {{ color: #f4d160; border-color: #8a5a12; }}

/* ============================================================
   INPUTS
   ============================================================ */
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
    background-color: #100f0d; color: #e8e0c8; border: 1px solid #3a3a3f;
    border-radius: 5px; padding: 4px 8px; font-family: {F_BODY}; font-size: 14px; min-height: 26px;
}}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
    border: 1px solid #d4af37; background-color: #15130d;
}}
QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {{ background-color: #0c0b0a; color: #4a4a44; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: #201d14; border: none; width: 18px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background-color: #2a2415; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-bottom: 5px solid #d4af37; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid #d4af37; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid #d4af37; }}
QComboBox QAbstractItemView {{ background-color: #15130d; color: #e8e0c8; border: 1px solid #3a3a1f;
    selection-background-color: #8a5a12; selection-color: #fff0c0; }}

QTextEdit {{ background-color: #100f0d; color: #c6c6ce; border: 1px solid #3a3a3f; border-radius: 5px;
    padding: 6px; font-family: {F_BODY}; font-size: 13px; }}
QTextEdit:focus {{ border: 1px solid #d4af37; }}
QTextEdit[readOnly="true"] {{ background-color: #0c0b0a; color: #9a9aa2; }}
/* monospace logs opt in with objectName log_view */
#log_view {{ font-family: {F_TYPE}; font-size: 12px; color: #9a9aa2; background-color: #100f0d; }}

/* ---- Progress Bar ---- */
QProgressBar {{ background-color: #100f0d; border: 1px solid #3a3a1f; border-radius: 5px;
    text-align: center; color: #f4d160; font-family: {F_TYPE}; height: 22px; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
    stop:0 #8a5a12, stop:0.5 #d4972b, stop:1 #f6c453); border-radius: 4px; }}

/* ---- Labels ---- */
QLabel {{ color: #c6c6ce; background-color: transparent; }}
#label_section {{ color: #d4af37; font-family: {F_DISPLAY}; font-size: 18px; letter-spacing: 1px; padding: 4px 0; }}
#label_field {{ color: #8a8a93; font-family: {F_TYPE}; font-size: 11px; letter-spacing: 1px; }}
#label_status_ok {{ color: #d4af37; font-size: 18px; }}
#label_status_err {{ color: #d9534f; font-size: 18px; }}
#label_status_unknown {{ color: #6a6a72; font-size: 18px; }}
#label_image_count {{ color: #f4d160; font-family: {F_TYPE}; font-size: 13px; }}
#label_step_calc {{ color: #f4d160; font-family: {F_TYPE}; font-size: 13px; padding: 6px;
    background-color: #161208; border: 1px solid #3a3a1f; border-radius: 5px; }}
#label_config_summary {{ color: #9a9aa2; font-size: 12px; font-family: {F_TYPE}; padding: 8px;
    background-color: #0c0b0a; border: 1px solid #2a2a1e; border-radius: 5px; }}

/* forge redesign text roles */
#af_eyebrow_flame {{ color: #ff9a5c; font-family: {F_TYPE}; font-size: 11px;
    letter-spacing: 2px; background-color: transparent; }}
#af_eyebrow_mute {{ color: #8a8a93; font-family: {F_TYPE}; font-size: 10px;
    letter-spacing: 2px; background-color: transparent; }}
#af_display_gold {{ color: #d4af37; font-family: {F_DISPLAY}; font-size: 32px; background-color: transparent; }}
#af_display_gold4 {{ color: #f4d160; font-family: {F_DISPLAY}; font-size: 23px; background-color: transparent; }}
#af_marker {{ color: #cfc4a6; font-family: {F_MARKER}; font-size: 15px; background-color: transparent; }}
#af_marker_gold {{ color: #d4972b; font-family: {F_MARKER}; font-size: 13px; background-color: transparent; }}
#af_stat_value {{ color: #f4d160; font-family: {F_DISPLAY}; font-size: 22px; background-color: transparent; }}
#af_num_gold {{ color: #f4d160; font-family: {F_DISPLAY}; font-size: 18px; background-color: transparent; }}

/* ---- Dashboard ---- */
#hero {{ background-color: #0a0a0b; border-image: url({_ASSETS}/hero_forge.png) 0 0 0 0 stretch stretch;
    border-radius: 10px; }}
#hero_title {{ color: #f4d160; font-family: {F_DISPLAY}; font-size: 26px; letter-spacing: 1px;
    background: transparent; padding: 18px; }}
#card {{ background-color: #141312; border: 1px solid #2a2a1e; border-radius: 10px; }}
#ready_row_ok {{ color: #d4af37; }}
#ready_row_idle {{ color: #6a6a72; }}
#ready_row_err {{ color: #d9534f; }}

/* forge redesign surfaces */
#af_card {{ background-color: #141312; border: 1px solid #2a2a1e; border-radius: 10px; }}
#af_card_hi {{ background-color: #141312; border: 1px solid #d4af37; border-radius: 10px; }}
#af_well {{ background-color: #100f0d; border: 1px solid #3a3a3f; border-radius: 5px; }}
#af_lever {{ border: 1px solid #8a5a12; border-radius: 10px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(212,151,43,0.10), stop:1 rgba(168,49,30,0.06)); }}
#af_pillar_accent {{ background-color: #8a5a12; max-height: 2px; min-height: 2px; }}
#af_chip {{ background-color: #100f0d; border: 1px solid #2a2a1e; border-radius: 5px;
    color: #e8e0c8; font-family: {F_TYPE}; font-size: 10px; letter-spacing: 1px; padding: 8px 4px; }}

/* ---- Slider (forge gold) ---- */
QSlider::groove:horizontal {{ height: 4px; background: #2a2a1e; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: #8a5a12; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #d4af37; width: 14px; margin: -6px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: #f4d160; }}

/* ---- Captioning step cards ---- */
#step_card {{ background-color: #100f0d; border: 1px solid #2a2a1e; border-radius: 8px; }}
#step_badge {{ background-color: #d4af37; color: #1a1206; font-size: 14px; border-radius: 14px; }}
#step_title {{ color: #e8e0c8; font-size: 13px; }}
#step_desc {{ color: #8a8a93; font-size: 11px; }}
#step_status {{ color: #8a8a93; font-size: 11px; }}

/* ---- Scroll Areas ---- */
QScrollArea {{ background-color: transparent; border: none; }}
QScrollBar:vertical {{ background-color: #100f0d; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background-color: #3a3a1f; border-radius: 5px; min-height: 20px; }}
QScrollBar::handle:vertical:hover {{ background-color: #8a5a12; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background-color: #100f0d; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background-color: #3a3a1f; border-radius: 5px; min-width: 20px; }}
QScrollBar::handle:horizontal:hover {{ background-color: #8a5a12; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ---- Image Card ---- */
#image_card {{ background-color: #141312; border: 1px solid #2a2a1e; border-radius: 8px; padding: 6px; }}
#image_card:hover {{ border: 1px solid #d4af37; background-color: #1a1810; }}
#image_card[processing="true"] {{ border: 2px solid #d4af37; background-color: #1a1810; }}
#image_thumb {{ background-color: #0c0b0a; border-radius: 4px; }}
#image_filename {{ color: #8a8a93; font-family: {F_TYPE}; font-size: 10px; padding: 2px 0; }}

/* ---- GroupBox (brushed metal) ---- */
QGroupBox {{ background-color: #141312;
    border-image: url({_ASSETS}/panel_metal.png) 0 0 0 0 repeat repeat;
    border: 1px solid #2a2a1e; border-radius: 8px; margin-top: 12px; padding-top: 8px;
    font-family: {F_TYPE}; color: #d4af37; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding: 0 8px;
    color: #d4af37; letter-spacing: 1px; }}

/* ---- Modal (forge redesign) ---- */
#af_modal_scrim {{ background-color: rgba(6,6,8,0.72); }}
#af_modal_card {{ background-color: #141312; border: 1px solid #8a5a12; border-radius: 10px; }}
#af_modal_rule {{ background-color: #d4972b; max-height: 2px; min-height: 2px; }}
#af_modal_title {{ color: #d4af37; font-family: {F_DISPLAY}; font-size: 30px; background-color: transparent; }}
#af_modal_close {{ background-color: transparent; border: none; color: #8a8a93; font-size: 18px; }}
#af_modal_close:hover {{ color: #f4d160; }}

/* ---- Splitter / Status / Frame / Tooltip / Dialog ---- */
QSplitter::handle {{ background-color: #2a2a1e; width: 2px; }}
QStatusBar {{ background-color: #08080a; color: #8a8a93; border-top: 1px solid #2a2a1e;
    font-family: {F_TYPE}; font-size: 12px; }}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{ color: #2a2a1e; }}
QToolTip {{ background-color: #141312; color: #e8e0c8; border: 1px solid #d4af37; padding: 4px 8px; border-radius: 4px; }}
QDialog {{ background-color: #0a0a0b; color: #c6c6ce; }}
QMessageBox {{ background-color: #0a0a0b; }}
QMessageBox QLabel {{ color: #c6c6ce; }}
"""


# Built at import with fallback families so existing imports/tests keep working;
# main.py rebuilds it with the registered fonts once the QApplication exists.
DARK_STYLESHEET = build_stylesheet()
