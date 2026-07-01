import os

_ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
).replace("\\", "/")


def asset_url(name: str) -> str:
    return f"{_ASSETS}/{name}"


DARK_STYLESHEET = f"""
QMainWindow {{ background-color: #0a0a0b; color: #c6c6ce; }}

QWidget {{
    background-color: #0a0a0b;
    color: #c6c6ce;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}}

/* window/content backdrop carries the ember vignette (scoped, not universal) */
#app_bg {{
    background-color: #0a0a0b;
    border-image: url({_ASSETS}/bg_embers.png) 0 0 0 0 stretch stretch;
}}

/* ---- Sidebar / Nav ---- */
#sidebar {{ background-color: #08080a; border-right: 1px solid #2a2a1e; }}
#nav_button {{
    background-color: transparent; color: #8a8a93; border: none;
    border-left: 3px solid transparent; padding: 14px 18px; text-align: left;
    font-size: 14px; font-weight: 500;
}}
#nav_button:hover {{ background-color: #14130f; color: #e8e0c8; border-left: 3px solid #8a5a12; }}
#nav_button[selected="true"] {{
    background-color: #161208; color: #f4d160; border-left: 3px solid #d4af37; font-weight: 700;
}}
#app_title {{ color: #d4af37; font-size: 19px; font-weight: 800; letter-spacing: 2px;
    padding: 16px 16px 2px 16px; background-color: #08080a; }}
#app_subtitle {{ color: #8a8a93; font-size: 10px; padding: 0 16px 16px 16px; background-color: #08080a; }}

/* ---- App header bar ---- */
#app_header {{ background-color: #0d0c0a; border-bottom: 1px solid #2a2a1e; }}
#header_title {{ color: #d4af37; font-size: 16px; font-weight: 700; letter-spacing: 1px; padding: 10px 16px; }}
#header_status {{ color: #8a8a93; font-size: 12px; padding: 10px 16px; }}

/* ---- General Buttons ---- */
QPushButton {{
    background-color: #161512; color: #e8e0c8; border: 1px solid #3a3a1f;
    border-radius: 6px; padding: 8px 16px; font-size: 13px; font-weight: 500; min-height: 22px;
}}
QPushButton:hover {{ background-color: #201d14; border: 1px solid #d4af37; color: #f4d160; }}
QPushButton:pressed {{ background-color: #2a2415; border: 1px solid #f4d160; }}
QPushButton:disabled {{ background-color: #121110; color: #4a4a44; border: 1px solid #222; }}

#btn_primary, #quick_action {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f6c453, stop:0.5 #d4972b, stop:1 #8a5a12);
    color: #1a1206; border: 1px solid #f4d160; padding: 10px 20px; font-size: 14px; font-weight: 700;
}}
#btn_primary:hover, #quick_action:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffd874, stop:1 #a8701a); border: 1px solid #fff0c0;
}}
#btn_primary:pressed, #quick_action:pressed {{ background: #8a5a12; }}
#btn_primary:disabled {{ background: #2a2415; color: #6a5a30; border: 1px solid #3a3a1f; }}

/* Compact gold button for the rarely-used individual caption steps (4-up row). */
#btn_step_compact {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f6c453, stop:0.5 #d4972b, stop:1 #8a5a12);
    color: #1a1206; border: 1px solid #f4d160; padding: 4px 6px; font-size: 11px; font-weight: 700;
    border-radius: 5px;
}}
#btn_step_compact:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffd874, stop:1 #a8701a); border: 1px solid #fff0c0;
}}
#btn_step_compact:pressed {{ background: #8a5a12; }}
#btn_step_compact:disabled {{ background: #2a2415; color: #6a5a30; border: 1px solid #3a3a1f; }}

#btn_start {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f6c453, stop:1 #b8860b);
    color: #1a1206; border: 1px solid #f4d160; padding: 12px 24px; font-size: 15px;
    font-weight: 800; border-radius: 8px;
}}
#btn_start:hover {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffe085, stop:1 #d4972b); }}
#btn_start:pressed {{ background: #8a5a12; }}
#btn_start:disabled {{ background: #211d12; color: #6a5a30; border: 1px solid #3a3a1f; }}

#btn_stop {{
    background-color: #2a1410; color: #ff9a5c; border: 1px solid #ff7a18;
    padding: 12px 24px; font-size: 15px; font-weight: 800; border-radius: 8px;
}}
#btn_stop:hover {{ background-color: #3a1c12; border: 1px solid #ffa050; color: #ffc090; }}
#btn_stop:pressed {{ background-color: #1f0f0a; }}
#btn_stop:disabled {{ background-color: #1a120f; color: #6a4030; border: 1px solid #3a2418; }}

#btn_ponify {{ background-color: #201d14; color: #f4d160; border: 1px solid #8a5a12; padding: 8px 16px;
    font-size: 13px; font-weight: 600; }}
#btn_ponify:hover {{ background-color: #2a2415; border: 1px solid #d4af37; color: #ffe085; }}
#btn_ponify:pressed {{ background-color: #161208; }}

#btn_danger {{ background-color: #2a1410; color: #ff9a5c; border: 1px solid #8a2a18; }}
#btn_danger:hover {{ background-color: #3a1c12; color: #ffc090; }}

/* ---- Spin / Combo / Inputs ---- */
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
    background-color: #100f0d; color: #e8e0c8; border: 1px solid #3a3a3f;
    border-radius: 5px; padding: 4px 8px; font-size: 13px; min-height: 26px;
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
    padding: 6px; font-size: 12px; }}
QTextEdit:focus {{ border: 1px solid #d4af37; }}
QTextEdit[readOnly="true"] {{ background-color: #0c0b0a; color: #9a9aa2; }}

/* ---- Progress Bar ---- */
QProgressBar {{ background-color: #100f0d; border: 1px solid #3a3a1f; border-radius: 5px;
    text-align: center; color: #f4d160; font-weight: 700; height: 22px; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
    stop:0 #8a5a12, stop:0.5 #d4972b, stop:1 #f6c453); border-radius: 4px; }}

/* ---- Labels ---- */
QLabel {{ color: #c6c6ce; background-color: transparent; }}
#label_section {{ color: #d4af37; font-size: 14px; font-weight: 800; letter-spacing: 1px; padding: 4px 0; }}
#label_field {{ color: #8a8a93; font-size: 12px; font-weight: 500; }}
#label_status_ok {{ color: #d4af37; font-size: 18px; }}
#label_status_err {{ color: #d9534f; font-size: 18px; }}
#label_status_unknown {{ color: #6a6a72; font-size: 18px; }}
#label_image_count {{ color: #f4d160; font-size: 13px; font-weight: 700; }}
#label_step_calc {{ color: #f4d160; font-size: 13px; font-weight: 600; padding: 6px;
    background-color: #161208; border: 1px solid #3a3a1f; border-radius: 5px; }}
#label_config_summary {{ color: #9a9aa2; font-size: 12px; font-family: "Consolas", monospace; padding: 8px;
    background-color: #0c0b0a; border: 1px solid #2a2a1e; border-radius: 5px; }}

/* ---- Dashboard ---- */
#hero {{ background-color: #0a0a0b; border-image: url({_ASSETS}/hero_forge.png) 0 0 0 0 stretch stretch;
    border-radius: 10px; }}
#hero_title {{ color: #f4d160; font-size: 26px; font-weight: 800; letter-spacing: 1px;
    background: transparent; padding: 18px; }}
#card {{ background-color: #141312; border: 1px solid #2a2a1e; border-radius: 10px; }}
#ready_row_ok {{ color: #d4af37; font-weight: 600; }}
#ready_row_idle {{ color: #6a6a72; }}
#ready_row_err {{ color: #d9534f; font-weight: 600; }}

/* ---- Slider (forge gold) ---- */
QSlider::groove:horizontal {{ height: 4px; background: #2a2a1e; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: #8a5a12; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #d4af37; width: 14px; margin: -6px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: #f4d160; }}

/* ---- Captioning step cards ---- */
#step_card {{ background-color: #100f0d; border: 1px solid #2a2a1e; border-radius: 8px; }}
#step_badge {{ background-color: #d4af37; color: #1a1206; font-size: 14px; font-weight: 800; border-radius: 14px; }}
#step_title {{ color: #e8e0c8; font-size: 13px; font-weight: 700; }}
#step_desc {{ color: #8a8a93; font-size: 11px; }}
#step_status {{ color: #8a8a93; font-size: 11px; font-weight: 600; }}

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
#image_filename {{ color: #8a8a93; font-size: 10px; padding: 2px 0; }}

/* ---- GroupBox (brushed metal) ---- */
QGroupBox {{ background-color: #141312;
    border-image: url({_ASSETS}/panel_metal.png) 0 0 0 0 repeat repeat;
    border: 1px solid #2a2a1e; border-radius: 8px; margin-top: 12px; padding-top: 8px;
    font-weight: 700; color: #d4af37; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding: 0 8px;
    color: #d4af37; letter-spacing: 1px; }}

/* ---- Splitter / Status / Frame / Tooltip / Dialog ---- */
QSplitter::handle {{ background-color: #2a2a1e; width: 2px; }}
QStatusBar {{ background-color: #08080a; color: #8a8a93; border-top: 1px solid #2a2a1e; font-size: 12px; }}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{ color: #2a2a1e; }}
QToolTip {{ background-color: #141312; color: #e8e0c8; border: 1px solid #d4af37; padding: 4px 8px; border-radius: 4px; }}
QDialog {{ background-color: #0a0a0b; color: #c6c6ce; }}
QMessageBox {{ background-color: #0a0a0b; }}
QMessageBox QLabel {{ color: #c6c6ce; }}
"""
