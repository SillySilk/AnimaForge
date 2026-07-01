from pathlib import Path

from PySide6.QtCore import Qt, QSettings, QTimer, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core import characters as characters_mod
from core.caption_progress import parse_progress
from core.dataset_manager import scan_folder, save_caption, apply_prefix, combine_all
from core.tagger import TaggerProcess
from core.joycaption import JoyCaptionProcess
from core.llm_refine import LLMRefineProcess
from core.settings import SETTINGS_ORG, SETTINGS_APP
from ui.image_editor import ImageEditorDialog

# (display label, repo_id, use_onnx)
TAGGER_MODELS = [
    ("WD SwinV2 Tagger v3 (recommended)", "SmilingWolf/wd-swinv2-tagger-v3",    True),
    ("WD ViT Tagger v3",                  "SmilingWolf/wd-vit-tagger-v3",        True),
    ("WD SwinV2 Tagger v2 (Keras)",       "SmilingWolf/wd-v1-4-swinv2-tagger-v2", False),
    ("WD ViT Tagger v2 (Keras)",          "SmilingWolf/wd-v1-4-vit-tagger-v2",    False),
]

THUMB_SIZE = 220
GRID_COLS = 4
WORKFLOW_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

PROCESS_STEPS = ["tag", "describe", "refine", "combine"]
STEP_NAMES = {"tag": "Tag", "describe": "Describe", "refine": "Refine", "combine": "Combine"}


def phase_text(step_key: str, chain=None) -> str:
    """Status-line text for a step during a Process run, e.g. 'Step 2/4 · Describe…'.

    Numbers against the actual chain so a Refine-less run reads 'Step 3/3 · Combine…'.
    Falls back to the full PROCESS_STEPS when no chain is supplied.
    """
    chain = chain if chain else PROCESS_STEPS
    idx = chain.index(step_key)
    return f"Step {idx + 1}/{len(chain)} · {STEP_NAMES[step_key]}…"


def read_tagger_defaults():
    """Saved Auto-Tag settings (model index, threshold, overwrite) used by a Process run."""
    s = QSettings(SETTINGS_ORG, SETTINGS_APP)
    return (
        s.value("tagger_model_index", 0, type=int),
        s.value("tagger_threshold", 0.35, type=float),
        s.value("tagger_overwrite", False, type=bool),
    )


class CaptionEdit(QTextEdit):
    """QTextEdit with a debounced save signal."""

    caption_changed = Signal(str, str)  # (txt_path, new_text)

    def __init__(self, txt_path: str, parent=None):
        super().__init__(parent)
        self._txt_path = txt_path
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._do_save)
        self.textChanged.connect(self._on_text_changed)

    def _on_text_changed(self):
        self._timer.start()

    def _do_save(self):
        text = self.toPlainText()
        self.caption_changed.emit(self._txt_path, text)


class ImageCard(QWidget):
    """A single image card: thumbnail + filename + caption editor."""

    image_deleted = Signal(str)  # emits image_path when deleted
    cast_clicked = Signal(str)   # emits image_path when the Cast button is clicked
    opened = Signal(str)         # emits image_path when the thumbnail is clicked

    _STATUS_COLOR = {"done": "#8fa86b", "partial": "#ff7a18", "bare": "#6a6a72"}
    _STATUS_TIP = {"done": "Captioned", "partial": "Partial — needs Combine", "bare": "No caption yet"}

    def __init__(self, image_path: str, txt_path: str, caption: str, cast_count: int = 0,
                 status: str = "bare", parent=None):
        super().__init__(parent)
        self.setObjectName("image_card")
        self._image_path = image_path
        self._txt_path = txt_path
        self._status = status
        self._setup_ui(caption, cast_count)

    def mousePressEvent(self, event):
        # Clicks on the thumbnail / filename / empty card area open the modal editor;
        # clicks on the caption box and buttons are consumed by those child widgets.
        if event.button() == Qt.LeftButton:
            self.opened.emit(self._image_path)
        super().mousePressEvent(event)

    def _setup_ui(self, caption: str, cast_count: int = 0):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Thumbnail with a status dot (top-left) + cast pill (bottom-left) overlaid.
        self._thumb_label = QLabel()
        self._thumb_label.setObjectName("image_thumb")
        self._thumb_label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self._thumb_label.setAlignment(Qt.AlignCenter)
        self._thumb_label.setScaledContents(False)
        self._thumb_label.setCursor(Qt.PointingHandCursor)
        self._thumb_label.setToolTip("Click to open the caption editor")
        self._load_thumbnail()

        self._status_dot = QLabel(self._thumb_label)
        self._status_dot.setFixedSize(14, 14)
        self._status_dot.move(7, 7)
        self._apply_status_dot()

        self._cast_btn = QPushButton(self._cast_label(cast_count), self._thumb_label)
        self._cast_btn.setObjectName("btn_cast_pill")
        self._cast_btn.setToolTip("Assign which characters appear in this image")
        self._cast_btn.setCursor(Qt.PointingHandCursor)
        self._cast_btn.adjustSize()
        self._cast_btn.move(7, THUMB_SIZE - self._cast_btn.height() - 7)
        self._cast_btn.clicked.connect(lambda: self.cast_clicked.emit(self._image_path))
        layout.addWidget(self._thumb_label, alignment=Qt.AlignHCenter)

        # Filename + always-visible trash (delete image + caption).
        meta = QHBoxLayout()
        meta.setContentsMargins(0, 0, 0, 0)
        meta.setSpacing(4)
        filename = Path(self._image_path).name
        name_label = QLabel(filename)
        name_label.setObjectName("image_filename")
        name_label.setToolTip(filename)
        fm = name_label.fontMetrics()
        name_label.setText(fm.elidedText(filename, Qt.ElideMiddle, THUMB_SIZE - 30))
        meta.addWidget(name_label, 1)
        delete_btn = QPushButton("🗑")
        delete_btn.setObjectName("btn_delete_image")
        delete_btn.setFixedSize(24, 24)
        delete_btn.setToolTip("Delete this image and caption")
        delete_btn.setCursor(Qt.PointingHandCursor)
        delete_btn.clicked.connect(self._on_delete_clicked)
        delete_btn.setStyleSheet(
            "QPushButton { background-color: transparent; border: none; color: #a05048; font-size: 13px; }"
            "QPushButton:hover { color: #e05050; }"
        )
        meta.addWidget(delete_btn, 0)
        layout.addLayout(meta)

        # Read-only 2-line caption preview — editing happens in the editor modal (click).
        self._caption_preview = QLabel()
        self._caption_preview.setObjectName("image_caption_preview")
        self._caption_preview.setWordWrap(True)
        self._caption_preview.setFixedWidth(THUMB_SIZE)
        self._caption_preview.setFixedHeight(34)
        self._caption_preview.setCursor(Qt.PointingHandCursor)
        self._set_preview_text(caption)
        layout.addWidget(self._caption_preview)

        self.setFixedWidth(THUMB_SIZE + 20)

    def _apply_status_dot(self):
        color = self._STATUS_COLOR.get(self._status, "#6a6a72")
        self._status_dot.setStyleSheet(
            f"color:{color}; background:#0c0b0aee; border-radius:7px; font-size:12px;")
        self._status_dot.setText("●")
        self._status_dot.setAlignment(Qt.AlignCenter)
        self._status_dot.setToolTip(self._STATUS_TIP.get(self._status, ""))

    def matches(self, query: str, mode: str) -> bool:
        """Filter predicate for the Dataset search + segmented (all/captioned/needs) control."""
        if mode == "captioned" and self._status != "done":
            return False
        if mode == "needs" and self._status == "done":
            return False
        if query:
            hay = (self.filename + " " + (self._caption_text or "")).lower()
            if query.lower() not in hay:
                return False
        return True

    def _set_preview_text(self, caption: str):
        self._caption_text = caption or ""
        text = (caption or "").strip()
        if text:
            self._caption_preview.setText(text)
            self._caption_preview.setProperty("empty", "false")
        else:
            self._caption_preview.setText("No caption yet")
            self._caption_preview.setProperty("empty", "true")
        self._caption_preview.style().unpolish(self._caption_preview)
        self._caption_preview.style().polish(self._caption_preview)

    def set_status(self, status: str):
        self._status = status
        self._apply_status_dot()

    @staticmethod
    def _cast_label(n: int) -> str:
        return f"👤 Cast ({n})" if n else "👤 Cast"

    def set_cast_count(self, n: int):
        self._cast_btn.setText(self._cast_label(n))

    def _load_thumbnail(self):
        try:
            from PIL import Image
            import io

            img = Image.open(self._image_path)
            img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)

            # Pad to exact THUMB_SIZE x THUMB_SIZE with dark background
            padded = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (14, 14, 30))
            offset_x = (THUMB_SIZE - img.width) // 2
            offset_y = (THUMB_SIZE - img.height) // 2
            if img.mode == "RGBA":
                padded.paste(img, (offset_x, offset_y), img)
            else:
                padded.paste(img.convert("RGB"), (offset_x, offset_y))

            buf = io.BytesIO()
            padded.save(buf, format="PNG")
            buf.seek(0)

            qimg = QImage.fromData(buf.read())
            pixmap = QPixmap.fromImage(qimg)
            self._thumb_label.setPixmap(pixmap)
        except Exception:
            self._thumb_label.setText("No preview")
            self._thumb_label.setStyleSheet(
                "background-color: #0c0b0a; color: #4a4a44; font-size: 11px;"
            )

    def _on_delete_clicked(self):
        """Emit signal to request deletion of this image."""
        self.image_deleted.emit(self._image_path)

    def refresh_caption(self, caption: str):
        """Update the read-only preview after an edit in the modal (or the refresh timer)."""
        self._set_preview_text(caption)
        if (caption or "").strip():
            self.set_status("done")

    def set_processing(self, on: bool):
        """Amber border while this image is the one being captioned."""
        self.setProperty("processing", "true" if on else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    @property
    def filename(self) -> str:
        return Path(self._image_path).name

    @property
    def txt_path(self) -> str:
        return self._txt_path


class DatasetTab(QWidget):
    dataset_loaded = Signal(str, int)  # (folder_path, image_count)
    characters_changed = Signal()      # per-image cast/chat edited here -> Characters tab resync
    open_characters_requested = Signal()  # "name characters" prompt -> jump to Characters tab
    caption_finished = Signal()        # caption step status changed -> refresh progress rail
    auto_caption_finished = Signal(bool)  # headless Process chain ended (Home pipeline)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._folder_path = ""
        self._sdscripts_path = ""
        self._image_data = []   # list of dicts from scan_folder
        self._cards = []        # list of ImageCard widgets
        self._cards_by_name = {}    # image basename -> ImageCard (for the processing highlight)
        self._processing_card = None
        self._tagger = TaggerProcess(self)
        self._tagger.log_line.connect(self._on_tagger_log)
        self._tagger.finished.connect(self._on_tagger_finished)
        self._joycaption = JoyCaptionProcess(self)
        self._joycaption.log_line.connect(self._on_tagger_log)
        self._joycaption.finished.connect(self._on_joycaption_finished)
        self._llm = LLMRefineProcess(self)
        self._llm.log_line.connect(self._on_tagger_log)
        self._llm.finished.connect(self._on_llm_finished)
        self._lms_url = "http://localhost:1234/v1"
        self._lms_model = ""
        # Live caption feedback: track the running process + a refresh timer.
        self._active_caption_proc = None
        self._caption_stopped = False
        self._caption_timer = QTimer(self)
        self._caption_timer.setInterval(2000)
        self._caption_timer.timeout.connect(self._rebuild_txt_from_sidecars)
        self._characters = characters_mod.DatasetCharacters()
        self._chain = []
        self._chain_plan = []     # full planned step list for the active run (for phase numbering)
        self._chain_active = False
        self._auto_mode = False   # True while the Home pipeline runs the chain headless
        self._build_ui()
        self._load_llm_prefs()
        self._refresh_refine_reflection()

    def set_sdscripts_path(self, path: str):
        self._sdscripts_path = path

    def set_lmstudio_config(self, url: str, model: str):
        self._lms_url = url or "http://localhost:1234/v1"
        self._lms_model = model

    def _load_llm_prefs(self):
        """Seed sensible QSettings defaults the Refine/Combine dialogs read from."""
        s = QSettings(SETTINGS_ORG, SETTINGS_APP)
        if s.value("lora_type", None) is None:
            s.setValue("lora_type", "General")
        if s.value("lmstudio_max_tokens", None) is None:
            s.setValue("lmstudio_max_tokens", 1200)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_folder_path(self) -> str:
        return self._folder_path

    def get_image_count(self) -> int:
        return len(self._image_data)

    def get_trigger_word(self) -> str:
        return self._trigger_edit.text().strip()

    def get_prefix(self) -> str:
        return self._prefix_edit.text().strip()

    def set_prefix(self, prefix: str):
        """Quality prefix is edited on Home; keep this tab's (hidden) copy in sync."""
        self._prefix_edit.setText(prefix or "")

    def caption_counts(self) -> tuple:
        """(captioned, total) — training-ready .txt count and image count, for Home's status."""
        c = self._step_status_counts()
        return c["txt"], c["total"]

    def caption_controls(self) -> QWidget:
        """The caption control panel (Process + individual steps + stop + live progress).

        Built by this tab (so all engine wiring stays intact) but mounted onto the Home
        command center by MainWindow — Home is the single place to drive captioning.
        """
        return self._caption_side_panel

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _refine_in_process(self) -> bool:
        """Whether the LM Studio Refine step is part of automated Process / auto-pipeline runs."""
        s = QSettings(SETTINGS_ORG, SETTINGS_APP)
        return s.value("lmstudio_refine_in_process", False, type=bool)

    def _build_process_chain(self) -> list:
        """Steps the ▶ Process / Home auto-pipeline runs, in order. Refine is included only
        when the user has opted in via Setup; otherwise it stays manual-only."""
        steps = ["tag", "describe"]
        if self._refine_in_process():
            steps.append("refine")
        steps.append("combine")
        return steps

    def _chain_arrow_text(self, chain=None) -> str:
        chain = chain if chain else self._build_process_chain()
        return " → ".join(STEP_NAMES[k] for k in chain)

    def _refresh_refine_reflection(self):
        """Reflect the Setup toggle on the Dataset tab: the Refine pill label and the
        Process button tooltip. The ✨ Refine button itself stays enabled for manual use."""
        if not hasattr(self, "_step3_status"):
            return
        if self._refine_in_process():
            self._step3_status.setText("(in Process)")
        else:
            self._step3_status.setText("(manual only — enable in Setup)")
        self._process_btn.setToolTip(
            f"Run all steps in order ({self._chain_arrow_text()}) using your saved settings"
        )

    def _refresh_step_status(self):
        c = self._step_status_counts()
        total = c["total"]

        def fmt(n):
            return f"✓ {n}/{total}" if (total and n) else "— not run"
        self._step1_status.setText(fmt(c["tags"]))
        self._step2_status.setText(fmt(c["nl"]))
        self._step4_status.setText(fmt(c["txt"]))
        # step 3 (refine) has no distinct artifact -> label reflects the Setup toggle
        self._refresh_refine_reflection()
        if hasattr(self, "_status_summary"):
            self._update_status_summary()
        self.caption_finished.emit()  # keep the global progress rail in sync

    def _maybe_prompt_name_characters(self):
        s = QSettings(SETTINGS_ORG, SETTINGS_APP)
        enabled = s.value("prompt_name_characters", True, type=bool)
        has_roster = bool(characters_mod.load(self._folder_path).roster) if self._folder_path else False
        if not self._should_prompt_naming(has_roster, enabled):
            return
        box = QMessageBox(self)
        box.setWindowTitle("Name your characters?")
        box.setText(
            "Your filenames drive the captions. Name files as  NAME_SERIAL_CATEGORY  "
            "(e.g. Homer-Marge_004_Character; category = Character / Object / Style)."
            "\n\nOpen the validator to check and auto-format names."
        )
        open_btn = box.addButton("Validate names", QMessageBox.AcceptRole)
        box.addButton("Skip", QMessageBox.RejectRole)
        never_btn = box.addButton("Don't ask again", QMessageBox.DestructiveRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is open_btn:
            self._open_name_validator()
        elif clicked is never_btn:
            s.setValue("prompt_name_characters", False)

    def _open_name_validator(self):
        if not self._folder_path:
            QMessageBox.warning(self, "No Dataset", "Please load a dataset folder first.")
            return
        from ui.name_validate_view import NameValidateView
        dlg = NameValidateView(self._folder_path, self)
        dlg.characters_changed.connect(self._on_names_validated)
        dlg.exec()

    def _on_names_validated(self):
        self._load_characters()
        self._refresh_cast_badges()
        self.rebuild_captions_after_naming()
        self.characters_changed.emit()

    def rebuild_captions_after_naming(self):
        """Re-merge the .txt captions so freshly validated character names land in the prompt
        immediately (name leads the caption). Also used by the Characters tab's Fix-names path.
        No-op without a loaded folder."""
        if self._folder_path:
            self._rebuild_txt_from_sidecars()
            self._refresh_step_status()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 8)
        main_layout.setSpacing(10)

        # ---- Toolbar: set name · status summary · Clear Captions ----
        # (Folder loading is a Home control; this is a preview tab.)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(12)
        self._set_name_label = QLabel("No set loaded")
        self._set_name_label.setObjectName("af_screen_eyebrow")
        toolbar.addWidget(self._set_name_label)
        self._status_summary = QLabel("")
        self._status_summary.setObjectName("label_field")
        toolbar.addWidget(self._status_summary)
        # kept for load_folder_path/_load_images plumbing (not shown in this layout)
        self._path_label = QLabel("No folder loaded")
        self._count_label = QLabel("Image count: 0")
        toolbar.addStretch()
        clear_all_btn = QPushButton("🧹 Clear Captions")
        clear_all_btn.setObjectName("btn_danger")
        clear_all_btn.setToolTip("Delete all .txt, .tags, and .nl caption files in this folder (images untouched)")
        clear_all_btn.clicked.connect(self._clear_all_captions)
        toolbar.addWidget(clear_all_btn)
        main_layout.addLayout(toolbar)

        # ---- Filter row: search · All/Captioned/Needs Work · Validate Names ----
        filter_row = QHBoxLayout()
        filter_row.setSpacing(10)
        self._search_edit = QLineEdit()
        self._search_edit.setObjectName("af_search")
        self._search_edit.setPlaceholderText("Search captions & filenames…")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self._search_edit, 1)

        self._filter_mode = "all"
        self._seg_buttons = {}
        seg = QHBoxLayout()
        seg.setSpacing(0)
        for mode, text in [("all", "All"), ("captioned", "Captioned"), ("needs", "Needs Work")]:
            b = QPushButton(text)
            b.setObjectName("af_segment")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setChecked(mode == "all")
            b.clicked.connect(lambda _c=False, m=mode: self._set_filter_mode(m))
            self._seg_buttons[mode] = b
            seg.addWidget(b)
        filter_row.addLayout(seg)

        self._validate_names_btn = QPushButton("✓ Validate Names")
        self._validate_names_btn.setObjectName("btn_primary")
        self._validate_names_btn.setToolTip(
            "Check filenames against NAME_SERIAL_CATEGORY and auto-format them to convention")
        self._validate_names_btn.clicked.connect(self._open_name_validator)
        filter_row.addWidget(self._validate_names_btn)
        main_layout.addLayout(filter_row)

        # ---- Identity: trigger word + quality prefix (baked in at Combine) ----
        # Trigger word + quality prefix are edited on the Home command center (single source
        # of truth). These widgets stay here — the combine/rebuild engine reads them — but are
        # hidden on this preview tab and kept in sync from Home via set_trigger_word/set_prefix.
        prefix_row = QHBoxLayout()
        prefix_row.setSpacing(10)
        trig_label = QLabel("Trigger Word:")
        trig_label.setObjectName("label_field")
        prefix_row.addWidget(trig_label)
        self._trigger_edit = QLineEdit()
        self._trigger_edit.setPlaceholderText("e.g. mycharacter (optional)")
        self._trigger_edit.setFixedWidth(200)
        prefix_row.addWidget(self._trigger_edit)
        prefix_label = QLabel("Quality Prefix:")
        prefix_label.setObjectName("label_field")
        prefix_row.addWidget(prefix_label)
        self._prefix_edit = QLineEdit()
        self._prefix_edit.setPlaceholderText("optional, e.g. masterpiece, best quality")
        self._prefix_edit.setFixedWidth(220)
        prefix_row.addWidget(self._prefix_edit)
        prefix_row.addStretch()
        self._identity_row_widget = QWidget()
        self._identity_row_widget.setLayout(prefix_row)
        self._identity_row_widget.setVisible(False)
        main_layout.addWidget(self._identity_row_widget)

        # ---- Gallery (left) | control panel (right), draggable ----
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._gallery_widget = QWidget()
        self._gallery_widget.setObjectName("gallery_widget")
        self._gallery_layout = QGridLayout(self._gallery_widget)
        self._gallery_layout.setContentsMargins(8, 8, 8, 8)
        self._gallery_layout.setSpacing(12)
        self._gallery_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._scroll.setWidget(self._gallery_widget)

        # The caption control panel is built here but relocated onto the Home command
        # center (single source of truth). The Dataset tab keeps only the gallery (preview)
        # and Validate names. MainWindow calls caption_controls() to mount it on Home.
        self._caption_side_panel = self._build_side_panel()

        self._hsplit = QSplitter(Qt.Horizontal)
        self._hsplit.addWidget(self._scroll)
        self._hsplit.setStretchFactor(0, 1)
        self._hsplit.setCollapsible(0, False)

        # ---- Content (above) | captioning log (below), draggable ----
        self._tagger_log = QTextEdit()
        self._tagger_log.setReadOnly(True)
        self._tagger_log.setPlaceholderText("Captioning output will appear here…")
        self._tagger_log.setObjectName("log_output")
        self._tagger_log.setVisible(False)

        self._vsplit = QSplitter(Qt.Vertical)
        self._vsplit.addWidget(self._hsplit)
        self._vsplit.addWidget(self._tagger_log)
        self._vsplit.setStretchFactor(0, 1)
        self._vsplit.setStretchFactor(1, 0)
        self._vsplit.setCollapsible(0, False)
        self._vsplit.setCollapsible(1, True)
        main_layout.addWidget(self._vsplit, 1)

        self._panel_collapsed = False
        self._restore_splits()
        self._hsplit.splitterMoved.connect(lambda *a: self._save_splits())
        self._vsplit.splitterMoved.connect(lambda *a: self._save_splits())

    def _build_side_panel(self):
        panel = QWidget()
        panel.setObjectName("caption_panel")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # Everything that hides when the panel is collapsed lives in _panel_body.
        self._panel_body = QWidget()
        self._panel_body.setMinimumWidth(190)
        v = QVBoxLayout(self._panel_body)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Captioning")
        title.setObjectName("step_title")
        header.addWidget(title)
        header.addStretch()
        v.addLayout(header)

        self._process_btn = QPushButton("📝 Run captioning")
        self._process_btn.setObjectName("btn_primary")
        self._process_btn.setToolTip("Caption all images: run the enabled steps in order using your saved settings")
        self._process_btn.clicked.connect(self._process_clicked)
        v.addWidget(self._process_btn)

        self._phase_label = QLabel("Idle")
        self._phase_label.setObjectName("step_status")
        v.addWidget(self._phase_label)

        self._steps_toggle = QPushButton("Individual steps ▾")
        self._steps_toggle.setObjectName("btn_ponify")
        self._steps_toggle.setCheckable(True)
        self._steps_toggle.toggled.connect(self._on_steps_toggled)
        v.addWidget(self._steps_toggle)

        # The individual steps are rarely used (Run captioning does all of them), so they're
        # compact: one row of four small buttons with a parallel row of tiny status labels.
        self._steps_box = QWidget()
        sb = QVBoxLayout(self._steps_box)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(2)

        self._autotag_btn = QPushButton("🏷 Tag")
        self._autotag_btn.setToolTip("Auto-generate booru tags (.tags)")
        self._autotag_btn.clicked.connect(self._open_tagger_dialog)

        self._describe_btn = QPushButton("📝 Describe")
        self._describe_btn.setToolTip("Natural-language captions with JoyCaption (.nl)")
        self._describe_btn.clicked.connect(self._describe_joycaption)

        self._llm_btn = QPushButton("✨ Refine")
        self._llm_btn.setToolTip("Fuse + verify into Anima format with LM Studio")
        self._llm_btn.clicked.connect(self._open_refine_dialog)

        self._combine_btn = QPushButton("🧩 Combine")
        self._combine_btn.setToolTip("Merge .nl + .tags (with trigger/prefix) into training .txt")
        self._combine_btn.clicked.connect(self._open_combine_dialog)

        self._step1_status = QLabel("— not run")
        self._step2_status = QLabel("— not run")
        self._step3_status = QLabel("(optional)")
        self._step4_status = QLabel("— not run")

        steps_btn_row = QHBoxLayout()
        steps_btn_row.setSpacing(4)
        steps_status_row = QHBoxLayout()
        steps_status_row.setSpacing(4)
        for btn, status in (
            (self._autotag_btn, self._step1_status),
            (self._describe_btn, self._step2_status),
            (self._llm_btn, self._step3_status),
            (self._combine_btn, self._step4_status),
        ):
            btn.setObjectName("btn_step_compact")
            btn.setMinimumHeight(28)
            steps_btn_row.addWidget(btn, 1)
            status.setObjectName("step_status")
            status.setAlignment(Qt.AlignCenter)
            status.setStyleSheet("font-size: 9px;")
            steps_status_row.addWidget(status, 1)
        sb.addLayout(steps_btn_row)
        sb.addLayout(steps_status_row)

        self._steps_box.setVisible(False)
        v.addWidget(self._steps_box)

        self._stop_caption_btn = QPushButton("■ Stop")
        self._stop_caption_btn.setObjectName("btn_stop")
        self._stop_caption_btn.setToolTip("Stop the running process (completed captions are kept)")
        self._stop_caption_btn.setEnabled(False)
        self._stop_caption_btn.clicked.connect(self._stop_captioning)
        v.addWidget(self._stop_caption_btn)

        self._live_progress = QWidget()
        lp = QVBoxLayout(self._live_progress)
        lp.setContentsMargins(0, 6, 0, 0)
        lp.setSpacing(4)
        self._caption_bar = QProgressBar()
        self._caption_bar.setTextVisible(True)
        self._caption_bar.setRange(0, 1)
        self._caption_bar.setValue(0)
        self._caption_file = QLabel("")
        self._caption_file.setObjectName("step_status")
        self._caption_file.setWordWrap(True)
        lp.addWidget(self._caption_bar)
        lp.addWidget(self._caption_file)
        self._live_progress.setVisible(False)
        v.addWidget(self._live_progress)

        v.addStretch()
        outer.addWidget(self._panel_body)

        return panel

    def _on_steps_toggled(self, checked: bool):
        self._steps_box.setVisible(checked)
        self._steps_toggle.setText("Individual steps ▴" if checked else "Individual steps ▾")

    def _show_caption_help(self):
        QMessageBox.information(
            self, "How captioning works",
            "Run the steps in order. Tag adds booru tags, Describe writes a plain-English "
            "draft, and Combine merges your trigger word + draft + tags into the .txt files "
            "the trainer reads.\n\n"
            "Refine is optional: it uses your local AI (LM Studio) to fuse + verify the draft "
            "into Anima's format. It's off by default for ▶ Process runs — turn it on in the "
            "Setup tab's LM Studio section, or click ✨ Refine to run it manually any time.\n\n"
            "▶ Process runs the enabled steps in order with your saved settings. Open "
            "'Individual steps' to run just one. Name recurring characters on the Characters "
            "tab so the AI uses real names instead of guessing."
        )

    def _save_splits(self):
        s = QSettings(SETTINGS_ORG, SETTINGS_APP)
        s.setValue("dataset_hsplit", self._hsplit.saveState())
        s.setValue("dataset_vsplit", self._vsplit.saveState())

    def _restore_splits(self):
        s = QSettings(SETTINGS_ORG, SETTINGS_APP)
        h = s.value("dataset_hsplit")
        v = s.value("dataset_vsplit")
        if h is not None:
            self._hsplit.restoreState(h)
        else:
            self._hsplit.setSizes([820, 240])
        if v is not None:
            self._vsplit.restoreState(v)
        else:
            self._vsplit.setSizes([900, 0])

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _load_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder", self._folder_path or ""
        )
        if not folder:
            return
        self.load_folder_path(folder)

    def load_folder_path(self, folder: str):
        """Load a dataset folder programmatically (no dialog)."""
        if not folder:
            return
        self._folder_path = folder
        fm = self.fontMetrics()
        elided = fm.elidedText(folder, Qt.ElideLeft, 500)
        self._path_label.setText(elided)
        self._path_label.setToolTip(folder)

        self._load_images()
        self._maybe_prompt_name_characters()

    def set_trigger_word(self, trigger: str):
        self._trigger_edit.setText(trigger or "")

    def _load_images(self):
        # Clear existing cards
        self._clear_gallery()

        self._image_data = scan_folder(self._folder_path)
        self._load_characters()
        count = len(self._image_data)
        self._count_label.setText(f"Image count: {count}")

        for idx, item in enumerate(self._image_data):
            name = Path(item["image_path"]).name
            cast_count = len(characters_mod.explicit_tokens_for_image(self._characters, name))
            card = ImageCard(
                item["image_path"],
                item["txt_path"],
                item["caption"],
                cast_count,
                status=self._image_status(item),
                parent=self._gallery_widget,
            )
            card.image_deleted.connect(self._on_image_delete_requested)
            card.cast_clicked.connect(self._open_cast_dialog)
            card.opened.connect(self._open_image_editor)
            row = idx // GRID_COLS
            col = idx % GRID_COLS
            self._gallery_layout.addWidget(card, row, col)
            self._cards.append(card)
            self._cards_by_name[name] = card

        set_name = Path(self._folder_path).name if self._folder_path else ""
        self._set_name_label.setText(
            f"Set · {set_name} · {count} images" if set_name else "No set loaded")
        self._update_status_summary()
        self._apply_filter()
        self.dataset_loaded.emit(self._folder_path, count)
        self._refresh_step_status()

    def _set_filter_mode(self, mode: str):
        self._filter_mode = mode
        for m, b in self._seg_buttons.items():
            b.setChecked(m == mode)
        self._apply_filter()

    def _apply_filter(self):
        """Re-flow the gallery to only the cards matching the search + segmented filter."""
        query = self._search_edit.text().strip()
        for c in self._cards:
            self._gallery_layout.removeWidget(c)
        idx = 0
        for c in self._cards:
            if c.matches(query, self._filter_mode):
                self._gallery_layout.addWidget(c, idx // GRID_COLS, idx % GRID_COLS)
                c.setVisible(True)
                idx += 1
            else:
                c.setVisible(False)

    def _update_status_summary(self):
        counts = {"done": 0, "partial": 0, "bare": 0}
        for c in self._cards:
            counts[c._status] = counts.get(c._status, 0) + 1
        self._status_summary.setText(
            f"{counts['done']} Captioned · {counts['partial']} Partial · {counts['bare']} Bare")

    @staticmethod
    def _image_status(item: dict) -> str:
        """done = has training .txt · partial = has .tags/.nl only · bare = nothing."""
        if (item.get("caption") or "").strip():
            return "done"
        stem = Path(item["image_path"])
        if stem.with_suffix(".tags").exists() or stem.with_suffix(".nl").exists():
            return "partial"
        return "bare"

    def _clear_gallery(self):
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        self._cards_by_name.clear()
        self._processing_card = None
        self._image_data.clear()

    def _dataset_ready(self) -> bool:
        if not self._folder_path:
            QMessageBox.warning(self, "No Dataset", "Please load a dataset folder first.")
            return False
        if len(self._image_data) == 0:
            QMessageBox.warning(self, "No Images", "No images found in the loaded folder.")
            return False
        return True

    def _clear_all_captions(self):
        if not self._dataset_ready():
            return
        msg = QMessageBox(self)
        msg.setWindowTitle("Clear All Captions")
        msg.setIcon(QMessageBox.Warning)
        msg.setText(
            f"Delete ALL caption files (.txt, .tags, .nl) for {len(self._image_data)} "
            f"images in:\n\n{self._folder_path}\n\n"
            "Your images are not touched. This cannot be undone. Continue?"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        if msg.exec() != QMessageBox.Yes:
            return

        deleted = 0
        for item in self._image_data:
            for ext in (".txt", ".tags", ".nl"):
                p = Path(item["image_path"]).with_suffix(ext)
                if p.is_file():
                    try:
                        p.unlink()
                        deleted += 1
                    except OSError:
                        pass
        self._load_images()  # recreates empty .txt files and refreshes the gallery
        QMessageBox.information(self, "Cleared", f"Deleted {deleted} caption file(s).")

    def _apply_prefix(self):
        if not self._dataset_ready():
            return

        trigger = self._trigger_edit.text().strip()
        prefix = self._prefix_edit.text().strip()
        if not trigger and not prefix:
            QMessageBox.information(
                self, "Nothing to Apply",
                "Enter a trigger word and/or quality prefix first."
            )
            return

        bits = ", ".join(b for b in [trigger, prefix] if b)
        msg = QMessageBox(self)
        msg.setWindowTitle("Confirm Apply Prefix")
        msg.setIcon(QMessageBox.Question)
        msg.setText(
            f'This will prepend "{bits}" to all {len(self._image_data)} '
            f"caption files in:\n\n{self._folder_path}\n\n"
            "Files that already start with it will be skipped.\n\nContinue?"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        if msg.exec() != QMessageBox.Yes:
            return

        modified, skipped, errors = apply_prefix(self._folder_path, prefix, trigger)
        self._refresh_all_captions()

        result_parts = []
        if modified:
            result_parts.append(f"{modified} file(s) updated")
        if skipped:
            result_parts.append(f"{skipped} already had prefix (skipped)")
        if errors:
            result_parts.append(f"{errors} error(s)")
        summary = ", ".join(result_parts) if result_parts else "No changes made."
        QMessageBox.information(self, "Apply Prefix Complete", summary)

    def _rebuild_txt_from_sidecars(self) -> tuple:
        """Merge .nl + .tags (+ prefix) into the training .txt files and refresh the gallery.

        This is what makes the caption boxes reflect what training will actually read.
        Returns (written, errors).
        """
        order = "tags_first" if QSettings(SETTINGS_ORG, SETTINGS_APP).value("combine_order", 0, type=int) == 1 else "nl_first"
        trigger = self._trigger_edit.text().strip()
        prefix = self._prefix_edit.text().strip()
        full_prefix = ", ".join(b for b in [trigger, prefix] if b)
        written, errors = combine_all(self._folder_path, prefix=full_prefix, order=order)
        self._refresh_all_captions()
        return written, errors

    def _begin_caption(self, proc):
        """Enter captioning mode: track the process, enable Stop, start the live-refresh timer."""
        self._active_caption_proc = proc
        self._caption_stopped = False
        self._stop_caption_btn.setEnabled(True)
        self._caption_timer.start()
        self._caption_bar.setRange(0, 1)
        self._caption_bar.setValue(0)
        self._caption_file.setText("")
        self._live_progress.setVisible(True)
        if not self._tagger_log.isVisible():
            self._tagger_log.setVisible(True)

    def _end_caption(self) -> int:
        """Leave captioning mode: stop timer/Stop button, do a final rebuild. Returns written count."""
        self._caption_timer.stop()
        self._active_caption_proc = None
        self._stop_caption_btn.setEnabled(False)
        self._live_progress.setVisible(False)
        self._clear_processing_frame()
        written, _ = self._rebuild_txt_from_sidecars()
        return written

    def _set_processing_frame(self, name: str):
        """Amber-highlight the card currently being captioned (passive — no scroll)."""
        if self._processing_card is not None:
            self._processing_card.set_processing(False)
        card = self._cards_by_name.get(name)
        if card is not None:
            card.set_processing(True)
        self._processing_card = card

    def _clear_processing_frame(self):
        if self._processing_card is not None:
            self._processing_card.set_processing(False)
            self._processing_card = None

    def _stop_captioning(self):
        if self._active_caption_proc and self._active_caption_proc.is_running():
            self._caption_stopped = True
            self._chain = []
            self._on_tagger_log("[Stop] Stopping captioning — finishing the current image…")
            self._active_caption_proc.stop()

    def _combine_captions(self):
        if not self._dataset_ready():
            return
        written, errors = self._rebuild_txt_from_sidecars()
        self._refresh_step_status()
        summary = f"{written} caption file(s) rebuilt from .nl + .tags sidecars."
        if errors:
            summary += f" {errors} error(s)."
        QMessageBox.information(self, "Combine Complete", summary)

    def _process_clicked(self):
        if not self._dataset_ready():
            return
        if not self._sdscripts_path:
            QMessageBox.warning(self, "No sd-scripts", "Set the sd-scripts path in the Setup tab first.")
            return
        if self._chain_active or self._llm.is_running() or self._joycaption.is_running() or self._tagger.is_running():
            QMessageBox.information(self, "Busy", "A captioning process is already running.")
            return
        chain = self._build_process_chain()
        box = QMessageBox(self)
        box.setWindowTitle("Run all steps")
        box.setIcon(QMessageBox.Question)
        box.setText(
            f"Run all {len(chain)} steps on {len(self._image_data)} images?\n\n"
            f"{self._chain_arrow_text(chain)}\n(using your saved settings)"
        )
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.Yes)
        if box.exec() != QMessageBox.Yes:
            return
        self._chain = list(chain)
        self._chain_plan = list(chain)
        self._chain_active = True
        self._process_btn.setEnabled(False)
        self._chain_start_next()

    def start_auto_caption(self) -> bool:
        """Headless Process run for the Home pipeline — no confirm dialog, no completion
        popup. Emits auto_caption_finished(success) when the chain ends. Returns False
        if it could not start (caller should surface its own error)."""
        if not self._folder_path or len(self._image_data) == 0 or not self._sdscripts_path:
            return False
        if (self._chain_active or self._llm.is_running()
                or self._joycaption.is_running() or self._tagger.is_running()):
            return False
        self._auto_mode = True
        self._caption_stopped = False
        chain = self._build_process_chain()
        self._chain = list(chain)
        self._chain_plan = list(chain)
        self._chain_active = True
        self._process_btn.setEnabled(False)
        self._chain_start_next()
        return True

    def _chain_start_next(self):
        if not self._chain:
            self._chain_active = False
            self._chain_finish_ok()
            return
        key = self._chain[0]
        self._phase_label.setText(phase_text(key, self._chain_plan))
        if key == "tag":
            self._start_tag_with_defaults()
        elif key == "describe":
            self._start_describe()
        elif key == "refine":
            self._start_refine()
        elif key == "combine":
            self._rebuild_txt_from_sidecars()
            self._refresh_step_status()
            self._chain.pop(0)
            self._chain_start_next()

    def _chain_step_done(self, key: str, success: bool, fail_reason: str):
        """Called from a *_finished handler while a Process chain is active."""
        if not success:
            self._chain_fail(key, fail_reason)
            return
        if self._chain and self._chain[0] == key:
            self._chain.pop(0)
        self._chain_start_next()

    def _chain_fail(self, key: str, reason: str):
        self._chain = []
        self._chain_active = False
        self._process_btn.setEnabled(True)
        self._phase_label.setText(f"Failed at {STEP_NAMES[key]}")
        if self._auto_mode:
            # Home pipeline: stay silent here; the orchestrator surfaces one error popup.
            self._auto_mode = False
            self.auto_caption_finished.emit(False)
            return
        QMessageBox.warning(
            self, "Process stopped",
            f"{STEP_NAMES[key]} failed: {reason}\n\nStopped. Completed steps are kept."
        )

    def _chain_finish_ok(self, *_):
        self._chain = []
        self._chain_active = False
        self._process_btn.setEnabled(True)
        self._phase_label.setText("Idle")
        if self._auto_mode:
            self._auto_mode = False
            self.auto_caption_finished.emit(True)
            return
        c = self._step_status_counts()
        QMessageBox.information(
            self, "Process complete",
            f"All steps complete — {c['txt']} caption file(s) built."
        )

    def _chain_cancelled(self):
        self._chain = []
        self._chain_active = False
        self._process_btn.setEnabled(True)
        self._phase_label.setText("Idle")
        if self._auto_mode:
            self._auto_mode = False
            self.auto_caption_finished.emit(False)

    def _describe_joycaption(self):
        if not self._dataset_ready():
            return
        if not self._sdscripts_path:
            QMessageBox.warning(self, "No sd-scripts", "Set the sd-scripts path in the Setup tab first.")
            return
        if self._joycaption.is_running() or self._tagger.is_running():
            QMessageBox.information(self, "Busy", "A captioning process is already running.")
            return
        msg = QMessageBox(self)
        msg.setWindowTitle("Describe with JoyCaption")
        msg.setIcon(QMessageBox.Question)
        msg.setText(
            f"Generate natural-language captions for {len(self._image_data)} images?\n\n"
            "Captions are written to .nl sidecar files (your tags and .txt are untouched).\n"
            "The JoyCaption model (~17GB) downloads on first use."
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        if msg.exec() != QMessageBox.Yes:
            return
        self._start_describe()

    def _start_describe(self):
        """Start JoyCaption (no confirm). Reused by Process."""
        self._tagger_log.setVisible(True)
        self._tagger_log.clear()
        self._tagger_log.append("⚠ Do not close the app while captioning/downloading is in progress.\n")
        self._describe_btn.setEnabled(False)
        self._describe_btn.setText("📝 Captioning…")
        self._joycaption.start(
            sdscripts_path=self._sdscripts_path,
            image_folder=self._folder_path,
        )
        self._begin_caption(self._joycaption)

    def _open_refine_dialog(self):
        if not self._dataset_ready():
            return
        if not self._sdscripts_path:
            QMessageBox.warning(self, "No sd-scripts", "Set the sd-scripts path in the Setup tab first.")
            return
        s = QSettings(SETTINGS_ORG, SETTINGS_APP)
        dlg = QDialog(self)
        dlg.setWindowTitle("Refine with AI")
        dlg.setMinimumWidth(440)
        lay = QVBoxLayout(dlg)

        row_a = QHBoxLayout()
        row_a.addWidget(QLabel("Type:"))
        type_combo = QComboBox()
        type_combo.addItems(["General", "Character", "Style", "Concept"])
        idx = type_combo.findText(s.value("lora_type", "General", type=str))
        type_combo.setCurrentIndex(idx if idx >= 0 else 0)
        type_combo.setToolTip("Steers what the model emphasizes vs omits for this LoRA type")
        row_a.addWidget(type_combo, 1)
        lay.addLayout(row_a)

        row_f = QHBoxLayout()
        row_f.addWidget(QLabel("Focus:"))
        focus_edit = QLineEdit(s.value("lmstudio_focus", "", type=str))
        focus_edit.setPlaceholderText("optional, e.g. emphasize the lighting")
        row_f.addWidget(focus_edit, 1)
        lay.addLayout(row_f)

        row_t = QHBoxLayout()
        maxtok_label = QLabel("Max tokens:")
        maxtok_label.setToolTip(
            "Token budget per image. Reasoning models spend most of this 'thinking' before "
            "writing the caption, so too low a value yields an empty/truncated result."
        )
        row_t.addWidget(maxtok_label)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(256)
        slider.setMaximum(4000)
        slider.setSingleStep(100)
        slider.setPageStep(200)
        slider.setValue(s.value("lmstudio_max_tokens", 1200, type=int))
        row_t.addWidget(slider)
        val = QLabel(str(slider.value()))
        val.setMinimumWidth(40)
        slider.valueChanged.connect(lambda v: val.setText(str(v)))
        row_t.addWidget(val)
        lay.addLayout(row_t)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Run Refine")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return
        s.setValue("lora_type", type_combo.currentText())
        s.setValue("lmstudio_focus", focus_edit.text().strip())
        s.setValue("lmstudio_max_tokens", slider.value())
        self._start_refine()

    def _start_refine(self):
        """Start the LLM refine pass using saved settings (no dialog). Reused by Process."""
        if self._llm.is_running() or self._joycaption.is_running() or self._tagger.is_running():
            QMessageBox.information(self, "Busy", "A captioning process is already running.")
            return
        s = QSettings(SETTINGS_ORG, SETTINGS_APP)
        lora_type = s.value("lora_type", "General", type=str).lower()
        if lora_type == "general":
            lora_type = ""
        self._tagger_log.setVisible(True)
        self._tagger_log.clear()
        self._tagger_log.append("⚠ Ensure the LM Studio server is running with a vision model loaded.\n")
        self._llm_btn.setEnabled(False)
        self._llm_btn.setText("✨ Refining…")
        self._llm.start(
            sdscripts_path=self._sdscripts_path,
            image_folder=self._folder_path,
            url=self._lms_url,
            model=self._lms_model,
            focus=s.value("lmstudio_focus", "", type=str),
            lora_type=lora_type,
            max_tokens=s.value("lmstudio_max_tokens", 1200, type=int),
            characters_file=characters_mod.path_for(self._folder_path),
        )
        self._begin_caption(self._llm)

    def _open_combine_dialog(self):
        if not self._dataset_ready():
            return
        s = QSettings(SETTINGS_ORG, SETTINGS_APP)
        dlg = QDialog(self)
        dlg.setWindowTitle("Build captions")
        dlg.setMinimumWidth(440)
        lay = QVBoxLayout(dlg)
        row = QHBoxLayout()
        row.addWidget(QLabel("Combine order:"))
        order_combo = QComboBox()
        order_combo.addItems(["NL then tags", "Tags then NL"])
        order_combo.setCurrentIndex(s.value("combine_order", 0, type=int))
        row.addWidget(order_combo, 1)
        lay.addLayout(row)
        buttons = QDialogButtonBox()
        buttons.addButton("Combine → .txt", QDialogButtonBox.AcceptRole)
        prefix_b = buttons.addButton("Apply Prefix", QDialogButtonBox.ActionRole)
        buttons.addButton(QDialogButtonBox.Cancel)
        prefix_b.clicked.connect(lambda: (s.setValue("combine_order", order_combo.currentIndex()), self._apply_prefix()))
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.Accepted:
            return
        s.setValue("combine_order", order_combo.currentIndex())
        self._combine_captions()

    def _on_llm_finished(self, success: bool):
        self._llm_btn.setEnabled(True)
        self._llm_btn.setText("✨ Refine")
        written = self._end_caption()
        self._refresh_step_status()
        if self._caption_stopped:
            self._chain_cancelled()
            QMessageBox.information(self, "Stopped", "Captioning stopped. Captions completed so far are kept.")
            return
        if self._chain_active:
            self._chain_step_done("refine", success, "LM Studio not reachable / refiner error.")
            return
        if success:
            QMessageBox.information(
                self, "LLM Pass Complete",
                f"Captions refined and merged into {written} caption file(s)."
            )
        else:
            QMessageBox.warning(
                self, "LLM Pass Failed",
                "The LLM refiner exited with an error (is LM Studio running with a vision model "
                "loaded?). Check the log above."
            )

    def _on_joycaption_finished(self, success: bool):
        self._describe_btn.setEnabled(True)
        self._describe_btn.setText("📝 Describe")
        written = self._end_caption()
        self._refresh_step_status()
        if self._caption_stopped:
            self._chain_cancelled()
            QMessageBox.information(self, "Stopped", "Captioning stopped. Captions completed so far are kept.")
            return
        if self._chain_active:
            self._chain_step_done("describe", success, "JoyCaption exited with an error.")
            return
        if success:
            QMessageBox.information(
                self, "Captioning Complete",
                f"Natural-language captions generated and merged into {written} caption file(s).\n"
                "They now appear in the caption boxes and will be used for training."
            )
        else:
            QMessageBox.warning(self, "Captioning Failed", "JoyCaption exited with an error. Check the log above.")

    def _refresh_all_captions(self):
        """Re-read caption files and update all card widgets."""
        from core.dataset_manager import load_caption
        for card in self._cards:
            caption = load_caption(card.txt_path)
            card.refresh_caption(caption)

    # ------------------------------------------------------------------
    # Characters & style anchor
    # ------------------------------------------------------------------

    def _load_characters(self):
        self._characters = (characters_mod.load(self._folder_path)
                            if self._folder_path else characters_mod.DatasetCharacters())

    def reload_characters(self):
        """Reload from disk after the Characters tab edited things, and refresh cast badges."""
        self._load_characters()
        self._refresh_cast_badges()

    def _refresh_cast_badges(self):
        for card in self._cards:
            nm = Path(card._image_path).name
            card.set_cast_count(
                len(characters_mod.explicit_tokens_for_image(self._characters, nm))
            )


    def _open_image_editor(self, image_path: str):
        if not self._cards:
            return
        items = [
            {"image_path": c._image_path, "txt_path": c._txt_path,
             "caption": c._caption_edit.toPlainText()}
            for c in self._cards
        ]
        index = next((i for i, it in enumerate(items) if it["image_path"] == image_path), 0)
        dlg = ImageEditorDialog(items, index, self._characters, self)
        dlg.caption_saved.connect(self._on_modal_caption_saved)
        dlg.cast_changed.connect(self._on_modal_cast_changed)
        dlg.exec()

    def _on_modal_caption_saved(self, txt_path: str, text: str):
        for card in self._cards:
            if card.txt_path == txt_path:
                card.refresh_caption(text)
                break

    def _on_modal_cast_changed(self, name: str):
        if self._folder_path:
            characters_mod.save(self._folder_path, self._characters)
        count = len(characters_mod.explicit_tokens_for_image(self._characters, name))
        for card in self._cards:
            if Path(card._image_path).name == name:
                card.set_cast_count(count)
                break
        self.characters_changed.emit()

    def _open_cast_dialog(self, image_path: str):
        name = Path(image_path).name
        entry = self._characters.assignments.get(name, {"present": [], "oneoffs": []})
        present = set(entry.get("present", []))
        oneoffs = list(entry.get("oneoffs", []))

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Characters in {name}")
        dlg.setMinimumWidth(420)
        v = QVBoxLayout(dlg)

        v.addWidget(QLabel("Tick the roster characters present in this image:"))
        checks = []
        if self._characters.roster:
            for c in self._characters.roster:
                cb = QCheckBox(f"{c.token}  —  {c.description}" if c.description else c.token)
                cb.setChecked(c.token in present)
                v.addWidget(cb)
                checks.append((c.token, cb))
        else:
            v.addWidget(QLabel("(No roster yet — add characters in the Characters group above.)"))

        v.addWidget(QLabel("One-off character for this image only (optional):"))
        oneoff_row = QHBoxLayout()
        oneoff_tok = QLineEdit(oneoffs[0]["token"] if oneoffs else "")
        oneoff_tok.setPlaceholderText("token")
        oneoff_tok.setFixedWidth(140)
        oneoff_desc = QLineEdit(oneoffs[0]["description"] if oneoffs else "")
        oneoff_desc.setPlaceholderText("recognition description")
        oneoff_row.addWidget(oneoff_tok)
        oneoff_row.addWidget(oneoff_desc, 1)
        v.addLayout(oneoff_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        v.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return

        new_present = [tok for tok, cb in checks if cb.isChecked()]
        new_oneoffs = []
        if oneoff_tok.text().strip():
            new_oneoffs.append({"token": oneoff_tok.text().strip(),
                                "description": oneoff_desc.text().strip()})
        self._characters.assignments[name] = {"present": new_present, "oneoffs": new_oneoffs}
        if self._folder_path:
            characters_mod.save(self._folder_path, self._characters)
        count = len(characters_mod.explicit_tokens_for_image(self._characters, name))
        for card in self._cards:
            if Path(card._image_path).name == name:
                card.set_cast_count(count)
                break
        self.characters_changed.emit()

    def _step_status_counts(self, folder: str = None) -> dict:
        """Count images (one per stem) that have a non-empty .tags/.nl/.txt sidecar."""
        folder = folder or self._folder_path
        counts = {"tags": 0, "nl": 0, "txt": 0, "total": 0}
        p = Path(folder) if folder else None
        if not p or not p.is_dir():
            return counts
        seen = set()
        for f in sorted(p.iterdir()):
            if f.suffix.lower() not in WORKFLOW_IMAGE_EXTS or f.stem in seen:
                continue
            seen.add(f.stem)
            counts["total"] += 1
            for ext, key in ((".tags", "tags"), (".nl", "nl"), (".txt", "txt")):
                side = f.with_suffix(ext)
                try:
                    if side.is_file() and side.stat().st_size > 0:
                        counts[key] += 1
                except OSError:
                    pass
        return counts

    @staticmethod
    def _should_prompt_naming(has_roster: bool, enabled: bool) -> bool:
        """Prompt to name characters only when prompting is enabled and no roster exists yet."""
        return bool(enabled) and not has_roster

    def _open_tagger_dialog(self):
        if not self._folder_path:
            QMessageBox.warning(self, "No Dataset", "Please load a dataset folder first.")
            return
        if not self._sdscripts_path:
            QMessageBox.warning(self, "No sd-scripts", "Set the sd-scripts path in the Setup tab first.")
            return
        if self._tagger.is_running():
            QMessageBox.information(self, "Tagger Running", "Auto-tagger is already running.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Auto-Tag with WD14 Tagger")
        dlg.setMinimumWidth(480)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)

        # Model selection
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        model_combo = QComboBox()
        for label, _, _onnx in TAGGER_MODELS:
            model_combo.addItem(label)
        _saved_idx, _saved_thr, _saved_ow = read_tagger_defaults()
        model_combo.setCurrentIndex(_saved_idx)
        model_row.addWidget(model_combo, 1)
        layout.addLayout(model_row)

        # Threshold
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Confidence threshold:"))
        thresh_spin = QDoubleSpinBox()
        thresh_spin.setRange(0.10, 0.90)
        thresh_spin.setSingleStep(0.05)
        thresh_spin.setValue(_saved_thr)
        thresh_spin.setDecimals(2)
        thresh_row.addWidget(thresh_spin)
        thresh_row.addWidget(QLabel("(lower = more tags, higher = fewer tags)"))
        thresh_row.addStretch()
        layout.addLayout(thresh_row)

        # Overwrite
        overwrite_chk = QCheckBox("Overwrite existing captions")
        overwrite_chk.setChecked(_saved_ow)
        layout.addWidget(overwrite_chk)

        info = QLabel(
            f"Will tag <b>{len(self._image_data)}</b> images in:<br>"
            f"<code>{self._folder_path}</code><br><br>"
            "The model will be downloaded from HuggingFace on first use (~400MB)."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Start Tagging")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return

        _s = QSettings(SETTINGS_ORG, SETTINGS_APP)
        _s.setValue("tagger_model_index", model_combo.currentIndex())
        _s.setValue("tagger_threshold", thresh_spin.value())
        _s.setValue("tagger_overwrite", overwrite_chk.isChecked())

        _, model_id, use_onnx = TAGGER_MODELS[model_combo.currentIndex()]
        threshold = thresh_spin.value()
        overwrite = overwrite_chk.isChecked()

        self._tagger_log.setVisible(True)
        self._tagger_log.clear()
        self._tagger_log.append("⚠ Do not close the app while tagging/downloading is in progress.\n")
        self._autotag_btn.setEnabled(False)
        self._autotag_btn.setText("🏷 Tagging…")

        self._tagger.start(
            sdscripts_path=self._sdscripts_path,
            image_folder=self._folder_path,
            model_id=model_id,
            threshold=threshold,
            overwrite=overwrite,
            use_onnx=use_onnx,
        )
        self._begin_caption(self._tagger)

    def _start_tag_with_defaults(self):
        """Start auto-tagging with saved settings (no dialog). Reused by Process."""
        model_index, threshold, overwrite = read_tagger_defaults()
        _, model_id, use_onnx = TAGGER_MODELS[model_index]
        self._tagger_log.setVisible(True)
        self._tagger_log.clear()
        self._tagger_log.append("⚠ Do not close the app while tagging/downloading is in progress.\n")
        self._autotag_btn.setEnabled(False)
        self._autotag_btn.setText("🏷 Tagging…")
        self._tagger.start(
            sdscripts_path=self._sdscripts_path,
            image_folder=self._folder_path,
            model_id=model_id,
            threshold=threshold,
            overwrite=overwrite,
            use_onnx=use_onnx,
        )
        self._begin_caption(self._tagger)

    def _on_tagger_log(self, line: str):
        self._tagger_log.append(line)
        self._tagger_log.verticalScrollBar().setValue(
            self._tagger_log.verticalScrollBar().maximum()
        )
        tick = parse_progress(line)
        if tick is not None:
            self._phase_label.setText(tick.phase)
            self._caption_bar.setRange(0, tick.total)
            self._caption_bar.setValue(tick.done)
            self._caption_file.setText(tick.filename)
            self._set_processing_frame(tick.filename)

    def _on_tagger_finished(self, success: bool):
        self._autotag_btn.setEnabled(True)
        self._autotag_btn.setText("🏷 Tag")
        written = self._end_caption()
        self._refresh_step_status()
        if self._caption_stopped:
            self._chain_cancelled()
            QMessageBox.information(self, "Stopped", "Captioning stopped. Captions completed so far are kept.")
            return
        if self._chain_active:
            self._chain_step_done("tag", success, "The tagger exited with an error.")
            return
        if success:
            QMessageBox.information(
                self, "Tagging Complete",
                f"Tagged successfully and merged into {written} caption file(s).\n"
                "Tags now appear in the caption boxes and will be used for training."
            )
            self._maybe_prompt_name_characters()
        else:
            QMessageBox.warning(self, "Tagging Failed", "The tagger exited with an error. Check the log above.")

    def _on_image_delete_requested(self, image_path: str):
        """Handle deletion of an image and its caption file."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Delete Image")
        msg.setIcon(QMessageBox.Question)
        filename = Path(image_path).name
        msg.setText(
            f"Delete '{filename}' and its caption file?\n\n"
            "This cannot be undone."
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)

        if msg.exec() != QMessageBox.Yes:
            return

        try:
            Path(image_path).unlink()
            # Also delete the caption + sidecar files if they exist
            for ext in (".txt", ".tags", ".nl"):
                sidecar = Path(image_path).with_suffix(ext)
                if sidecar.is_file():
                    sidecar.unlink()

            # Reload the gallery
            self._load_images()
            QMessageBox.information(self, "Deleted", f"'{filename}' deleted successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Delete Failed", f"Could not delete file:\n{e}")
