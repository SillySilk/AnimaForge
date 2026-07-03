from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config_generator import generate_configs, get_config_summary
from core.step_calculator import (
    calculate_training_params,
    format_calculation_string,
    suggest_target_steps,
    is_capped,
    SOFT_CAP_STEPS,
)
from core.trainer import TrainingProcess
from core.train_metrics import parse_tqdm
from ui.collapsible import CollapsibleBox
from ui.gauge import DialRow
from ui.run_progress import RunProgress

import re as _re


def phase_for_line(line: str):
    """Map a known sd-scripts warmup marker to a human phase label, else None."""
    low = line.lower()
    if "caching latents" in low:
        return "Caching latents…"
    if "running training" in low or "学習開始" in line:
        return "Warming up…"
    return None


class LogDenoiser:
    """Collapse the DataLoader-worker startup flood in the training log (display-only)."""

    _BARE_CONT = _re.compile(r"^\s*current_epoch:\s*\d+,\s*epoch:\s*\d+\s*$")
    _EPOCH_INC = _re.compile(r"epoch is incremented\. current_epoch:\s*(\d+)")

    def __init__(self):
        self._last_epoch = None

    def filter(self, line: str):
        if self._BARE_CONT.match(line):
            return None
        m = self._EPOCH_INC.search(line)
        if m:
            epoch = m.group(1)
            if epoch == self._last_epoch:
                return None
            self._last_epoch = epoch
            return line
        return line


class _ClickableThumb(QLabel):
    """A thumbnail QLabel that emits its image path when clicked."""
    clicked = Signal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path = path
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        self.clicked.emit(self._path)


class TickBar(QWidget):
    """A thin strip under the progress bar marking the steps where a preview will render.

    Ticks already passed (a preview should exist) are drawn brighter than upcoming ones.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._positions = []   # step numbers where samples are generated
        self._total = 0
        self._current = 0
        self._spe = 0.0        # steps per epoch (to label the next tick with its epoch)
        self.setFixedHeight(30)
        self.setToolTip("Preview images render at these points during training. "
                        "The red mark is the next upcoming set.")

    def set_schedule(self, positions, total: int, steps_per_epoch: float = 0.0):
        self._positions = sorted(p for p in positions if 0 <= p <= max(total, 1))
        self._total = max(int(total), 1)
        self._spe = float(steps_per_epoch or 0.0)
        self.update()

    def set_progress(self, step: int):
        self._current = step
        self.update()

    def _epoch_for(self, pos: int) -> int:
        return round(pos / self._spe) if self._spe else 0

    def paintEvent(self, event):
        if not self._positions or self._total <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w = self.width()
        upcoming = [p for p in self._positions if p > self._current]
        nxt = min(upcoming) if upcoming else None
        # passed / future ticks (thin)
        for pos in self._positions:
            if pos == nxt:
                continue
            x = max(1, min(w - 1, int(w * (pos / self._total))))
            painter.setPen(QColor("#d4af37") if pos <= self._current else QColor("#3a3a1f"))
            painter.drawLine(x, 2, x, 13)
        # the next upcoming set — a taller red/flame mark with a tiny label underneath
        if nxt is not None:
            x = max(1, min(w - 1, int(w * (nxt / self._total))))
            painter.setPen(QColor("#ff7a18"))
            painter.drawLine(x, 0, x, 15)
            painter.setPen(QColor("#ff9a5c"))
            f = painter.font()
            f.setPixelSize(9)
            painter.setFont(f)
            ep = self._epoch_for(nxt)
            label = f"next preview · epoch {ep}" if ep else "next preview"
            tw = painter.fontMetrics().horizontalAdvance(label)
            lx = min(max(0, x - tw // 2), max(0, w - tw))
            painter.drawText(lx, 27, label)
        painter.end()


class TrainTab(QWidget):
    status_message = Signal(str)  # for main window status bar
    add_to_batch_requested = Signal(object)  # emits a RunDefinition
    load_set_requested = Signal(str, str)    # (dataset_folder, trigger_word)
    subject_type_changed = Signal()          # Person/Object/Style changed -> refresh rail
    run_progress = Signal(object)            # RunProgress payload mirrored onto Home
    optimizer_changed = Signal(str)          # preset label for the Home OPTIMIZER tile
    training_active = Signal(bool)           # run started/ended -> Home Start/Stop state

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dataset_path = ""
        self._image_count = 0
        self._training_params = {}
        self._config_path = ""
        self._dataset_config_path = ""
        self._trainer = TrainingProcess(self)
        self._setup_path = ""
        self._dit_path = ""
        self._qwen3_path = ""
        self._vae_path = ""
        self._output_dir = ""
        self._trigger_word = ""
        self._resume_state_path = None
        self._app_settings = None
        self._lms_url = "http://localhost:1234/v1"
        self._lms_model = ""
        # Tracks which dataset folder the sample box was auto-filled for, so a repeated
        # set_dataset for the same folder won't clobber edits.
        self._sample_autofill_folder = None
        # Text-encoder training: backend stays wired; UI toggle removed (rarely useful, only
        # worthwhile on large coherent datasets). Re-add a checkbox bound to this to restore it.
        self._train_text_encoder = False
        self._last_preview = []
        self._denoiser = LogDenoiser()
        self._stepping = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(3000)
        self._preview_timer.timeout.connect(self._poll_sample_dir)
        self._build_ui()
        self._connect_trainer()

    # ------------------------------------------------------------------
    # Public API — called by main window to sync shared state
    # ------------------------------------------------------------------

    def set_dataset(self, folder_path: str, image_count: int):
        self._dataset_path = folder_path
        self._image_count = image_count
        self._dataset_path_edit.setText(folder_path)
        self._maybe_autofill_sample_prompts(folder_path)
        self._apply_suggestion()
        self._recalculate()
        self._refresh_readiness_summary()

    def set_environment(self, sdscripts_path: str, dit_path: str, qwen3_path: str,
                        vae_path: str, output_dir: str):
        self._setup_path = sdscripts_path
        self._dit_path = dit_path
        self._qwen3_path = qwen3_path
        self._vae_path = vae_path
        self._output_dir = output_dir

    def set_trigger_word(self, trigger: str):
        self._trigger_word = trigger

    # ---- accessors the Home cockpit drives / reads (single source of truth) ----
    def set_lora_name(self, name: str):
        if name and name != self._lora_name_edit.text():
            self._lora_name_edit.setText(name)

    def get_lora_name(self) -> str:
        return self._lora_name_edit.text().strip()

    def set_subject_type(self, key: str):
        """Set the subject type from a RunDefinition key (character/concept|object/style)."""
        idx = {"character": 0, "person": 0, "face": 0,
               "concept": 1, "object": 1, "style": 2}.get((key or "").lower())
        if idx is not None and idx != self._subject_combo.currentIndex():
            self._subject_combo.setCurrentIndex(idx)  # fires _on_subject_changed (recalc)

    def get_subject_type(self) -> str:
        return self._lora_type_for_subject()

    def set_target_steps(self, n: int):
        n = int(n or 0)
        if n and n != self._target_steps_spin.value():
            self._target_steps_spin.setValue(n)  # fires _recalculate

    def get_target_steps(self) -> int:
        return self._target_steps_spin.value()

    def start_from_cockpit(self):
        """Entry point for Home's Run button — same path as Start Training, but unattended
        (no pre-flight confirm dialog; the VRAM safety guard still applies)."""
        self._start_training(confirm=False)

    def add_current_to_batch(self):
        """Public entry for Home's 'Add to Batch' — queues the current (cockpit-mirrored) run."""
        self._add_to_batch()

    def control_panel(self) -> QWidget:
        """Sample Previews + Actions + Advanced (set-once) controls, owned and wired by this
        tab but mounted onto the Home command center (single source of truth). Reparenting is
        a display-only move; all engine wiring stays intact."""
        return self._relocated_controls

    def step_calculator(self) -> QWidget:
        """The Step Calculator group (subject type, target steps, uncap, readout), owned and
        wired by this tab but displayed on Home — the single place to tune steps."""
        return self._step_calc_group

    def apply_preset(self, preset):
        """Apply a core.train_presets.TrainPreset (front-page picker). Sets values
        only — never starts a run. target_steps 0 = re-suggest from the dataset."""
        self.set_subject_type(preset.subject_type)
        self._set_optimizer(preset.optimizer)
        self._on_optimizer_changed()
        if preset.optimizer == "adamw8bit" and preset.learning_rate:
            self._lr_spin.setValue(preset.learning_rate)
        self._dim_spin.setValue(preset.network_dim)
        self._alpha_spin.setValue(preset.network_alpha)
        self._uncap_check.setChecked(preset.uncap_steps)
        if preset.target_steps:
            self.set_target_steps(preset.target_steps)
        else:
            self._apply_suggestion()
        self._recalculate()

    def apply_defaults(self, app_settings):
        """Initialize the Train tab controls from the App Defaults in Settings."""
        self._app_settings = app_settings
        self._set_optimizer(app_settings.get("default_optimizer") or "prodigy")
        self._on_optimizer_changed()  # sync LR row visibility + note + summary
        self._dim_spin.setValue(app_settings.get("default_network_dim"))
        self._alpha_spin.setValue(app_settings.get("default_network_alpha"))
        self._uncap_check.setChecked(app_settings.get("default_uncap_steps"))
        self._target_steps_spin.setValue(app_settings.get("default_target_steps"))
        self._train_text_encoder = app_settings.get("default_train_text_encoder")
        # Resumability + sample preview controls (shared AppSettings keys)
        self._ckpt_steps_spin.setValue(app_settings.get("save_every_n_steps"))
        self._sample_enable_check.setChecked(app_settings.get("sample_enable"))
        self._sample_prompts_edit.setPlainText(app_settings.get("sample_prompts"))
        self._sample_every_spin.setValue(app_settings.get("sample_every_n_epochs"))
        self._preview_count_spin.setValue(app_settings.get("sample_count"))
        self._update_gen_button_label()
        # Persist edits back to settings (connected after load so loading doesn't write)
        a = app_settings
        self._ckpt_steps_spin.valueChanged.connect(lambda v: a.set("save_every_n_steps", v))
        self._sample_enable_check.toggled.connect(lambda b: a.set("sample_enable", b))
        self._sample_enable_check.toggled.connect(self._update_sample_schedule)
        self._sample_prompts_edit.textChanged.connect(
            lambda: a.set("sample_prompts", self._sample_prompts_edit.toPlainText()))
        self._sample_every_spin.valueChanged.connect(lambda v: a.set("sample_every_n_epochs", v))
        self._preview_count_spin.valueChanged.connect(lambda v: a.set("sample_count", v))
        self._update_sample_schedule()

    def set_lmstudio_config(self, url: str, model: str):
        self._lms_url = url or "http://localhost:1234/v1"
        self._lms_model = model or ""

    def showEvent(self, event):
        """Re-sync sample prompts from settings so Setup-tab edits are reflected."""
        super().showEvent(event)
        if self._app_settings is not None:
            current = self._app_settings.get("sample_prompts")
            if current != self._sample_prompts_edit.toPlainText():
                self._sample_prompts_edit.blockSignals(True)
                self._sample_prompts_edit.setPlainText(current)
                self._sample_prompts_edit.blockSignals(False)
        self._refresh_readiness_summary()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # ---- Left panel: config & controls ----
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(20, 20, 12, 20)
        left_layout.setSpacing(14)
        left_widget.setMinimumWidth(340)

        # Crash-recovery banner (hidden unless an interrupted run is detected on launch)
        self._recovery_banner = QFrame()
        self._recovery_banner.setStyleSheet(
            "QFrame{background:#3a2f1a;border:1px solid #8a6d3b;border-radius:6px;}")
        rb = QHBoxLayout(self._recovery_banner)
        self._recovery_label = QLabel("")
        self._recovery_label.setWordWrap(True)
        rb.addWidget(self._recovery_label, 1)
        self._recovery_btn = QPushButton("Restore & Resume")
        self._recovery_dismiss = QPushButton("✕")
        self._recovery_dismiss.setFixedWidth(28)
        rb.addWidget(self._recovery_btn)
        rb.addWidget(self._recovery_dismiss)
        self._recovery_banner.setVisible(False)
        self._recovery_dismiss.clicked.connect(self._dismiss_recovery)
        left_layout.addWidget(self._recovery_banner)

        title = QLabel("Training Status")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #d4af37;")
        title.setToolTip("All run controls live on the Home page now — this tab is the live "
                         "preview: readiness, progress, log, and sample images.")
        left_layout.addWidget(title)

        # Low-VRAM active indicator (hidden unless the user enabled+acknowledged it).
        self._lowvram_indicator = QLabel("")
        self._lowvram_indicator.setWordWrap(True)
        self._lowvram_indicator.setStyleSheet(
            "color: #ff9a5c; background:#2a1c10; border:1px solid #8a5a12; "
            "border-radius:5px; padding:5px 8px; font-size:11px; font-weight:600;")
        self._lowvram_indicator.setVisible(False)
        left_layout.addWidget(self._lowvram_indicator)

        # Readiness summary — mirrors the global rail so the Train tab never looks
        # "blank": shows images / naming / caption state for the loaded dataset.
        self._readiness_summary = QLabel("Load a dataset in the Dataset tab to begin.")
        self._readiness_summary.setWordWrap(True)
        self._readiness_summary.setObjectName("train_readiness")
        left_layout.addWidget(self._readiness_summary)
        self._style_readiness("dim")

        # Settings summary (static — collapsed at the bottom)
        config_box = CollapsibleBox("Anima Settings")
        cg_layout = config_box.content_layout()
        self._config_summary = QLabel(get_config_summary())
        self._config_summary.setObjectName("label_config_summary")
        self._config_summary.setWordWrap(True)
        self._config_summary.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px; "
            "color: #9a9aa2; background-color: #0c0b0a; "
            "border: 1px solid #2a2a1e; border-radius: 5px; padding: 8px;"
        )
        cg_layout.addWidget(self._config_summary)

        # Dataset path
        ds_grp = QGroupBox("Dataset")
        ds_layout = QVBoxLayout(ds_grp)
        ds_row = QHBoxLayout()
        self._dataset_path_edit = QLineEdit()
        self._dataset_path_edit.setPlaceholderText("Load a folder in the Dataset tab…")
        self._dataset_path_edit.setReadOnly(True)
        ds_browse = QPushButton("Browse…")
        ds_browse.setFixedWidth(96)
        ds_browse.clicked.connect(self._browse_dataset)
        ds_row.addWidget(self._dataset_path_edit)
        ds_row.addWidget(ds_browse)
        ds_layout.addLayout(ds_row)

        # LoRA name — the required, lead-with-this field (large, bold, red while empty)
        name_grp = QGroupBox("LoRA Name  ★ required")
        name_layout = QVBoxLayout(name_grp)
        self._lora_name_edit = QLineEdit()
        self._lora_name_edit.setPlaceholderText("Name this LoRA first — e.g. my_character_v1")
        self._lora_name_edit.setMinimumHeight(40)
        self._lora_name_edit.textChanged.connect(self._on_lora_name_changed)
        self._lora_name_edit.textChanged.connect(self._update_name_required_style)
        name_layout.addWidget(self._lora_name_edit)

        # Saved Sets live under Advanced now (collapsed) — they're a power-user
        # convenience, not part of the core name→start path.
        sets_box = CollapsibleBox("Saved Sets")
        sets_layout = sets_box.content_layout()

        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Set name:"))
        self._set_name_edit = QLineEdit()
        self._set_name_edit.setPlaceholderText("name this set")
        save_row.addWidget(self._set_name_edit, 1)
        self._save_set_btn = QPushButton("💾 Save Set")
        save_row.addWidget(self._save_set_btn)
        sets_layout.addLayout(save_row)

        load_row = QHBoxLayout()
        load_row.addWidget(QLabel("Saved sets:"))
        self._sets_combo = QComboBox()
        self._sets_combo.setMinimumWidth(160)
        load_row.addWidget(self._sets_combo, 1)
        self._load_set_btn = QPushButton("📂 Load")
        self._delete_set_btn = QPushButton("🗑 Delete")
        load_row.addWidget(self._load_set_btn)
        load_row.addWidget(self._delete_set_btn)
        sets_layout.addLayout(load_row)

        self._save_set_btn.clicked.connect(self._save_set)
        self._load_set_btn.clicked.connect(self._load_set)
        self._delete_set_btn.clicked.connect(self._delete_set)
        self._refresh_sets_combo()

        # Optimizer & Network — expanded and first in the Train Presets panel (users
        # asked to actually see the optimizer presets).
        opt_box = CollapsibleBox("Optimizer & Network", expanded=True)
        opt_layout = opt_box.content_layout()
        opt_layout.setSpacing(6)

        # Two named presets. Prodigy+ ScheduleFree stays the forgiving default;
        # AdamW8bit + constant LR mirrors the classic Civitai trainer recipe for
        # comparison runs (user feedback round 1).
        optim_row = QHBoxLayout()
        optim_row.addWidget(QLabel("Optimizer preset:"))
        self._optimizer_combo = QComboBox()
        self._optimizer_combo.addItem(
            "Prodigy+ ScheduleFree — auto LR (recommended)", "prodigy")
        self._optimizer_combo.addItem(
            "AdamW8bit — constant LR (Civitai classic)", "adamw8bit")
        self._optimizer_combo.currentIndexChanged.connect(self._on_optimizer_changed)
        optim_row.addWidget(self._optimizer_combo, 1)
        opt_layout.addLayout(optim_row)

        # Learning rate — only meaningful (and only shown) for AdamW8bit; Prodigy is
        # learning-rate-free.
        self._lr_row_widget = QWidget()
        lr_row = QHBoxLayout(self._lr_row_widget)
        lr_row.setContentsMargins(0, 0, 0, 0)
        lr_row.addWidget(QLabel("Learning rate:"))
        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setDecimals(6)
        self._lr_spin.setRange(0.000001, 0.01)
        self._lr_spin.setValue(1e-4)
        lr_row.addWidget(self._lr_spin)
        lr_row.addStretch()
        opt_layout.addWidget(self._lr_row_widget)
        self._lr_row_widget.setVisible(False)  # shown only for AdamW8bit

        dim_row = QHBoxLayout()
        dim_row.addWidget(QLabel("Network dim:"))
        self._dim_spin = QSpinBox()
        self._dim_spin.setRange(1, 128)
        self._dim_spin.setValue(16)
        self._dim_spin.valueChanged.connect(self._update_config_summary)
        dim_row.addWidget(self._dim_spin)
        dim_row.addWidget(QLabel("Alpha:"))
        self._alpha_spin = QSpinBox()
        self._alpha_spin.setRange(1, 128)
        self._alpha_spin.setValue(8)
        self._alpha_spin.valueChanged.connect(self._update_config_summary)
        dim_row.addWidget(self._alpha_spin)
        dim_row.addStretch()
        opt_layout.addLayout(dim_row)

        self._opt_note = QLabel(self._OPT_NOTES["prodigy"])
        self._opt_note.setWordWrap(True)
        self._opt_note.setStyleSheet("font-size: 10px; color: #8a8a93; font-style: italic;")
        opt_layout.addWidget(self._opt_note)

        # Run Options (bucketing / state / continue / metadata) — static, collapsed
        run_box = CollapsibleBox("Run Options")
        run_layout = run_box.content_layout()
        run_layout.setSpacing(6)

        self._bucket_check = QCheckBox("Enable aspect-ratio bucketing (no cropping)")
        self._bucket_check.setChecked(True)
        run_layout.addWidget(self._bucket_check)

        self._save_state_check = QCheckBox("Save training state (resumable)")
        self._save_state_check.setChecked(True)
        self._save_state_check.setToolTip(
            "Write a resumable training state so you can Stop and continue later."
        )
        run_layout.addWidget(self._save_state_check)

        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(QLabel("Checkpoint every"))
        self._ckpt_steps_spin = QSpinBox()
        self._ckpt_steps_spin.setRange(0, 5000)
        self._ckpt_steps_spin.setSingleStep(50)
        self._ckpt_steps_spin.setValue(250)
        self._ckpt_steps_spin.setToolTip(
            "Save a resumable state this often (in steps) so a mid-epoch Stop loses at most "
            "this many steps. 0 = save only at epoch boundaries."
        )
        ckpt_row.addWidget(self._ckpt_steps_spin)
        ckpt_row.addWidget(QLabel("steps (0 = per epoch)"))
        ckpt_row.addStretch()
        run_layout.addLayout(ckpt_row)
        self._save_state_check.toggled.connect(
            lambda b: self._ckpt_steps_spin.setEnabled(b)
        )

        self._resume_check = QCheckBox("Resume from last saved state")
        self._resume_check.setChecked(False)
        self._resume_check.setVisible(False)
        run_layout.addWidget(self._resume_check)

        self._metadata_check = QCheckBox("Embed trigger word in LoRA metadata")
        self._metadata_check.setChecked(True)
        run_layout.addWidget(self._metadata_check)

        cont_row = QHBoxLayout()
        cont_row.addWidget(QLabel("Start from existing LoRA:"))
        self._network_weights_edit = QLineEdit()
        self._network_weights_edit.setPlaceholderText("optional .safetensors to continue training")
        cont_row.addWidget(self._network_weights_edit)
        cont_browse = QPushButton("Browse…")
        cont_browse.setFixedWidth(96)
        cont_browse.clicked.connect(self._browse_network_weights)
        cont_row.addWidget(cont_browse)
        run_layout.addLayout(cont_row)

        # Sample Previews (auto-generated images shown on the right during training)
        sample_box = CollapsibleBox("Sample Previews", expanded=True)
        sample_layout = sample_box.content_layout()
        sample_layout.setSpacing(6)

        self._sample_enable_check = QCheckBox("Auto-generate preview images during training")
        self._sample_enable_check.setChecked(True)
        sample_layout.addWidget(self._sample_enable_check)

        sample_layout.addWidget(QLabel("Sample prompts (one per line; trigger auto-prepended):"))
        self._sample_prompts_edit = QTextEdit()
        self._sample_prompts_edit.setFixedHeight(60)
        self._sample_prompts_edit.setPlaceholderText(
            "Auto-fills with random real captions once every image is captioned — "
            "or grab a fresh set →"
        )
        sample_layout.addWidget(self._sample_prompts_edit)

        sample_btn_row = QHBoxLayout()
        sample_btn_row.addWidget(QLabel("Preview images:"))
        # Hard-gated at 4 — the preview grid is a fixed 4-wide row per epoch (two rows shown).
        self._preview_count_spin = QSpinBox()
        self._preview_count_spin.setRange(4, 4)
        self._preview_count_spin.setValue(4)
        self._preview_count_spin.setEnabled(False)
        self._preview_count_spin.setToolTip("Fixed at 4 — one row of four preview images per epoch.")
        self._preview_count_spin.valueChanged.connect(self._update_gen_button_label)
        sample_btn_row.addWidget(self._preview_count_spin)
        self._gen_prompts_btn = QPushButton("🎲 Grab 4 random captions")
        self._gen_prompts_btn.setToolTip(
            "Fill the box with that many of the dataset's actual captions, chosen at random, "
            "so previews render from prompts that look exactly like the training data. "
            "Requires every image to be captioned."
        )
        self._gen_prompts_btn.clicked.connect(self._on_grab_prompts_clicked)
        sample_btn_row.addWidget(self._gen_prompts_btn)
        sample_btn_row.addStretch()
        sample_layout.addLayout(sample_btn_row)

        every_row = QHBoxLayout()
        every_row.addWidget(QLabel("Render every"))
        self._sample_every_spin = QSpinBox()
        self._sample_every_spin.setRange(1, 50)
        self._sample_every_spin.setValue(1)
        self._sample_every_spin.valueChanged.connect(self._update_sample_schedule)
        every_row.addWidget(self._sample_every_spin)
        every_row.addWidget(QLabel("epoch(s)"))
        every_row.addStretch()
        sample_layout.addLayout(every_row)

        # Step calculator
        calc_grp = QGroupBox("Step Calculator")
        calc_layout = QVBoxLayout(calc_grp)

        # Subject type
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Subject type:"))
        self._subject_combo = QComboBox()
        self._subject_combo.addItems([
            "Person / Face",
            "Object / Concept",
            "Art Style",
        ])
        self._subject_combo.setToolTip(
            "Person / Face  — overfit risk is high, keep steps lower\n"
            "Object / Concept — forgiving, moderate steps\n"
            "Art Style — needs higher steps to capture subtle nuance"
        )
        self._subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        type_row.addWidget(self._subject_combo, 1)
        calc_layout.addLayout(type_row)

        # Target steps
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target steps:"))
        self._target_steps_spin = QSpinBox()
        self._target_steps_spin.setRange(500, 40000)
        self._target_steps_spin.setSingleStep(100)
        self._target_steps_spin.setValue(500)
        self._target_steps_spin.setToolTip("Auto-suggested to hit the subject type's exposures-per-image target. Override freely.")
        self._target_steps_spin.valueChanged.connect(self._recalculate)
        target_row.addWidget(self._target_steps_spin)
        self._suggest_label = QLabel("")
        self._suggest_label.setStyleSheet("color: #8a8a93; font-size: 11px; font-style: italic;")
        target_row.addWidget(self._suggest_label)
        target_row.addStretch()
        calc_layout.addLayout(target_row)

        # Uncap toggle — power-user escape hatch. Off by default; the auto suggestion is
        # capped at SOFT_CAP_STEPS so a large dataset won't run overnight. Checking it
        # lets the suggestion scale past the cap (the floor still applies).
        uncap_row = QHBoxLayout()
        self._uncap_check = QCheckBox("Remove step cap (advanced)")
        self._uncap_check.setToolTip(
            f"By default the auto-suggested steps are capped at {SOFT_CAP_STEPS:,} so a "
            "large dataset won't run for many hours. Check this to let the suggestion "
            "scale with dataset size past the cap."
        )
        self._uncap_check.toggled.connect(self._on_uncap_toggled)
        uncap_row.addWidget(self._uncap_check)
        uncap_row.addStretch()
        calc_layout.addLayout(uncap_row)

        self._uncap_warn = QLabel(
            "⚠ Uncapped runs scale with dataset size and can take many hours — and "
            "over-training can degrade the LoRA. Only push past "
            f"{SOFT_CAP_STEPS:,} if you know you need it."
        )
        self._uncap_warn.setWordWrap(True)
        self._uncap_warn.setStyleSheet("color: #e0a93c; font-size: 11px;")
        self._uncap_warn.setVisible(False)
        calc_layout.addWidget(self._uncap_warn)

        self._step_calc_label = QLabel("No images loaded")
        self._step_calc_label.setObjectName("label_step_calc")
        self._step_calc_label.setWordWrap(True)
        self._step_calc_label.setStyleSheet(
            "color: #f4d160; font-size: 13px; font-weight: 600; "
            "padding: 6px; background-color: #161208; "
            "border: 1px solid #3a3a1f; border-radius: 5px;"
        )
        calc_layout.addWidget(self._step_calc_label)

        # Action buttons
        btn_grp = QGroupBox("Actions")
        btn_layout = QVBoxLayout(btn_grp)
        btn_layout.setSpacing(8)

        # Preview Config lives under Advanced — Start already regenerates the config
        # on every launch, so it's an inspection aid, not part of the main path. It
        # generates the real TOMLs and shows them (user feedback: config preview).
        gen_btn = QPushButton("⚙ Preview Config Files")
        gen_btn.setObjectName("btn_primary")
        gen_btn.setToolTip("Generate the exact TOML files training will use and view them")
        gen_btn.clicked.connect(self._preview_config)

        add_batch_btn = QPushButton("➕ Add to Batch")
        add_batch_btn.setToolTip("Snapshot the current setup as a queued run on the Batch tab")
        add_batch_btn.clicked.connect(self._add_to_batch)
        btn_layout.addWidget(add_batch_btn)

        lowvram_btn = QPushButton("🧰 Low VRAM…")
        lowvram_btn.setToolTip("For small GPUs only — fit training on less VRAM (same quality, slower)")
        lowvram_btn.clicked.connect(self._open_lowvram)
        btn_layout.addWidget(lowvram_btn)

        forge_row = QHBoxLayout()
        deliver_btn = QPushButton("📤 Deliver to Forge")
        deliver_btn.setToolTip("Copy the trained LoRA into Forge's models/Lora folder")
        deliver_btn.clicked.connect(self._deliver_to_forge)
        test_btn = QPushButton("🖼 Test in Forge")
        test_btn.setToolTip("Render the LoRA via Forge's txt2img API and show the results")
        test_btn.clicked.connect(self._test_in_forge)
        forge_row.addWidget(deliver_btn)
        forge_row.addWidget(test_btn)
        btn_layout.addLayout(forge_row)

        comfy_btn = QPushButton("📤 Deliver to ComfyUI")
        comfy_btn.setToolTip("Copy the trained LoRA into your ComfyUI models/loras folder "
                             "(set it in Setup → Forge / API)")
        comfy_btn.clicked.connect(self._deliver_to_comfyui)
        btn_layout.addWidget(comfy_btn)

        self._config_path_label = QLabel("No config generated yet")
        self._config_path_label.setObjectName("label_field")
        self._config_path_label.setWordWrap(True)
        self._config_path_label.setStyleSheet("font-size: 11px; color: #8a8a93;")
        btn_layout.addWidget(self._config_path_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #2a2a1e;")
        btn_layout.addWidget(sep)

        start_stop_row = QHBoxLayout()
        self._start_btn = QPushButton("▶  Start Training")
        self._start_btn.setObjectName("btn_start")
        self._start_btn.clicked.connect(lambda: self._start_training(confirm=True))

        self._stop_btn = QPushButton("■  Stop Training")
        self._stop_btn.setObjectName("btn_stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_training)

        start_stop_row.addWidget(self._start_btn)
        start_stop_row.addWidget(self._stop_btn)
        btn_layout.addLayout(start_stop_row)
        # Launching lives on the FRONT only — a second Start inside the Train Options
        # modal confused a tester ("Home has Start, Options has another Start"). The
        # buttons stay alive (run-state code toggles their enabled flags) but hidden.
        self._start_btn.setVisible(False)
        self._stop_btn.setVisible(False)

        # ---- Assemble left panel ----
        # LoRA name + dataset path are owned here (engine reads them) but the front page is
        # the single visible source — keep them parented (alive + kept in sync by Home) yet
        # hidden on this preview tab. The Step Calculator is relocated onto Home.
        name_grp.setParent(left_widget)
        name_grp.setVisible(False)
        ds_grp.setParent(left_widget)
        ds_grp.setVisible(False)
        self._step_calc_group = calc_grp

        # Sample Previews, Actions, and the Advanced set-once block are relocated onto the
        # Home command center (single source of truth). They stay owned & wired by this tab
        # — all engine logic is untouched — but are displayed on Home via control_panel().
        # Two-column layout: the modal was a single tall stack and felt cramped
        # (user feedback asked for a full Presets page — the columns give it page
        # room while Home stays the only control surface).
        relocated = QWidget()
        rl = QHBoxLayout(relocated)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(18)

        # Left column — the per-run dials: optimizer presets (most-requested) + previews.
        left_col = QVBoxLayout()
        left_col.setSpacing(14)
        left_col.addWidget(opt_box)
        left_col.addWidget(sample_box)
        left_col.addStretch()

        # Right column — actions + the set-once advanced blocks.
        right_col = QVBoxLayout()
        right_col.setSpacing(14)
        right_col.addWidget(btn_grp)
        adv_label = QLabel("Advanced (set once)")
        adv_label.setStyleSheet("color: #6a6a72; font-size: 11px; font-weight: 600; "
                                "letter-spacing: 1px; padding-top: 6px;")
        right_col.addWidget(adv_label)
        right_col.addWidget(gen_btn)
        for box in (sets_box, run_box, config_box):
            right_col.addWidget(box)
        right_col.addStretch()

        rl.addLayout(left_col, 1)
        rl.addLayout(right_col, 1)
        self._relocated_controls = relocated

        left_layout.addStretch()
        self._update_name_required_style()
        self._refresh_lowvram_indicator()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setMinimumWidth(360)
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # ---- Right panel: progress + log ----
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(12, 20, 20, 20)
        right_layout.setSpacing(10)

        log_title = QLabel("Training Progress & Log")
        log_title.setStyleSheet("font-size: 16px; font-weight: 700; color: #d4af37;")
        right_layout.addWidget(log_title)

        # Thick shared progress widget (same one used on Home)
        self._run_progress = RunProgress()
        right_layout.addWidget(self._run_progress)

        # Tick marks showing where preview images will render
        self._tickbar = TickBar()
        right_layout.addWidget(self._tickbar)

        # Analog dials — Epoch / Loss / Speed / ETA, live from the training log
        self._dials = DialRow()
        right_layout.addWidget(self._dials)

        # Live sample preview (filled from {output_dir}/sample during training).
        # Per the handoff: a 4-wide grid bounded to ~2 visible rows, scroll for older sets.
        # Preview header: title + display-mode toggle (flat newest-first grid vs
        # labeled per-epoch rows for side-by-side checkpoint comparison). A view
        # switch on a preview surface — displays state, drives nothing.
        preview_head = QHBoxLayout()
        preview_title = QLabel("Live Preview")
        preview_title.setStyleSheet("font-size: 12px; font-weight: 600; color: #8a8a93;")
        preview_head.addWidget(preview_title)
        preview_head.addStretch()
        self._compare_toggle = QCheckBox("Compare epochs")
        self._compare_toggle.setToolTip(
            "Group previews by epoch, newest on top — spot the earliest epoch that "
            "already looks right and stop there.")
        self._compare_toggle.toggled.connect(
            lambda _on: self._render_preview(getattr(self, "_preview_files", []) or []))
        preview_head.addWidget(self._compare_toggle)
        right_layout.addLayout(preview_head)
        self._preview_container = QWidget()
        self._preview_grid = QGridLayout(self._preview_container)
        self._preview_grid.setContentsMargins(0, 0, 0, 0)
        self._preview_grid.setHorizontalSpacing(8)
        self._preview_grid.setVerticalSpacing(8)
        self._preview_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._preview_hint = QLabel("Preview images appear here as training reaches each tick mark above.")
        self._preview_hint.setStyleSheet("font-size: 11px; color: #8a8a93; font-style: italic;")
        self._preview_grid.addWidget(self._preview_hint, 0, 0, 1, 4)
        self._preview_scroll = QScrollArea()
        self._preview_scroll.setWidgetResizable(True)
        self._preview_scroll.setWidget(self._preview_container)
        self._preview_scroll.setFixedHeight(366)  # two rows of ~168px thumbs + gap
        self._preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_layout.addWidget(self._preview_scroll)

        # Log output
        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setObjectName("log_output")
        self._log_edit.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 12px; "
            "background-color: #0c0b0a; color: #c6c6ce; "
            "border: 1px solid #2a2a1e; border-radius: 5px;"
        )
        right_layout.addWidget(self._log_edit)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([640, 460])

    # ------------------------------------------------------------------
    # Trainer signals
    # ------------------------------------------------------------------

    def _connect_trainer(self):
        self._trainer.training_started.connect(self._on_training_started)
        self._trainer.training_finished.connect(self._on_training_finished)
        self._trainer.log_line.connect(self._on_log_line)
        self._trainer.progress_updated.connect(self._on_progress_updated)

    def _rp(self, **payload):
        """Update the Train tab's RunProgress and mirror the same payload to Home."""
        self._run_progress.apply(payload)
        self.run_progress.emit(payload)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _suggested_steps(self) -> int:
        """Recommended step count from subject type, image count, and roster size.

        Delegates to the shared core.step_calculator so Home and Train never drift.
        """
        return suggest_target_steps(
            self._lora_type_for_subject(), self._image_count,
            n_characters=self._detected_character_count(),
            uncapped=self._uncap_check.isChecked())

    def _detected_character_count(self) -> int:
        """Distinct named characters in the loaded dataset's roster (>=1)."""
        if not self._dataset_path:
            return 1
        try:
            from core import characters as ch
            n = len([c for c in ch.load(self._dataset_path).roster if c.token.strip()])
            return max(n, 1)
        except Exception:
            return 1

    _OPT_NOTES = {
        "prodigy": ("Prodigy+ ScheduleFree auto-tunes the learning rate and needs no "
                    "schedule — the most forgiving choice for anchoring concepts."),
        "adamw8bit": ("AdamW8bit with a constant learning rate — the classic Civitai-era "
                      "recipe, handy for comparing against older runs. Uses the learning "
                      "rate above (default 1e-4)."),
    }
    _OPT_TILE = {"prodigy": "Prodigy", "adamw8bit": "AdamW8bit"}

    def _current_optimizer(self) -> str:
        return self._optimizer_combo.currentData() or "prodigy"

    def optimizer_label(self) -> str:
        """Short name for the Home cockpit tile."""
        return self._OPT_TILE.get(self._current_optimizer(), "Prodigy")

    def _set_optimizer(self, optimizer: str):
        idx = self._optimizer_combo.findData((optimizer or "prodigy").lower())
        self._optimizer_combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _on_optimizer_changed(self, *_):
        opt = self._current_optimizer()
        self._lr_row_widget.setVisible(opt == "adamw8bit")
        self._opt_note.setText(self._OPT_NOTES.get(opt, ""))
        self.optimizer_changed.emit(self.optimizer_label())
        self._update_config_summary()

    def _update_config_summary(self):
        self._config_summary.setText(get_config_summary(
            optimizer=self._current_optimizer(),
            network_dim=self._dim_spin.value(),
            network_alpha=self._alpha_spin.value(),
            train_text_encoder=self._train_text_encoder,
        ))

    def _on_subject_changed(self):
        self._apply_suggestion()
        self._recalculate()
        self.subject_type_changed.emit()
        self._refresh_readiness_summary()

    def _on_uncap_toggled(self, checked: bool):
        """Show/hide the warning, persist the choice, and re-suggest under the new cap state."""
        self._uncap_warn.setVisible(checked)
        if self._app_settings:
            self._app_settings.set("default_uncap_steps", checked)
        self._apply_suggestion()
        self._recalculate()

    def is_style_subject(self) -> bool:
        """True when training an art style — naming is genuinely not needed then."""
        return self._subject_combo.currentText().strip().lower().startswith("art style")

    def _style_readiness(self, level: str):
        """Color the readiness summary: 'dim' (no dataset), 'ok' (green), 'warn' (amber)."""
        color = {"dim": "#8a8a93", "ok": "#7ed957", "warn": "#e0a93c"}.get(level, "#8a8a93")
        self._readiness_summary.setStyleSheet(
            f"color:{color}; font-size:12px; font-weight:600; padding:6px 8px; "
            "background-color:#161208; border:1px solid #3a3a1f; border-radius:5px;")

    def _refresh_readiness_summary(self):
        """Recompute the at-a-glance readiness line from the loaded dataset folder."""
        from core import workflow
        folder = self._dataset_path
        if not folder:
            self._readiness_summary.setText("Load a dataset in the Dataset tab to begin.")
            self._style_readiness("dim")
            return
        load = workflow.dataset_state(folder)
        name = workflow.naming_state(folder)
        cap = workflow.caption_state(folder)
        parts = [f"✓ {load['images']} images" if load["done"] else "⚠ no images found"]
        if self.is_style_subject():
            parts.append("naming not needed for styles")
        elif name["done"]:
            parts.append(f"✓ {name['named']} named")
        else:
            parts.append("naming optional")
        if cap["done"]:
            parts.append("✓ captioned")
        elif cap["captioned"]:
            parts.append(f"⚠ {cap['captioned']}/{cap['images']} captioned")
        else:
            parts.append("⚠ not captioned")
        self._readiness_summary.setText("    ·    ".join(parts))
        self._style_readiness("ok" if (load["done"] and cap["done"]) else "warn")

    def _apply_suggestion(self):
        if self._image_count <= 0:
            self._suggest_label.setText("")
            return
        suggested = self._suggested_steps()
        # Block valueChanged signal so we don't double-recalculate
        self._target_steps_spin.blockSignals(True)
        self._target_steps_spin.setValue(suggested)
        self._target_steps_spin.blockSignals(False)
        if (not self._uncap_check.isChecked()
                and is_capped(self._lora_type_for_subject(), self._image_count)):
            self._suggest_label.setText(
                f"(suggested · capped at {SOFT_CAP_STEPS:,} — uncap to train longer)")
        else:
            self._suggest_label.setText("(suggested)")

    def _browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder", self._dataset_path or ""
        )
        if folder:
            from core.dataset_manager import count_images_in_folder
            count = count_images_in_folder(folder)
            self.set_dataset(folder, count)

    def _update_name_required_style(self):
        """Red outline + bold gold text on the required LoRA-name field while it's empty."""
        empty = not self._lora_name_edit.text().strip()
        if empty:
            self._lora_name_edit.setStyleSheet(
                "QLineEdit { font-size: 16px; font-weight: 700; color: #f4d160; "
                "border: 2px solid #b8442e; border-radius: 6px; padding: 4px 10px; }"
                "QLineEdit:focus { border: 2px solid #d4af37; }"
            )
        else:
            self._lora_name_edit.setStyleSheet(
                "QLineEdit { font-size: 16px; font-weight: 700; color: #f4d160; "
                "border: 1px solid #3a3a1f; border-radius: 6px; padding: 4px 10px; }"
                "QLineEdit:focus { border: 1px solid #d4af37; }"
            )

    def _on_lora_name_changed(self):
        # Reset config path label when name changes
        self._config_path_label.setText("Config not yet generated.")
        self._config_path_label.setStyleSheet("font-size: 11px; color: #8a8a93;")
        self._refresh_resume_option()

    def build_run_definition(self):
        """Snapshot current settings into a RunDefinition. Returns (rd|None, message)."""
        from core.batch import RunDefinition
        valid, msg = self._validate_for_config()
        if not valid:
            return None, msg
        self._ensure_sample_prompts()
        rd = RunDefinition(
            lora_name=self._lora_name_edit.text().strip(),
            dataset_folder=self._dataset_path,
            image_count=self._image_count,
            trigger_word=self._trigger_word,
            optimizer=self._current_optimizer(),
            learning_rate=self._lr_spin.value(),
            network_dim=self._dim_spin.value(),
            network_alpha=self._alpha_spin.value(),
            train_text_encoder=self._train_text_encoder,
            target_steps=self._target_steps_spin.value(),
            enable_bucket=self._bucket_check.isChecked(),
            save_state=self._save_state_check.isChecked(),
            save_every_n_steps=self._ckpt_steps_spin.value(),
            network_weights=self._network_weights_edit.text().strip(),
            embed_metadata=self._metadata_check.isChecked(),
            sample_enabled=self._sample_enable_check.isChecked(),
            sample_prompts=[ln for ln in self._sample_prompts_edit.toPlainText().splitlines() if ln.strip()],
            sample_every=self._sample_every_spin.value(),
            sample_count=self._preview_count_spin.value(),
            subject_type=self._lora_type_for_subject(),
            sdscripts_path=self._setup_path,
            dit_path=self._dit_path,
            qwen3_path=self._qwen3_path,
            vae_path=self._vae_path,
            output_dir=self._output_dir,
        )
        return rd, ""

    def apply_run_definition(self, rd):
        """Inverse of build_run_definition: push a saved set back into the widgets."""
        self._lora_name_edit.setText(rd.lora_name or "")
        self._set_optimizer(getattr(rd, "optimizer", "prodigy"))
        self._lr_spin.setValue(rd.learning_rate)
        self._dim_spin.setValue(rd.network_dim)
        self._alpha_spin.setValue(rd.network_alpha)
        self._train_text_encoder = rd.train_text_encoder
        self._target_steps_spin.setValue(rd.target_steps)
        self._bucket_check.setChecked(rd.enable_bucket)
        self._save_state_check.setChecked(rd.save_state)
        self._ckpt_steps_spin.setValue(rd.save_every_n_steps)
        self._network_weights_edit.setText(rd.network_weights or "")
        self._metadata_check.setChecked(rd.embed_metadata)
        self._sample_enable_check.setChecked(rd.sample_enabled)
        self._sample_prompts_edit.setPlainText("\n".join(rd.sample_prompts or []))
        self._sample_every_spin.setValue(rd.sample_every or 1)
        self._preview_count_spin.setValue(getattr(rd, "sample_count", 4) or 4)
        self._update_gen_button_label()
        idx = {"character": 0, "concept": 1, "style": 2}.get(rd.subject_type, 0)
        self._subject_combo.setCurrentIndex(idx)
        # Internal dataset/trigger state used by config generation
        self._dataset_path = rd.dataset_folder
        self._image_count = rd.image_count
        self._trigger_word = rd.trigger_word
        if rd.dataset_folder:
            self._dataset_path_edit.setText(rd.dataset_folder)
        self._recalculate()
        self._refresh_resume_option()
        # Ask MainWindow to load the dataset into the Dataset tab + restore trigger
        self.load_set_requested.emit(rd.dataset_folder or "", rd.trigger_word or "")

    def _refresh_sets_combo(self):
        from core import sets
        self._sets_combo.blockSignals(True)
        self._sets_combo.clear()
        self._sets_combo.addItems(sets.list_sets())
        self._sets_combo.blockSignals(False)

    def _save_set(self):
        from core import sets
        rd, msg = self.build_run_definition()
        if rd is None:
            QMessageBox.warning(self, "Cannot Save Set", msg)
            return
        name = self._set_name_edit.text().strip()
        decision = sets.set_save_decision(name, sets.list_sets())
        if decision == "empty":
            QMessageBox.information(self, "Save Set", "Enter a set name first.")
            return
        if decision == "exists":
            if QMessageBox.question(
                self, "Overwrite Set",
                f"A set named '{name}' already exists. Overwrite it?",
            ) != QMessageBox.Yes:
                return
        sets.save_set(name, rd)
        self._refresh_sets_combo()
        self._sets_combo.setCurrentText(name)
        self._set_name_edit.clear()
        self.status_message.emit(f"Saved set '{name}'.")

    def _load_set(self):
        from core import sets
        name = self._sets_combo.currentText().strip()
        if not name:
            return
        rd = sets.load_set(name)
        if rd is None:
            QMessageBox.warning(self, "Load Set", f"Could not load set '{name}'.")
            return
        self.apply_run_definition(rd)
        self.status_message.emit(f"Loaded set '{name}'.")

    def _delete_set(self):
        from core import sets
        name = self._sets_combo.currentText().strip()
        if not name:
            return
        if QMessageBox.question(self, "Delete Set", f"Delete set '{name}'?") \
                != QMessageBox.Yes:
            return
        sets.delete_set(name)
        self._refresh_sets_combo()
        self.status_message.emit(f"Deleted set '{name}'.")

    def show_recovery_banner(self, rd):
        from core.state_utils import find_saved_state
        from pathlib import Path as _P
        state = find_saved_state(rd.output_dir, rd.lora_name)
        where = _P(state).name if state else "a saved state"
        self._recovery_label.setText(
            f"⚠ '{rd.lora_name}' was interrupted — resume from {where}?")
        if getattr(self, "_recovery_connected", False):
            self._recovery_btn.clicked.disconnect()
        self._recovery_btn.clicked.connect(lambda: self._restore_interrupted(rd))
        self._recovery_connected = True
        self._recovery_banner.setVisible(True)

    def _dismiss_recovery(self):
        self._recovery_banner.setVisible(False)

    def _restore_interrupted(self, rd):
        self.apply_run_definition(rd)
        if self._resume_state_path:
            self._resume_check.setChecked(True)
        self._recovery_banner.setVisible(False)
        self.status_message.emit(
            f"Restored '{rd.lora_name}'. Review and Start to resume.")

    def _add_to_batch(self):
        rd, msg = self.build_run_definition()
        if rd is None:
            QMessageBox.warning(self, "Cannot Add to Batch", msg)
            return
        self.add_to_batch_requested.emit(rd)
        self.status_message.emit(f"Added '{rd.lora_name}' to batch queue.")

    def _browse_network_weights(self):
        start = self._output_dir or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select LoRA to continue from", start, "Safetensors (*.safetensors)")
        if path:
            self._network_weights_edit.setText(path)

    # ------------------------------------------------------------------
    # Sample prompts + preview schedule
    # ------------------------------------------------------------------

    def _lora_type_for_subject(self) -> str:
        return {0: "character", 1: "concept", 2: "style"}.get(
            self._subject_combo.currentIndex(), "")

    def _maybe_autofill_sample_prompts(self, folder_path: str):
        """On a NEW dataset, refresh the sample box from the dataset's own captions:
        grab N random real caption blocks when every image is captioned, otherwise
        clear it so a previous dataset's prompts never linger. Fires once per folder."""
        if not folder_path or folder_path == self._sample_autofill_folder:
            return
        self._sample_autofill_folder = folder_path
        from core.workflow import caption_state
        if caption_state(folder_path).get("done"):
            self._fill_sample_prompts(folder_path)
        else:
            self._sample_prompts_edit.clear()

    def _fill_sample_prompts(self, folder_path: str):
        """Drop N random verbatim caption blocks (N = preview count) into the box."""
        from core.sample_prompts import grab_caption_blocks
        n = self._preview_count_spin.value()
        blocks = grab_caption_blocks(folder_path, n)
        if not blocks:
            return
        self._sample_prompts_edit.setPlainText("\n".join(blocks))
        self._sample_enable_check.setChecked(True)
        self.status_message.emit(f"Filled {len(blocks)} sample prompt(s) from the dataset.")

    def _ensure_sample_prompts(self):
        """Guarantee real dataset captions back the previews before a run is snapshotted
        or launched. The box is session-ephemeral and the load-time autofill fires before
        captioning exists (load → caption → train), so an untouched box would otherwise
        reach the trainer empty and previews would collapse to a single trigger-only
        image per epoch. Fills only when empty — never clobbers authored prompts."""
        if not self._sample_enable_check.isChecked():
            return
        if self._sample_prompts_edit.toPlainText().strip():
            return
        if not self._dataset_path:
            return
        from core.workflow import caption_state
        if caption_state(self._dataset_path).get("done"):
            self._fill_sample_prompts(self._dataset_path)

    def refresh_sample_prompts_from_captions(self, *_):
        """Captioning just finished: fill the (empty) sample box from the new captions
        so the prompts are visible/editable before launch. Wired in MainWindow."""
        self._ensure_sample_prompts()

    def _on_grab_prompts_clicked(self):
        if not self._dataset_path or not Path(self._dataset_path).is_dir():
            QMessageBox.warning(self, "No Dataset", "Load a captioned dataset first.")
            return
        from core.workflow import caption_state
        if not caption_state(self._dataset_path).get("done"):
            QMessageBox.warning(
                self, "Not Fully Captioned",
                "Caption every image first — sample prompts are grabbed from the finished "
                "captions.")
            return
        self._fill_sample_prompts(self._dataset_path)

    def _update_gen_button_label(self, *_):
        self._gen_prompts_btn.setText(
            f"🎲 Grab {self._preview_count_spin.value()} random captions")

    def _dataset_style_anchor(self):
        """The style activation word (@-anchor) for the loaded dataset, if any."""
        if not self._dataset_path:
            return ""
        try:
            from core import characters as ch
            return ch.load(self._dataset_path).style_anchor.strip()
        except Exception:
            return ""

    def _sample_positions(self):
        """Return the step numbers at which preview images will render, for the tick bar."""
        params = self._training_params
        total = params.get("total_steps", 0)
        epochs = params.get("epochs", 0)
        if not (total and epochs) or not self._sample_enable_check.isChecked():
            return [], total
        every = max(1, self._sample_every_spin.value())
        steps_per_epoch = total / epochs
        positions = []
        if self._app_settings and self._app_settings.get("sample_at_first"):
            positions.append(0)
        e = every
        while e <= epochs:
            positions.append(round(e * steps_per_epoch))
            e += every
        return positions, total

    def _update_sample_schedule(self):
        positions, total = self._sample_positions()
        p = self._training_params
        spe = (p.get("total_steps", 0) / p["epochs"]) if p.get("epochs") else 0.0
        self._tickbar.set_schedule(positions, total, spe)

    def _refresh_resume_option(self):
        from core.state_utils import find_saved_state
        from pathlib import Path as _P
        name = self._lora_name_edit.text().strip()
        state = find_saved_state(self._output_dir, name) if (self._output_dir and name) else None
        self._resume_state_path = state
        if state:
            self._resume_check.setText(f"Resume from last saved state ({_P(state).name})")
            self._resume_check.setVisible(True)
        else:
            self._resume_check.setVisible(False)
            self._resume_check.setChecked(False)

    def _recalculate(self):
        if self._image_count > 0:
            target = self._target_steps_spin.value()
            params = calculate_training_params(self._image_count, target_steps=target)
            self._training_params = params
            text = format_calculation_string(params)
            if self._uncap_check.isChecked() and target > SOFT_CAP_STEPS:
                text += (f"\n⚠ Uncapped — {target:,} steps. Long run; watch for over-training.")
                level = "warn"
            else:
                level = "ok"
            self._step_calc_label.setText(text)
            self._style_step_calc(level)
            self._trainer.set_total_steps(params["total_steps"])
            self._rp(kind="progress", step=0, total=params["total_steps"])
        else:
            self._training_params = {}
            self._step_calc_label.setText("No images loaded")
            self._style_step_calc("ok")
            self._rp(kind="reset")
        self._update_sample_schedule()

    def _style_step_calc(self, level: str):
        """Color the step-calc readout by exposure safety: ok (gold) / warn (amber) / over (red)."""
        color, bg, border = {
            "ok": ("#f4d160", "#161208", "#3a3a1f"),
            "warn": ("#e0a93c", "#241a0a", "#6e5320"),
            "over": ("#ff6b5c", "#2a1210", "#7a2a22"),
        }.get(level, ("#f4d160", "#161208", "#3a3a1f"))
        self._step_calc_label.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: 600; "
            f"padding: 6px; background-color: {bg}; "
            f"border: 1px solid {border}; border-radius: 5px;"
        )

    def _generate_config(self):
        valid, msg = self._validate_for_config()
        if not valid:
            QMessageBox.warning(self, "Cannot Generate Config", msg)
            return

        self._ensure_sample_prompts()
        params = self._training_params
        lora_name = self._lora_name_edit.text().strip()
        run_dir = self._run_dir()  # per-run folder: {output_dir}/{lora_name}
        extra = {}
        if self._app_settings:
            extra.update(self._app_settings.build_extra_training_args())
            extra.update(self._app_settings.prepare_sample_args(
                run_dir, lora_name, self._trigger_word, self._dataset_style_anchor()))
        # Low-VRAM overrides (only present when the user enabled+acknowledged it this session).
        from core import lowvram
        lv = lowvram.get_current() or {}
        try:
            main_cfg, dataset_cfg = generate_configs(
                output_dir=run_dir,
                micro_batch=lv.get("micro_batch"),
                grad_accum=lv.get("grad_accum"),
                blocks_to_swap=lv.get("blocks_to_swap"),
                fp8_base=bool(lv.get("fp8_base")),
                lora_name=lora_name,
                dit_path=self._dit_path,
                qwen3_path=self._qwen3_path,
                vae_path=self._vae_path,
                dataset_folder=self._dataset_path,
                epochs=params["epochs"],
                repeats=params["repeats"],
                optimizer=self._current_optimizer(),
                learning_rate=self._lr_spin.value(),
                network_dim=self._dim_spin.value(),
                network_alpha=self._alpha_spin.value(),
                train_text_encoder=self._train_text_encoder,
                enable_bucket=self._bucket_check.isChecked(),
                save_state=self._save_state_check.isChecked(),
                save_every_n_steps=self._ckpt_steps_spin.value(),
                resume_state_path=self._resume_state_path if self._resume_check.isChecked() else None,
                network_weights=(self._network_weights_edit.text().strip() or None),
                training_comment=(self._trigger_word if self._metadata_check.isChecked() and self._trigger_word else None),
                extra_args=extra,
            )
            self._config_path = main_cfg
            self._dataset_config_path = dataset_cfg
            self._config_path_label.setText(
                f"✔ Config: {main_cfg}\n✔ Dataset: {dataset_cfg}"
            )
            self._config_path_label.setStyleSheet(
                "font-size: 11px; color: #d4af37;"
            )
            self.status_message.emit(f"Config generated: {main_cfg}")
        except Exception as e:
            QMessageBox.critical(self, "Config Error", str(e))

    def _preview_config(self):
        """Generate the exact TOMLs training will use and show them read-only.

        No parallel rendering path: this previews the same files _start_training
        regenerates at launch, so what you see is what the trainer reads.
        """
        self._generate_config()
        if not self._config_path:
            return  # _generate_config already surfaced the validation/config error
        from ui.forge_modal import ForgeModal
        parts = []
        for path in (self._config_path, self._dataset_config_path):
            if not path:
                continue
            try:
                body = Path(path).read_text(encoding="utf-8")
            except OSError as e:
                body = f"(could not read: {e})"
            parts.append(f"# ══ {path}\n\n{body}")
        viewer = QTextEdit()
        viewer.setReadOnly(True)
        viewer.setPlainText("\n\n".join(parts))
        viewer.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px; color: #c9c9ce; "
            "background-color: #0c0b0a; border: 1px solid #2a2a1e; border-radius: 5px;")
        viewer.setMinimumHeight(420)
        modal = ForgeModal(
            self.window(), title="Config Preview", eyebrow="Step 02 · Inspect",
            subtitle="The exact files the trainer reads — regenerated from the current settings.",
            max_width=760)
        modal.body.addWidget(viewer)
        modal.add_footer_button("Close", primary=True).clicked.connect(modal.close_modal)
        modal.open()

    def _start_training(self, confirm: bool = True):
        valid, msg = self._validate_for_training()
        if not valid:
            QMessageBox.warning(self, "Cannot Start Training", msg)
            return

        if not self._check_empty_captions(interactive=confirm):
            return

        if confirm and not self._confirm_preflight():
            return

        # Always (re)generate the config at launch so the current UI state is
        # honored — in particular the resume checkbox. Reusing a stale config from
        # a previous Start would silently ignore a resume request and restart from
        # zero (the exact bug this fixes).
        self._config_path = None
        self._generate_config()
        if not self._config_path or not Path(self._config_path).exists():
            return

        self._log_edit.clear()
        params = self._training_params
        total = params.get("total_steps", 3000)
        self._rp(kind="progress", step=0, total=total)
        self._trainer.set_total_steps(total)

        # Pre-launch VRAM guard (best-effort; never blocks if the probe is unavailable).
        # Skipped when Low-VRAM mode is active — fitting on less VRAM is the whole point.
        from core import gpu_check, lowvram
        free = gpu_check.free_vram_mb()
        if lowvram.get_current() is None and self._app_settings is not None and free is not None:
            need = self._app_settings.get("min_free_vram_mb")
            if need and free < need:
                apps = gpu_check.resident_gpu_apps()
                detail = (" Detected: " + ", ".join(apps) + ".") if apps else ""
                reply = QMessageBox.warning(
                    self, "Low GPU Memory",
                    f"Only {free / 1024:.1f} GB VRAM free (need ~{need / 1024:.1f} GB)."
                    f"{detail}\n\nTraining may run out of memory. Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

        # Record the active run so a crash can be recovered on next launch
        from core import sets
        rd, _ = self.build_run_definition()
        if rd is not None:
            sets.mark_run_active(rd)

        self._trainer.start(self._config_path, self._setup_path)

    def _check_empty_captions(self, interactive: bool = True) -> bool:
        """Guard against empty/missing .txt captions (training fails on them).

        Interactive: offer a one-click trigger-word fill (trigger-only training),
        continue-anyway, or cancel. Non-interactive (unattended pipeline): auto-fill
        with the trigger when one is set, otherwise proceed as before. Returns True
        to proceed with the run.
        """
        from core import dataset_manager
        empty = dataset_manager.find_empty_captions(self._dataset_path)
        if not empty:
            return True
        trigger = (self._trigger_word or "").strip()
        if not interactive:
            if trigger:
                n = dataset_manager.fill_empty_captions(self._dataset_path, trigger)
                self.status_message.emit(
                    f"Filled {n} empty caption(s) with the trigger word “{trigger}”.")
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Empty Captions")
        box.setText(
            f"{len(empty)} of {self._image_count} images have empty or missing captions.")
        info = "Training fails on captionless images."
        if trigger:
            info += (f"\n\nFill the empty ones with the trigger word “{trigger}” "
                     "(trigger-only training), continue as-is, or cancel and caption first.")
        else:
            info += ("\n\nNo trigger word is set, so one-click fill isn't available — "
                     "run captioning (or set a trigger word), or continue at your own risk.")
        box.setInformativeText(info)
        fill_btn = None
        if trigger:
            fill_btn = box.addButton("Fill with trigger word", QMessageBox.AcceptRole)
        box.addButton("Continue anyway", QMessageBox.DestructiveRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(fill_btn or cancel_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is cancel_btn:
            return False
        if fill_btn is not None and clicked is fill_btn:
            n = dataset_manager.fill_empty_captions(self._dataset_path, trigger)
            self.status_message.emit(
                f"Filled {n} empty caption(s) with the trigger word “{trigger}”.")
        return True

    def _confirm_preflight(self) -> bool:
        """Show a pre-flight summary before a (potentially long) run. Returns True to proceed.

        The exact TOMLs the trainer will read ride along under "Show Details…" so the
        recipe can be checked at the last moment (user feedback: the config preview is
        most useful right before launch). Same generation path as Start — no parallel
        rendering.
        """
        from pathlib import Path
        name = self._lora_name_edit.text().strip()
        run_dir = self._run_dir()
        total = self._training_params.get("total_steps", 0)
        resuming = self._resume_check.isChecked() and self._resume_state_path
        if resuming:
            mode = f"↻ RESUME from {Path(self._resume_state_path).name}"
        else:
            mode = "✦ Fresh run"
        previews = f"{Path(run_dir) / 'sample'}"
        if self._app_settings is not None:
            if self._app_settings.get("sample_enable"):
                every = self._app_settings.get("sample_every_n_epochs")
                previews += "  (every epoch)" if every <= 1 else f"  (every {every} epochs)"
            else:
                previews += "  (disabled)"
        info = (
            f"• {self._image_count} images · {total} steps\n"
            f"• {mode}\n"
            f"• LoRA → {Path(run_dir) / (name + '.safetensors')}\n"
            f"• Previews → {previews}\n"
            f"• Exact config files → “Show Details…” below"
        )
        box = QMessageBox(self)
        # the brand badge instead of the stock "?" — launch is the ceremony moment
        from PySide6.QtGui import QPixmap
        from utils.styles import asset_url
        _badge = QPixmap(asset_url("emblem.png"))
        if not _badge.isNull():
            box.setIconPixmap(_badge.scaledToHeight(64, Qt.SmoothTransformation))
        else:
            box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Start Training")
        box.setText(f"Start run “{name}”?")
        box.setInformativeText(info)
        box.setDetailedText(self._config_preview_text())
        start_btn = box.addButton("▶  Start", QMessageBox.AcceptRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(start_btn)
        box.exec()
        return box.clickedButton() is start_btn

    def _config_preview_text(self) -> str:
        """Generate the real config TOMLs and return their text (for Show Details…).

        Empty string on any failure — the confirm dialog then simply has no details
        pane; Start regenerates and surfaces errors on its own path.
        """
        try:
            self._generate_config()
        except Exception:
            return ""
        parts = []
        for path in (self._config_path, self._dataset_config_path):
            if not path:
                continue
            try:
                parts.append(f"══ {path}\n\n{Path(path).read_text(encoding='utf-8')}")
            except OSError:
                continue
        return "\n\n".join(parts)

    def _stop_training(self):
        reply = QMessageBox.question(
            self,
            "Stop Training",
            "Are you sure you want to stop the training process?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._trainer.stop()
            self._log_edit.append("\n[AnimaForge] Training stopped by user.")
            self._on_training_finished(False)

    # ------------------------------------------------------------------
    # Live sample preview
    # ------------------------------------------------------------------

    def _poll_sample_dir(self):
        if not self._output_dir:
            return
        from pathlib import Path
        from core.dataset_manager import latest_files
        files = latest_files(str(Path(self._run_dir()) / "sample"), None)
        if files == self._last_preview:
            return
        self._last_preview = files
        self._render_preview(files)

    def _render_preview(self, files):
        from PySide6.QtGui import QPixmap
        self._preview_files = files
        # Clear current preview widgets (keep the hint widget alive for reuse).
        while self._preview_grid.count():
            item = self._preview_grid.takeAt(0)
            w = item.widget()
            if w is not None and w is not self._preview_hint:
                w.deleteLater()
        if not files:
            self._preview_grid.addWidget(self._preview_hint, 0, 0, 1, 4)
            self._preview_hint.show()
            return
        self._preview_hint.hide()
        # Size the four columns to the viewport so a row of 4 fits exactly — never overflowing
        # off the right. Newest first (top-left); the scroll shows exactly two rows (current
        # epoch + the one before) and older sets require scrolling (no peek of the 3rd row).
        gap, cols = 8, 4
        vw = self._preview_scroll.viewport().width()
        col_w = max(80, (vw - (cols - 1) * gap - 4) // cols)

        def _add_thumb(path, row, col):
            pm = QPixmap(path)
            if pm.isNull():
                return False
            scaled = pm.scaled(col_w, col_w, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            thumb = _ClickableThumb(path)
            thumb.setPixmap(scaled)
            thumb.setFixedSize(scaled.size())
            thumb.setToolTip(path)
            thumb.clicked.connect(self._show_image)
            self._preview_grid.addWidget(thumb, row, col)
            return True

        if self._compare_toggle.isChecked():
            # Checkpoint comparison: one labeled band per epoch/step round, newest on
            # top — scan down a column to watch one prompt evolve and pick the
            # earliest epoch that already looks right.
            from core.samples import group_by_round
            row = 0
            for label, group in group_by_round(files):
                band = QLabel(label.upper())
                band.setStyleSheet("color: #f4d160; font-size: 11px; font-weight: 700; "
                                   "letter-spacing: 1px; padding-top: 6px;")
                self._preview_grid.addWidget(band, row, 0, 1, cols)
                row += 1
                shown = 0
                for path in group:
                    if _add_thumb(path, row + shown // cols, shown % cols):
                        shown += 1
                row += max(1, (shown + cols - 1) // cols)
        else:
            shown = 0
            for path in files:
                if _add_thumb(path, shown // cols, shown % cols):
                    shown += 1
        self._preview_scroll.setFixedHeight(2 * col_w + gap + 4)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, "_preview_files", None):
            self._render_preview(self._preview_files)

    def _show_image(self, path: str):
        from PySide6.QtWidgets import QDialog, QScrollArea
        from PySide6.QtGui import QPixmap
        dlg = QDialog(self)
        dlg.setWindowTitle(path)
        dlg.setMinimumSize(560, 560)
        lay = QVBoxLayout(dlg)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        lbl = QLabel()
        lbl.setPixmap(QPixmap(path))
        lbl.setAlignment(Qt.AlignCenter)
        sa.setWidget(lbl)
        lay.addWidget(sa)
        dlg.exec()

    # ------------------------------------------------------------------
    # Forge pipe
    # ------------------------------------------------------------------

    def _open_lowvram(self):
        from ui.lowvram_dialog import LowVramDialog
        LowVramDialog(self).exec()
        self._refresh_lowvram_indicator()

    def _refresh_lowvram_indicator(self):
        from core import lowvram
        cur = lowvram.get_current()
        if cur:
            fp8 = " + fp8" if cur.get("fp8_base") else ""
            self._lowvram_indicator.setText(
                f"⚠ Low-VRAM active — micro-batch {cur.get('micro_batch')} × "
                f"accum {cur.get('grad_accum')}, swap {cur.get('blocks_to_swap')} blocks{fp8}. "
                f"Slower, same quality. Resets when you close the app.")
            self._lowvram_indicator.setVisible(True)
        else:
            self._lowvram_indicator.setVisible(False)

    def _run_dir(self) -> str:
        """Per-run output folder ({output_dir}/{lora_name}); isolates each run's
        LoRA, sample previews, logs and state. Falls back to the base when unset."""
        from core.paths import run_output_dir
        return run_output_dir(self._output_dir, self._lora_name_edit.text().strip())

    def _lora_output_path(self):
        from pathlib import Path
        name = self._lora_name_edit.text().strip()
        if not name or not self._output_dir:
            return None
        return Path(self._run_dir()) / f"{name}.safetensors"

    def _deliver_to_forge(self, silent: bool = False):
        if not self._app_settings:
            return False
        a = self._app_settings
        lora_dir = a.get("forge_lora_dir")
        src = self._lora_output_path()
        if not lora_dir:
            if not silent:
                QMessageBox.warning(self, "No Forge folder", "Set the Forge LoRA folder in the Setup tab.")
            return False
        if not src or not src.is_file():
            if not silent:
                QMessageBox.warning(self, "No LoRA", f"{src} not found — train it first.")
            return False
        from core import forge_api
        from core.paths import delivery_filename
        try:
            out = forge_api.deliver_lora(
                str(src), lora_dir, a.get("forge_api_url"),
                dest_name=delivery_filename(self.get_lora_name(), self._trigger_word))
            self._on_log_line(f"[Forge] Delivered LoRA → {out}")
            if not silent:
                QMessageBox.information(self, "Delivered to Forge", f"Copied to:\n{out}")
            return True
        except Exception as e:
            self._on_log_line(f"[Forge] Deliver failed: {e}")
            if not silent:
                QMessageBox.warning(self, "Deliver Failed", str(e))
            return False

    def _deliver_to_comfyui(self, silent: bool = False):
        """Copy the trained LoRA into the user's ComfyUI loras folder.

        Plain copy only — ComfyUI workflows vary too much for a test-render API
        (user feedback: the copy alone is the useful part).
        """
        if not self._app_settings:
            return False
        lora_dir = self._app_settings.get("comfyui_lora_dir")
        src = self._lora_output_path()
        if not lora_dir:
            if not silent:
                QMessageBox.warning(self, "No ComfyUI folder",
                                    "Set the ComfyUI LoRA folder in the Setup tab first.")
            return False
        if not src or not src.is_file():
            if not silent:
                QMessageBox.warning(self, "No LoRA", f"{src} not found — train it first.")
            return False
        from core import forge_api
        from core.paths import delivery_filename
        try:
            out = forge_api.deliver_lora(
                str(src), lora_dir,  # no API refresh for Comfy
                dest_name=delivery_filename(self.get_lora_name(), self._trigger_word))
            self._on_log_line(f"[ComfyUI] Delivered LoRA → {out}")
            if not silent:
                QMessageBox.information(self, "Delivered to ComfyUI", f"Copied to:\n{out}")
            return True
        except Exception as e:
            self._on_log_line(f"[ComfyUI] Deliver failed: {e}")
            if not silent:
                QMessageBox.warning(self, "Deliver Failed", str(e))
            return False

    def _test_in_forge(self, save_dir=None):
        if not self._app_settings:
            return
        from PySide6.QtCore import QThread
        from core.forge_worker import ForgeRenderWorker
        from core.paths import delivery_filename
        a = self._app_settings
        # The test prompt's <lora:...> tag must match the DELIVERED filename, which
        # carries the trigger suffix.
        name = delivery_filename(self._lora_name_edit.text().strip(),
                                 self._trigger_word)[:-len(".safetensors")]
        prompts = [p.strip() for p in a.get("sample_prompts").splitlines() if p.strip()]
        if not prompts:
            prompts = ["upper body portrait, looking at viewer"]
        self._forge_images = []
        self._forge_save_dir = save_dir
        self._forge_thread = QThread(self)
        self._forge_worker = ForgeRenderWorker(a.get("forge_api_url"), name, self._trigger_word, prompts)
        self._forge_worker.moveToThread(self._forge_thread)
        self._forge_thread.started.connect(self._forge_worker.run)
        self._forge_worker.log_line.connect(self._on_log_line)
        self._forge_worker.image_ready.connect(self._on_forge_image)
        self._forge_worker.finished.connect(self._on_forge_test_done)
        self._forge_thread.start()
        self.status_message.emit("Forge test-render started…")

    def _on_forge_image(self, png_bytes: bytes, idx: int):
        self._forge_images.append(png_bytes)
        if self._forge_save_dir:
            from pathlib import Path
            d = Path(self._forge_save_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"test_{len(self._forge_images):02d}.png").write_bytes(png_bytes)

    def _on_forge_test_done(self, ok: bool):
        self._forge_thread.quit()
        self._forge_thread.wait()
        if self._forge_save_dir:
            self._on_log_line(f"[Forge] Saved {len(self._forge_images)} test image(s) to {self._forge_save_dir}")
            return
        if self._forge_images:
            self._show_forge_results(self._forge_images)
        elif not ok:
            QMessageBox.warning(self, "Test Failed",
                                "No images returned. Is Forge running with --api and the LoRA delivered?")

    def _show_forge_results(self, images):
        from PySide6.QtWidgets import QDialog, QScrollArea, QWidget
        from PySide6.QtGui import QPixmap, QImage
        dlg = QDialog(self)
        dlg.setWindowTitle("Forge Test Render")
        dlg.setMinimumSize(540, 640)
        lay = QVBoxLayout(dlg)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        lay.addWidget(sa)
        cont = QWidget()
        cl = QVBoxLayout(cont)
        sa.setWidget(cont)
        for b in images:
            img = QImage.fromData(b)
            pm = QPixmap.fromImage(img).scaledToWidth(480, Qt.SmoothTransformation)
            lbl = QLabel()
            lbl.setPixmap(pm)
            cl.addWidget(lbl)
        dlg.exec()

    @Slot()
    def _on_training_started(self):
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self.training_active.emit(True)
        self._rp(kind="phase", label="Preparing…")
        self._stepping = False
        self._denoiser = LogDenoiser()
        self.status_message.emit("Training started…")
        self._last_preview = []
        self._preview_timer.start()

    @Slot(bool)
    def _on_training_finished(self, success: bool):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self.training_active.emit(False)
        self._preview_timer.stop()
        self._poll_sample_dir()  # catch the final sample set
        if success:
            from core import sets
            sets.clear_active_run()
            self._rp(kind="done", label="Complete!")
            self.status_message.emit("Training finished successfully.")
            if self._app_settings:
                if self._app_settings.get("forge_auto_deliver"):
                    self._deliver_to_forge(silent=True)
                if self._app_settings.get("forge_auto_test"):
                    from pathlib import Path
                    self._test_in_forge(save_dir=str(Path(self._run_dir()) / "forge_test"))
        else:
            self._rp(kind="error", label="Stopped")
            # A state was just written before the process exited. Surface it and
            # arm resume so the next Start continues from the last checkpoint
            # instead of restarting from zero (uncheck the box to start fresh).
            self._refresh_resume_option()
            if self._resume_state_path:
                self._resume_check.setChecked(True)
                from pathlib import Path as _P
                self.status_message.emit(
                    f"Stopped. Start will RESUME from {_P(self._resume_state_path).name} "
                    f"— uncheck 'Resume from last saved state' to start fresh."
                )
            else:
                self.status_message.emit("Training stopped or failed.")

    @Slot(str)
    def _on_log_line(self, line: str):
        shown = self._denoiser.filter(line)
        if shown is not None:
            self._log_edit.append(shown)
            sb = self._log_edit.verticalScrollBar()
            sb.setValue(sb.maximum())
        self._update_dials(line)
        if not self._stepping:
            phase = phase_for_line(line)
            if phase:
                self._rp(kind="phase", label=phase)

    def _update_dials(self, line: str):
        """Drive the Epoch/Loss/Speed/ETA dials from a training log line."""
        m = self._EPOCH_INC.search(line) if hasattr(self, "_EPOCH_INC") else None
        if m is None:
            m = _re.search(r"current_epoch:\s*(\d+)", line)
        if m:
            total_epochs = int(self._training_params.get("epochs", 0)) if \
                getattr(self, "_training_params", None) else 0
            # sd-scripts logs current_epoch 0-based at the START of each epoch. The dial shows
            # the COMPLETED epoch (whose samples we're viewing): 0 while epoch 1 trains, and it
            # advances to N only once epoch N has finished. Never used to auto-stop.
            self._dials.set_epoch(int(m.group(1)), total_epochs)
        # Only the main training bar (desc="steps") feeds the dials — the "Sampling"
        # bar during preview renders and the caching/loading bars are also tqdm and
        # would otherwise swing Speed/ETA to nonsense between steps.
        if not line.lstrip().lower().startswith("steps"):
            return
        metrics = parse_tqdm(line)
        if "loss" in metrics:
            self._dials.set_loss(metrics["loss"])
        if "it_s" in metrics:
            self._dials.set_speed(metrics["it_s"])
        if "eta" in metrics:
            self._dials.set_eta(metrics["eta"], metrics.get("elapsed", 0))

    @Slot(int)
    def _on_progress_updated(self, step: int):
        total = self._trainer.total_steps
        self._tickbar.set_progress(step)
        if step >= 1 and not self._stepping:
            self._stepping = True
            self._rp(kind="phase", label="Training")
        self._rp(kind="progress", step=step, total=total)
        # step < 1 (the initial 0/total parse): leave the current phase label in place

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_for_config(self) -> tuple:
        """Validate enough to generate config files."""
        if not self._setup_path:
            return False, "sd-scripts path is not set. Go to Setup tab."
        if not Path(self._setup_path).is_dir():
            return False, "sd-scripts path does not exist."
        for label, p in (
            ("Anima DiT checkpoint", self._dit_path),
            ("Qwen3 text encoder", self._qwen3_path),
            ("Qwen-Image VAE", self._vae_path),
        ):
            if not p or not Path(p).exists():
                return False, f"{label} is not set or does not exist. Go to Setup tab."
        if not self._output_dir:
            return False, "Output directory is not set. Go to Setup tab."
        if not self._dataset_path or not Path(self._dataset_path).is_dir():
            return False, "Dataset folder is not set or does not exist."
        if self._image_count == 0:
            return False, "Dataset folder contains no supported images."
        lora_name = self._lora_name_edit.text().strip()
        if not lora_name:
            return False, "LoRA name cannot be empty."
        if not self._training_params:
            return False, "Step calculation has not run. Ensure dataset is loaded."
        nw = self._network_weights_edit.text().strip()
        if nw and not Path(nw).is_file():
            return False, "The 'Start from existing LoRA' file does not exist."
        if self._resume_check.isChecked() and not (
            self._resume_state_path and Path(self._resume_state_path).is_dir()
        ):
            return False, "Resume is checked but no saved state was found."
        return True, ""

    def _validate_for_training(self) -> tuple:
        valid, msg = self._validate_for_config()
        if not valid:
            return False, msg
        if not (Path(self._setup_path) / "anima_train_network.py").is_file():
            return False, (
                "anima_train_network.py not found in sd-scripts path.\n"
                "Update sd-scripts to a version with Anima support (Setup tab)."
            )
        return True, ""
