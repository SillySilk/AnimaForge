import os
import re
from pathlib import Path

from PySide6.QtCore import QSettings, QSize, Qt, Signal, QTimer, QProcess, QProcessEnvironment
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFontComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.caption_rules import dump_rules, parse_rules
from core.describe_models import (
    CUSTOM_INDEX, DESCRIBE_MODELS, entry_for_repo, model_is_present,
    read_describe_defaults, valid_repo_id,
)
from core.env import subprocess_python
from core.gpu_check import free_vram_mb
from core.trainer import (
    clear_windows_user_env_libuv,
    libuv_status,
    windows_system_env_libuv,
    windows_user_env_libuv,
)
from core.settings import AppSettings
from ui.forge_modal import ForgeModal
from utils.proc import apply_no_window

# Shared QSettings ids (single source of truth in core.settings).
from core.settings import SETTINGS_ORG, SETTINGS_APP  # noqa: E402

# Canonical filenames, for the setup instructions only. Auto-detect no longer keys
# off them — core.model_locations identifies each file by its safetensors header,
# so renamed files and non-Forge folder layouts are found too.
QWEN3_FILENAME = "qwen_3_06b_base.safetensors"
QWEN_VAE_FILENAME = "qwen_image_vae.safetensors"
PREFERRED_DIT = "anima-base-v1.0.safetensors"


# Minimum PyTorch the trainer needs.
MIN_TORCH = (2, 5)


def parse_torch_probe(text: str):
    """(major, minor) from a `torch X.Y.Z cuda A.B` probe line, or None when absent.

    None means "the probe told us nothing" (interpreter missing, import error, a
    traceback) — deliberately distinct from "an old torch", so a failed probe never
    erases a previously detected install.
    """
    m = re.search(r"torch\s+(\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def torch_upgrade_plan(rtx50: bool) -> tuple:
    """(pip requirement, wheel index, human label) for the PyTorch upgrade.

    RTX 50-series (Blackwell, sm_120) needs cu128 wheels (torch >= 2.7); everything
    else stays on cu121 — older cards (e.g. Pascal) aren't covered by cu128 builds.
    """
    if rtx50:
        return ("torch>=2.7", "https://download.pytorch.org/whl/cu128",
                "CUDA 12.8 — RTX 50-series")
    return ("torch>=2.5", "https://download.pytorch.org/whl/cu121", "CUDA 12.1")


class SetupTab(QWidget):
    settings_changed = Signal()
    describe_model_changed = Signal()
    pytorch_status_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(1000)
        self._save_timer.timeout.connect(self._save_settings)
        self._torch_process = None
        # Restored from the last successful probe, not reset to False: Home reads this
        # for its health row, and starting cold made PyTorch look missing on every
        # launch until the user clicked Check PyTorch Version again.
        self._pytorch_ok = self._cached_pytorch_ok()
        # Set by _reset_training_environment; consumed exactly once by the next
        # _handle_torch_check_output so a Reset only clears the cached known-good
        # torch_version once the re-probe actually comes back and fails to parse —
        # never preemptively (see both methods below).
        self._reset_pending = False
        self._app = AppSettings()
        self._build_ui()
        self._load_settings()
        self._bind_app_widgets()
        self._refresh_libuv_status()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_sdscripts_path(self) -> str:
        return self._sdscripts_edit.text().strip()

    def get_dit_path(self) -> str:
        return self._dit_edit.text().strip()

    def get_qwen3_path(self) -> str:
        return self._qwen3_edit.text().strip()

    def get_vae_path(self) -> str:
        return self._vae_edit.text().strip()

    def get_moss_dit_path(self) -> str:
        return self._moss_dit_edit.text().strip()

    def get_moss_te_path(self) -> str:
        return self._moss_te_edit.text().strip()

    def get_moss_vae_path(self) -> str:
        return self._moss_vae_edit.text().strip()

    def model_paths(self, model: str) -> tuple:
        """(dit, text encoder, vae) paths for `model`."""
        if model == "moss":
            return (self.get_moss_dit_path(), self.get_moss_te_path(), self.get_moss_vae_path())
        if model == "anima":
            return (self.get_dit_path(), self.get_qwen3_path(), self.get_vae_path())
        raise ValueError(f"unknown model {model!r}")

    def get_output_dir(self) -> str:
        return self._output_edit.text().strip()

    def is_environment_valid(self, model: str = "anima") -> tuple:
        """Returns (is_valid: bool, message: str)."""
        sd = self.get_sdscripts_path()
        out = self.get_output_dir()

        if not sd:
            return False, "sd-scripts path is not set."
        if not Path(sd).is_dir():
            return False, "sd-scripts path does not exist."
        from core.kohya_recipes import recipe
        script = recipe(model).train_script
        if not (Path(sd) / script).is_file():
            hint = "update sd-scripts" if model == "anima" else "re-run install.bat"
            return False, f"sd-scripts path does not contain {script} ({hint})."

        if model == "moss":
            from core import model_check
            dit, te, vae = self.model_paths("moss")
            checks = (
                ("MOSS-Image DiT", dit, model_check.ROLE_MOSS_DIT),
                ("MOSS text encoder (Qwen3.5-9B)", te, model_check.ROLE_MOSS_TE),
                ("MOSS VAE (FLUX.2)", vae, model_check.ROLE_FLUX2_VAE),
            )
        else:
            checks = (
                ("Anima DiT checkpoint", self.get_dit_path(), None),
                ("Qwen3 text encoder", self.get_qwen3_path(), None),
                ("Qwen-Image VAE", self.get_vae_path(), None),
            )
        for label, path, role in checks:
            if not path:
                return False, f"{label} is not set."
            if not Path(path).is_file():
                return False, f"{label} file does not exist."
            if not path.lower().endswith((".safetensors", ".pth", ".ckpt")):
                return False, f"{label} must be a .safetensors/.pth file."
            if role:
                problem = model_check.check_model_file(path, role)
                if problem:
                    return False, f"{label}: {problem}"

        if not out:
            return False, "Output directory is not set."
        try:
            Path(out).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return False, f"Cannot create output directory: {e}"
        if not os.access(out, os.W_OK):
            return False, "Output directory is not writable."
        return True, "Environment OK"

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(20)

        title = QLabel("Environment Setup")
        title.setObjectName("label_section")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #d4af37;")
        layout.addWidget(title)

        # sd-scripts group
        sd_group = QGroupBox("Kohya sd-scripts")
        sd_layout = QVBoxLayout(sd_group)
        sd_layout.setSpacing(8)
        sd_hint = QLabel("Path to your local Kohya sd-scripts folder (must include anima_train_network.py).")
        sd_hint.setObjectName("label_field")
        sd_hint.setWordWrap(True)
        sd_layout.addWidget(sd_hint)
        sd_row = QHBoxLayout()
        self._sdscripts_edit = QLineEdit()
        self._sdscripts_edit.setPlaceholderText("e.g. .../Lora Forge/sd-scripts")
        self._sdscripts_status = self._make_status_dot()
        sd_browse = QPushButton("Browse…")
        sd_browse.setFixedWidth(108)
        sd_browse.clicked.connect(self._browse_sdscripts)
        sd_row.addWidget(self._sdscripts_edit)
        sd_row.addWidget(self._sdscripts_status)
        sd_row.addWidget(sd_browse)
        sd_layout.addLayout(sd_row)
        layout.addWidget(sd_group)

        # Anima model files group (three files)
        model_group = QGroupBox("Anima Model Files")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(8)

        model_hint = QLabel(
            "Anima needs three files: the DiT checkpoint, the Qwen3 text encoder, and the "
            "Qwen-Image VAE. Use auto-detect to pull them from your Forge Neo install."
        )
        model_hint.setObjectName("label_field")
        model_hint.setWordWrap(True)
        model_layout.addWidget(model_hint)

        self._dit_edit, dit_row = self._make_file_row(
            "e.g. .../Stable-diffusion/anima_baseV10.safetensors", self._browse_dit
        )
        self._dit_status = dit_row.itemAt(1).widget()
        model_layout.addWidget(QLabel("Anima DiT checkpoint:"))
        model_layout.addLayout(dit_row)

        self._qwen3_edit, q_row = self._make_file_row(
            "e.g. .../text_encoder/qwen_3_06b_base.safetensors", self._browse_qwen3
        )
        self._qwen3_status = q_row.itemAt(1).widget()
        model_layout.addWidget(QLabel("Qwen3 text encoder:"))
        model_layout.addLayout(q_row)

        self._vae_edit, v_row = self._make_file_row(
            "e.g. .../VAE/qwen_image_vae.safetensors", self._browse_vae
        )
        self._vae_status = v_row.itemAt(1).widget()
        model_layout.addWidget(QLabel("Qwen-Image VAE:"))
        model_layout.addLayout(v_row)

        scan_row = QHBoxLayout()
        scan_row.addWidget(QLabel("Model scan folder:"))
        self._scan_edit = QLineEdit()
        self._scan_edit.setPlaceholderText("folder auto-detect scans for Anima models (e.g. your Forge models dir)")
        scan_row.addWidget(self._scan_edit)
        scan_browse = QPushButton("Browse…")
        scan_browse.setFixedWidth(108)
        scan_browse.clicked.connect(self._browse_scan)
        scan_row.addWidget(scan_browse)
        model_layout.addLayout(scan_row)

        autodetect_btn = QPushButton("🔍 Auto-detect models")
        autodetect_btn.clicked.connect(self._auto_detect_models)
        model_layout.addWidget(autodetect_btn)
        self._autodetect_label = QLabel("")
        self._autodetect_label.setWordWrap(True)
        self._autodetect_label.setStyleSheet("font-size: 11px; color: #8a8a93;")
        model_layout.addWidget(self._autodetect_label)
        layout.addWidget(model_group)

        # MOSS-Image model files group (three files)
        moss_group = QGroupBox("MOSS-Image Model Files")
        moss_layout = QVBoxLayout(moss_group)
        moss_layout.setSpacing(8)

        moss_hint = QLabel(
            "MOSS-Image needs the Base DiT, the Qwen3.5-9B text encoder (int8 build; the "
            "W4A8 build is not supported) and the FLUX.2 VAE. Auto-detect scans the model "
            "scan folder above."
        )
        moss_hint.setObjectName("label_field")
        moss_hint.setWordWrap(True)
        moss_layout.addWidget(moss_hint)

        self._moss_dit_edit, moss_dit_row = self._make_file_row(
            "e.g. .../diffusion_models/moss_image_base_bf16.safetensors", self._browse_moss_dit
        )
        self._moss_dit_status = moss_dit_row.itemAt(1).widget()
        moss_layout.addWidget(QLabel("MOSS-Image DiT:"))
        moss_layout.addLayout(moss_dit_row)

        self._moss_te_edit, moss_te_row = self._make_file_row(
            "e.g. .../text_encoder/qwen3.5_9b_int8_convrot.safetensors", self._browse_moss_te
        )
        self._moss_te_status = moss_te_row.itemAt(1).widget()
        moss_layout.addWidget(QLabel("MOSS text encoder (Qwen3.5-9B):"))
        moss_layout.addLayout(moss_te_row)

        self._moss_vae_edit, moss_vae_row = self._make_file_row(
            "e.g. .../VAE/flux2-vae.safetensors", self._browse_moss_vae
        )
        self._moss_vae_status = moss_vae_row.itemAt(1).widget()
        moss_layout.addWidget(QLabel("MOSS VAE (FLUX.2):"))
        moss_layout.addLayout(moss_vae_row)

        moss_detect_btn = QPushButton("🔍 Auto-detect MOSS files")
        moss_detect_btn.clicked.connect(self._auto_detect_moss)
        moss_layout.addWidget(moss_detect_btn)
        self._moss_detect_label = QLabel("")
        self._moss_detect_label.setWordWrap(True)
        self._moss_detect_label.setStyleSheet("font-size: 11px; color: #8a8a93;")
        moss_layout.addWidget(self._moss_detect_label)
        layout.addWidget(moss_group)

        # PyTorch group
        torch_group = QGroupBox("PyTorch Runtime (Anima requires 2.5+)")
        torch_layout = QVBoxLayout(torch_group)
        torch_layout.setSpacing(8)
        torch_hint = QLabel(
            "Anima training needs PyTorch 2.5 or newer (older versions produce NaN loss). "
            "Check the version in your training environment and upgrade if needed. "
            "RTX 50-series (Blackwell) cards additionally need the CUDA 12.8 build — "
            "Upgrade installs the right one automatically."
        )
        torch_hint.setObjectName("label_field")
        torch_hint.setWordWrap(True)
        torch_layout.addWidget(torch_hint)
        torch_btn_row = QHBoxLayout()
        check_torch_btn = QPushButton("Check PyTorch Version")
        check_torch_btn.clicked.connect(self._check_torch_version)
        self._upgrade_torch_btn = QPushButton("⬆ Upgrade PyTorch (auto-picks CUDA build)")
        self._upgrade_torch_btn.clicked.connect(self._upgrade_torch)
        torch_btn_row.addWidget(check_torch_btn)
        torch_btn_row.addWidget(self._upgrade_torch_btn)
        torch_btn_row.addStretch()
        torch_layout.addLayout(torch_btn_row)
        self._torch_log = QTextEdit()
        self._torch_log.setReadOnly(True)
        self._torch_log.setFixedHeight(90)
        self._torch_log.setObjectName("log_output")
        self._torch_log.setPlaceholderText("PyTorch version / upgrade output appears here…")
        torch_layout.addWidget(self._torch_log)
        layout.addWidget(torch_group)

        # Training Environment group — self-service USE_LIBUV controls. Lives next to
        # PyTorch Runtime above (same "set-once environment diagnostics" category as
        # Check/Upgrade PyTorch — not a per-run control, so it stays off Home).
        libuv_group = QGroupBox("Training Environment")
        libuv_layout = QVBoxLayout(libuv_group)
        libuv_layout.setSpacing(8)
        libuv_hint = QLabel(
            "Windows PyTorch builds without libuv can make training die at exit code 1 "
            "before the first step. Lora Forge sets USE_LIBUV=0 automatically — unless "
            "your own Windows environment already has a USE_LIBUV, which Lora Forge "
            "leaves alone by default. Use these if that's still causing crashes. "
            "(Runs are also pinned to a single-process launcher, which avoids the most "
            "common form of this crash on its own.)"
        )
        libuv_hint.setObjectName("label_field")
        libuv_hint.setWordWrap(True)
        libuv_layout.addWidget(libuv_hint)

        self._libuv_status_label = QLabel("")
        self._libuv_status_label.setWordWrap(True)
        self._libuv_status_label.setObjectName("label_field")
        libuv_layout.addWidget(self._libuv_status_label)

        self._force_libuv_check = QCheckBox(
            "Ignore my environment — always let Lora Forge set USE_LIBUV")
        libuv_layout.addWidget(self._force_libuv_check)

        libuv_btn_row = QHBoxLayout()
        self._clear_libuv_btn = QPushButton("Clear USE_LIBUV from Windows")
        self._clear_libuv_btn.clicked.connect(self._clear_libuv_from_windows)
        self._reset_env_btn = QPushButton("🔄 Reset Training Environment")
        self._reset_env_btn.clicked.connect(self._reset_training_environment)
        libuv_btn_row.addWidget(self._clear_libuv_btn)
        libuv_btn_row.addWidget(self._reset_env_btn)
        libuv_btn_row.addStretch()
        libuv_layout.addLayout(libuv_btn_row)
        layout.addWidget(libuv_group)

        # Output group
        output_group = QGroupBox("Output Directory")
        output_layout = QVBoxLayout(output_group)
        output_layout.setSpacing(8)
        output_hint = QLabel("Directory where trained LoRA files will be saved.")
        output_hint.setObjectName("label_field")
        output_hint.setWordWrap(True)
        output_layout.addWidget(output_hint)
        output_row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Select your LoRA output folder")
        self._output_status = self._make_status_dot()
        output_browse = QPushButton("Browse…")
        output_browse.setFixedWidth(108)
        output_browse.clicked.connect(self._browse_output)
        output_row.addWidget(self._output_edit)
        output_row.addWidget(self._output_status)
        output_row.addWidget(output_browse)
        output_layout.addLayout(output_row)
        layout.addWidget(output_group)

        # Krea 2 trains in an external OneTrainer install.
        layout.addWidget(self._build_onetrainer_group())

        # Which model writes the natural-language captions.
        layout.addWidget(self._build_describe_group())

        # The Bench — optional service.
        layout.addWidget(self._build_forge_group())

        # Fine Tuning — the heavy set-once config is popped into modals so the Workshop
        # reads as a readiness checklist, not a wall of fields.
        self._advanced_group = self._build_advanced_group()
        self._defaults_group = self._build_defaults_group()
        self._presets_group = self._build_presets_group()
        for grp in (self._advanced_group, self._defaults_group, self._presets_group):
            grp.setParent(self)
            grp.setVisible(False)
        fine_label = QLabel("Fine Tuning")
        fine_label.setObjectName("af_screen_eyebrow")
        layout.addWidget(fine_label)
        fine_row = QHBoxLayout()
        fine_row.setSpacing(10)
        for text, opener in [("⚙  App Defaults", self._open_defaults_modal),
                             ("⚙  Advanced Training", self._open_advanced_modal),
                             ("🎛  Training Presets", self._open_presets_settings_modal),
                             ("📖  Setup Guide", self._show_install_dialog)]:
            b = QPushButton(text)
            b.setObjectName("af_btn_ghost")
            b.setMinimumHeight(40)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(opener)
            fine_row.addWidget(b)
        layout.addLayout(fine_row)

        # Buttons row
        btn_row = QHBoxLayout()
        verify_btn = QPushButton("✔ Verify Environment")
        verify_btn.setObjectName("btn_primary")
        verify_btn.clicked.connect(self._verify_environment)
        install_btn = QPushButton("ℹ Setup Instructions")
        install_btn.clicked.connect(self._show_install_dialog)
        # No explicit Save button: settings auto-save (1s debounce) on every edit via
        # _on_text_changed → _save_timer → _save_settings. A second "save" was confusing
        # next to the Train tab's "Save Set".
        btn_row.addWidget(verify_btn)
        btn_row.addWidget(install_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._verify_label = QLabel("")
        self._verify_label.setWordWrap(True)
        layout.addWidget(self._verify_label)
        layout.addStretch()

        # Auto-save wiring
        for edit in (self._sdscripts_edit, self._dit_edit, self._qwen3_edit,
                     self._vae_edit, self._moss_dit_edit, self._moss_te_edit,
                     self._moss_vae_edit, self._output_edit):
            edit.textChanged.connect(self._on_text_changed)

    # ------------------------------------------------------------------
    # Krea 2 — the OneTrainer engine and the Krea 2 Raw weights
    # ------------------------------------------------------------------

    KREA_REPO = "krea/Krea-2-Raw"

    def _build_onetrainer_group(self) -> QGroupBox:
        grp = QGroupBox("Krea 2 — OneTrainer")
        col = QVBoxLayout(grp)
        col.setSpacing(8)
        hint = QLabel(
            "Krea 2 LoRAs train in a stock OneTrainer install, driven in the background. "
            "Point this at the folder holding its venv and scripts\\train.py — Lora Forge "
            "never modifies it. Not needed for Anima.")
        hint.setObjectName("label_field")
        hint.setWordWrap(True)
        col.addWidget(hint)
        row = QHBoxLayout()
        self._ot_edit = QLineEdit(self.get_app_settings().get("onetrainer_root") or "")
        self._ot_edit.setPlaceholderText("Locate your OneTrainer folder")
        self._ot_edit.textChanged.connect(self._on_ot_root_changed)
        self._ot_dot = self._make_status_dot()
        ot_browse = QPushButton("Browse…")
        ot_browse.setFixedWidth(108)
        ot_browse.clicked.connect(self._browse_onetrainer)
        row.addWidget(self._ot_edit)
        row.addWidget(self._ot_dot)
        row.addWidget(ot_browse)
        col.addLayout(row)
        self._krea_weights_label = QLabel("")
        self._krea_weights_label.setObjectName("label_field")
        self._krea_weights_label.setWordWrap(True)
        self._krea_weights_label.setOpenExternalLinks(True)
        col.addWidget(self._krea_weights_label)
        self._profiles_check = QCheckBox("Write OneTrainer profiler traces (for diagnosing slow runs)")
        self._profiles_check.setChecked(bool(self._app.get("debug_profiles")))
        self._profiles_check.toggled.connect(lambda b: self._app.set("debug_profiles", b))
        col.addWidget(self._profiles_check)
        self._revalidate_onetrainer()
        self._refresh_krea_weights()
        return grp

    def get_onetrainer_root(self) -> str:
        return self._ot_edit.text().strip()

    def _browse_onetrainer(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Locate the OneTrainer folder", self._ot_edit.text())
        if folder:
            self._ot_edit.setText(folder)

    def _on_ot_root_changed(self, text: str):
        self.get_app_settings().set("onetrainer_root", text.strip())
        self._revalidate_onetrainer()
        self.settings_changed.emit()

    def _revalidate_onetrainer(self):
        from core.engines.onetrainer import engine
        root = self._ot_edit.text().strip()
        ok, why = engine.validate(root)
        self._set_status(self._ot_dot, ok, bool(root))
        self._ot_dot.setToolTip("" if ok else why)

    def _refresh_krea_weights(self):
        from core import describe_models
        if describe_models.has_model_snapshot(describe_models.hf_default_cache_dir(),
                                              self.KREA_REPO):
            self._krea_weights_label.setText("Krea 2 Raw is downloaded.")
        else:
            self._krea_weights_label.setText(
                "Krea 2 Raw downloads on the first Krea 2 run (tens of GB). It is a gated "
                "model: accept its licence first at https://huggingface.co/krea/Krea-2-Raw "
                "while signed in to Hugging Face.")

    def _make_status_dot(self) -> QLabel:
        dot = QLabel("●")
        dot.setObjectName("label_status_unknown")
        dot.setFixedWidth(24)
        dot.setAlignment(Qt.AlignCenter)
        dot.setStyleSheet("color: #6a6a72; font-size: 18px;")
        return dot

    def _make_file_row(self, placeholder: str, browse_slot):
        """Return (line_edit, row_layout) with edit + status dot + browse button."""
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        dot = self._make_status_dot()
        browse = QPushButton("Browse…")
        browse.setFixedWidth(108)
        browse.clicked.connect(browse_slot)
        row.addWidget(edit)
        row.addWidget(dot)
        row.addWidget(browse)
        return edit, row

    # ------------------------------------------------------------------
    # Browse slots
    # ------------------------------------------------------------------

    def _browse_sdscripts(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select sd-scripts Folder", self._sdscripts_edit.text()
        )
        if folder:
            self._sdscripts_edit.setText(folder)

    def _browse_model_file(self, edit: QLineEdit, title: str):
        current = edit.text()
        start_dir = str(Path(current).parent) if current else (
            self._scan_edit.text().strip() or self._app.first_run_scan_default())
        path, _ = QFileDialog.getOpenFileName(
            self, title, start_dir, "Model Files (*.safetensors *.pth *.ckpt)"
        )
        if path:
            edit.setText(path)

    def _browse_dit(self):
        self._browse_model_file(self._dit_edit, "Select Anima DiT Checkpoint")

    def _browse_qwen3(self):
        self._browse_model_file(self._qwen3_edit, "Select Qwen3 Text Encoder")

    def _browse_vae(self):
        self._browse_model_file(self._vae_edit, "Select Qwen-Image VAE")

    def _browse_moss_dit(self):
        self._browse_model_file(self._moss_dit_edit, "Select MOSS-Image DiT")

    def _browse_moss_te(self):
        self._browse_model_file(self._moss_te_edit, "Select MOSS Text Encoder (Qwen3.5-9B)")

    def _browse_moss_vae(self):
        self._browse_model_file(self._moss_vae_edit, "Select MOSS VAE (FLUX.2)")

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self._output_edit.text()
        )
        if folder:
            self._output_edit.setText(folder)

    # ------------------------------------------------------------------
    # Auto-detect
    # ------------------------------------------------------------------

    def _auto_detect_models(self):
        """Fill the three model paths from whatever is under the scan folder.

        Detection is delegated to core.model_locations, which identifies each file
        by its safetensors header rather than by folder and filename. The old
        lookups here were hardcoded to the A1111/Forge tree with exact names, so
        the official circlestone-labs/Anima download — published in the ComfyUI
        tree — matched only the VAE, and any renamed file matched nothing.
        """
        from core.model_locations import detect_model_files

        scan_root = Path(self._scan_edit.text().strip() or self._app.first_run_scan_default())
        if not scan_root.is_dir():
            self._autodetect_label.setText("Model scan folder not found — set it above.")
            self._verify_environment()
            return

        got = detect_model_files(scan_root)
        found, missing = [], []
        for key, label, edit in (
            ("dit", "Anima DiT checkpoint", self._dit_edit),
            ("text_encoder", "Qwen3 encoder", self._qwen3_edit),
            ("vae", "Qwen-Image VAE", self._vae_edit),
        ):
            if got.get(key):
                edit.setText(got[key])
                found.append(f"{label} ({Path(got[key]).name})")
            else:
                missing.append(label)

        parts = []
        if found:
            parts.append("Found: " + ", ".join(found) + ".")
        if missing:
            parts.append(
                "Not found: " + ", ".join(missing) + ". Download them from "
                "huggingface.co/circlestone-labs/Anima (split_files/) into the scan "
                "folder, or set each one manually."
            )
        self._autodetect_label.setText(" ".join(parts))
        self._verify_environment()

    def _auto_detect_moss(self):
        """Fill the three MOSS-Image paths from the scan folder (header-based, like Anima's)."""
        from core.model_locations import detect_model_files

        scan_root = Path(self._scan_edit.text().strip() or self._app.first_run_scan_default())
        if not scan_root.is_dir():
            self._moss_detect_label.setText("Model scan folder not found — set it above.")
            self._verify_environment()
            return

        got = detect_model_files(scan_root, model="moss")
        found, missing = [], []
        for key, label, edit in (
            ("dit", "MOSS-Image DiT", self._moss_dit_edit),
            ("text_encoder", "MOSS text encoder (Qwen3.5-9B, int8)", self._moss_te_edit),
            ("vae", "MOSS VAE (FLUX.2)", self._moss_vae_edit),
        ):
            if got.get(key):
                edit.setText(got[key])
                found.append(f"{label} ({Path(got[key]).name})")
            else:
                missing.append(label)

        parts = []
        if found:
            parts.append("Found: " + ", ".join(found) + ".")
        if missing:
            parts.append(
                "Not found: " + ", ".join(missing) + ". Download them from "
                "huggingface.co/OpenMOSS-Team/MOSS-Image-Base into the scan folder, "
                "or set each one manually."
            )
        self._moss_detect_label.setText(" ".join(parts))
        self._verify_environment()

    # ------------------------------------------------------------------
    # PyTorch check / upgrade
    # ------------------------------------------------------------------

    def _venv_python(self) -> str:
        # Resolve exactly like training does (core/env.py): unified .venv installs run
        # everything from sys.executable; a legacy sd-scripts/venv is a fallback only.
        # The old hardcoded sd-scripts/venv path broke this button on unified installs
        # even though training worked fine.
        return subprocess_python(self.get_sdscripts_path())

    def _check_torch_version(self) -> bool:
        """Launch the async version probe. Returns whether a probe was actually
        started — False on either early-return path (no interpreter found, or a
        PyTorch operation already in flight) means `_handle_torch_check_output`
        will never be called for this invocation. Callers that need to know
        whether a re-probe is genuinely in flight (e.g. Reset's deferred cache
        clear) rely on this return value.
        """
        py = self._venv_python()
        if not py or not Path(py).is_file():
            self._torch_log.append("Could not find a training python interpreter. Check the install (.venv) or set the sd-scripts path.")
            return False
        if self._torch_process is not None and self._torch_process.state() != QProcess.NotRunning:
            self._torch_log.append("A PyTorch operation is already running.")
            return False
        self._torch_log.append("Checking PyTorch version…")
        proc = QProcess(self)
        apply_no_window(proc)  # no console window pop-up on Windows
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda p=proc: self._handle_torch_check_output(p))
        proc.start(py, ["-c", "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"])
        return True

    def start_background_torch_check(self):
        """Probe PyTorch once at app start so Home's health row is right without a click.

        Runs off the UI thread (QProcess), so the ~2s `import torch` never blocks startup.
        The cached result from the previous run is already showing by then; this only
        corrects it when the user's install actually changed.
        """
        self._check_torch_version()

    def _cached_pytorch_ok(self) -> bool:
        ver = parse_torch_probe(self._settings.value("torch_version", "", type=str))
        return bool(ver and ver >= MIN_TORCH)

    def _on_force_libuv_toggled(self, checked: bool):
        self._app.set("force_use_libuv_off", checked)
        self._refresh_libuv_status()

    def _refresh_libuv_status(self):
        text = libuv_status(force=self._app.get("force_use_libuv_off"))
        user_val = windows_user_env_libuv()
        if user_val is None and windows_system_env_libuv() is not None:
            text += ("  Found at System scope — Lora Forge can only clear the User "
                     "scope; remove it via System Properties.")
        self._libuv_status_label.setText(text)
        self._clear_libuv_btn.setEnabled(user_val is not None)

    def _clear_libuv_from_windows(self):
        current = windows_user_env_libuv()
        if current is None:
            return
        if QMessageBox.question(
            self, "Clear USE_LIBUV",
            f"Delete USE_LIBUV={current} from your Windows user environment?\n\n"
            "New processes elsewhere will see the change immediately — restart "
            "Lora Forge for it to apply here.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        if clear_windows_user_env_libuv():
            self._torch_log.append(
                f"Cleared USE_LIBUV={current} from your Windows user environment. "
                "Restart Lora Forge for it to take effect here.")
        else:
            self._torch_log.append(
                "Could not clear USE_LIBUV from Windows — check your permissions and try again.")
        self._refresh_libuv_status()

    def _reset_training_environment(self):
        # Defer the cache clear until the re-probe actually resolves (see
        # _handle_torch_check_output) — clearing it here, before the async probe has
        # even started, could strand the last known-good state if the probe bails
        # early (no interpreter) or never calls back at all. Only arm the pending
        # flag when _check_torch_version actually launched a probe: if it took an
        # early-return path instead, there is no re-probe coming for THIS call, and
        # leaving the flag set would let some later, unrelated probe (a manual Check
        # PyTorch Version click, a post-Upgrade re-check) get misread as Reset's own
        # re-probe — and destructively clear known-good state if that unrelated
        # probe's output happens to be ambiguous.
        self._torch_log.append("Resetting training environment…")
        self._reset_pending = self._check_torch_version()
        self._refresh_libuv_status()

    def _handle_torch_check_output(self, p):
        text = bytes(p.readAllStandardOutput()).decode("utf-8", "replace").strip()
        if text:
            self._torch_log.append(text)
        ver = parse_torch_probe(text)
        reset_pending = getattr(self, "_reset_pending", False)
        self._reset_pending = False
        if ver:
            self._pytorch_ok = ver >= MIN_TORCH
            # Remember it across restarts (see __init__), then tell Home to redraw.
            self._settings.setValue("torch_version", text)
            self._settings.sync()
            self.pytorch_status_changed.emit()
        elif reset_pending:
            # An explicit Reset whose re-probe came back with nothing parseable: the
            # stale cached value is no longer trustworthy, so clear it now — but only
            # now that we know the re-probe genuinely ran and failed, not preemptively.
            self._settings.remove("torch_version")
            self._settings.sync()
            self._pytorch_ok = False
            self.pytorch_status_changed.emit()
        # Blackwell needs cu128 wheels: a cu121/cu124 torch on an RTX 50-series card
        # fails at run time with "no kernel image is available".
        cu = re.search(r"cuda\s+(\d+)\.(\d+)", text)
        if cu and (int(cu.group(1)), int(cu.group(2))) < (12, 8):
            from core import gpu_check
            if gpu_check.is_rtx_50_series(gpu_check.gpu_name()):
                self._torch_log.append(
                    "⚠ RTX 50-series GPU detected, but this PyTorch build lacks Blackwell "
                    "kernels (training fails with 'no kernel image is available'). "
                    "Click Upgrade to install the CUDA 12.8 build."
                )

    def is_pytorch_ok(self) -> bool:
        return bool(getattr(self, "_pytorch_ok", False))

    def _upgrade_torch(self):
        py = self._venv_python()
        if not py or not Path(py).is_file():
            self._torch_log.append("Could not find a training python interpreter. Check the install (.venv) or set the sd-scripts path.")
            return
        if self._torch_process is not None and self._torch_process.state() != QProcess.NotRunning:
            self._torch_log.append("A PyTorch operation is already running.")
            return
        from core import gpu_check
        spec, index, label = torch_upgrade_plan(
            gpu_check.is_rtx_50_series(gpu_check.gpu_name()))
        self._torch_log.append(
            f"\n⬆ Upgrading PyTorch ({label}). This downloads ~2.5GB and may take several minutes…")
        self._upgrade_torch_btn.setEnabled(False)
        self._torch_process = QProcess(self)
        apply_no_window(self._torch_process)  # no console window pop-up on Windows
        self._torch_process.setProcessChannelMode(QProcess.MergedChannels)
        self._torch_process.readyReadStandardOutput.connect(self._on_torch_output)
        self._torch_process.finished.connect(self._on_torch_finished)
        args = [
            "-m", "pip", "install", "--upgrade",
            spec, "torchvision",
            "--index-url", index,
        ]
        self._torch_process.start(py, args)

    def _on_torch_output(self):
        if not self._torch_process:
            return
        text = bytes(self._torch_process.readAllStandardOutput()).decode("utf-8", "replace")
        for line in text.splitlines():
            if line.strip():
                self._torch_log.append(line)
        self._torch_log.verticalScrollBar().setValue(self._torch_log.verticalScrollBar().maximum())

    def _on_torch_finished(self, exit_code, _status):
        self._upgrade_torch_btn.setEnabled(True)
        if exit_code == 0:
            self._torch_log.append("✔ PyTorch upgrade finished. Re-checking version…")
            self._check_torch_version()   # keeps the remembered version honest
        else:
            self._torch_log.append(f"✘ Upgrade failed (exit code {exit_code}).")

    # ------------------------------------------------------------------
    # Verify / status
    # ------------------------------------------------------------------

    def _verify_environment(self):
        sd = self.get_sdscripts_path()
        sd_ok = bool(sd) and Path(sd).is_dir() and (Path(sd) / "anima_train_network.py").is_file()
        self._set_status(self._sdscripts_status, sd_ok, bool(sd))

        for edit, dot in (
            (self._dit_edit, self._dit_status),
            (self._qwen3_edit, self._qwen3_status),
            (self._vae_edit, self._vae_status),
            (self._moss_dit_edit, self._moss_dit_status),
            (self._moss_te_edit, self._moss_te_status),
            (self._moss_vae_edit, self._moss_vae_status),
        ):
            val = edit.text().strip()
            ok = bool(val) and Path(val).is_file()
            self._set_status(dot, ok, bool(val))

        self._check_output()

        valid, msg = self.is_environment_valid()
        if valid:
            self._verify_label.setText(f"✔ {msg}")
            self._verify_label.setStyleSheet("color: #d4af37; font-weight: 600;")
        else:
            self._verify_label.setText(f"✘ {msg}")
            self._verify_label.setStyleSheet("color: #d9534f; font-weight: 600;")
        self._save_settings()

    def _check_output(self):
        out = self._output_edit.text().strip()
        if not out:
            self._set_status(self._output_status, False, False)
            return
        try:
            Path(out).mkdir(parents=True, exist_ok=True)
            self._set_status(self._output_status, os.access(out, os.W_OK), True)
        except OSError:
            self._set_status(self._output_status, False, True)

    @staticmethod
    def _set_status(label: QLabel, ok: bool, has_value: bool):
        if not has_value:
            label.setStyleSheet("color: #6a6a72; font-size: 18px;")
        elif ok:
            label.setStyleSheet("color: #d4af37; font-size: 18px;")
        else:
            label.setStyleSheet("color: #d9534f; font-size: 18px;")

    # ------------------------------------------------------------------
    # Settings control center (Forge / Advanced / Defaults) — via AppSettings
    # ------------------------------------------------------------------

    def get_app_settings(self) -> AppSettings:
        return self._app

    def _browse_scan(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Model Scan Folder", self._scan_edit.text())
        if folder:
            self._scan_edit.setText(folder)

    def _browse_forge_lora(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Forge LoRA Folder", self._forge_lora_edit.text())
        if folder:
            self._forge_lora_edit.setText(folder)

    def _browse_comfy_lora(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select ComfyUI LoRA Folder", self._comfy_lora_edit.text())
        if folder:
            self._comfy_lora_edit.setText(folder)

    def _test_forge(self):
        from core import forge_api
        url = self._forge_url_edit.text().strip() or "http://127.0.0.1:7860"
        if forge_api.ping(url):
            self._forge_status.setText("✔ Forge API reachable.")
            self._forge_status.setStyleSheet("font-size: 11px; color: #d4af37;")
        else:
            self._forge_status.setText("✘ Cannot reach Forge. Start it with --api enabled at this URL.")
            self._forge_status.setStyleSheet("font-size: 11px; color: #d9534f;")

    # ---- Fine-Tuning modals (host the stashed config groups) ----
    def _restash_setting(self, grp):
        if grp is not None:
            grp.setParent(self)
            grp.setVisible(False)

    def _open_setting_modal(self, grp, title, subtitle):
        grp.setVisible(True)
        modal = ForgeModal(self.window(), title=title, eyebrow="Fine Tuning",
                           subtitle=subtitle, max_width=600)
        modal.body.addWidget(grp)
        modal.closed.connect(lambda g=grp: self._restash_setting(g))
        modal.add_footer_button("Done", primary=True).clicked.connect(modal.close_modal)
        modal.open()

    def _open_defaults_modal(self):
        self._open_setting_modal(self._defaults_group, "App Defaults",
                                 "New runs start from these — dim, alpha, steps, caption order.")

    def _open_advanced_modal(self):
        self._open_setting_modal(self._advanced_group, "Advanced Training",
                                 "Flow weighting, dropout, VRAM warnings — leave default unless you know.")

    def _open_presets_settings_modal(self):
        self._refresh_presets_list()
        self._open_setting_modal(self._presets_group, "Training Presets",
                                 "Your named training intents — picked from the front page in one click.")

    # ------------------------------------------------------------------
    # Training presets management (custom intent profiles)
    # ------------------------------------------------------------------

    def _build_presets_group(self) -> QGroupBox:
        from PySide6.QtWidgets import QListWidget
        g = QGroupBox("Custom Training Presets")
        v = QVBoxLayout(g)
        v.setSpacing(8)
        hint = QLabel(
            "Built-ins (Person, Object / Concept, Style) are always available on the Home "
            "PRESET picker. Add your own here for bigger configurations — larger network "
            "dim/alpha, longer runs, a fixed optimizer — and they appear in the same picker.")
        hint.setObjectName("label_field")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self._presets_list = QListWidget()
        self._presets_list.setMinimumHeight(140)
        v.addWidget(self._presets_list)
        row = QHBoxLayout()
        add_btn = QPushButton("➕ New Preset…")
        add_btn.clicked.connect(self._new_preset_dialog)
        from utils.styles import asset_url
        del_btn = QPushButton(" Delete selected")
        del_btn.setIcon(QIcon(asset_url("trash.png")))
        del_btn.setIconSize(QSize(14, 14))
        del_btn.clicked.connect(self._delete_selected_preset)
        row.addWidget(add_btn)
        row.addWidget(del_btn)
        row.addStretch()
        v.addLayout(row)
        return g

    def _refresh_presets_list(self):
        from core import train_presets as tp
        self._presets_list.clear()
        for p in tp.parse_customs(self._app.get(tp.SETTINGS_KEY)):
            self._presets_list.addItem(f"{p.name}   —   {tp.summary_line(p)}")

    def _new_preset_dialog(self):
        from core import train_presets as tp
        from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                                       QDoubleSpinBox, QFormLayout, QSpinBox)
        dlg = QDialog(self)
        dlg.setWindowTitle("New Training Preset")
        form = QFormLayout(dlg)
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("e.g. Big Character (dim 64)")
        form.addRow("Name:", name_edit)
        subject = QComboBox()
        for label, key in [("Character / Person", "character"),
                           ("Object / Concept", "concept"), ("Art Style", "style")]:
            subject.addItem(label, key)
        form.addRow("Subject type:", subject)
        optimizer = QComboBox()
        optimizer.addItem("Prodigy+ ScheduleFree — auto LR", "prodigy")
        optimizer.addItem("AdamW8bit — constant LR", "adamw8bit")
        form.addRow("Optimizer:", optimizer)
        lr = QDoubleSpinBox()
        lr.setDecimals(6)
        lr.setRange(0.000001, 0.01)
        lr.setValue(0.0001)
        lr.setEnabled(False)  # Prodigy is LR-free
        optimizer.currentIndexChanged.connect(
            lambda _i: lr.setEnabled(optimizer.currentData() == "adamw8bit"))
        form.addRow("Learning rate:", lr)
        dim = QSpinBox()
        dim.setRange(1, 128)
        dim.setValue(16)
        form.addRow("Network dim:", dim)
        alpha = QSpinBox()
        alpha.setRange(1, 128)
        alpha.setValue(8)
        form.addRow("Alpha:", alpha)
        steps = QSpinBox()
        steps.setRange(0, 100000)
        steps.setSingleStep(100)
        steps.setSpecialValueText("Auto (from dataset)")
        form.addRow("Target steps:", steps)
        uncap = QCheckBox("Uncap steps (long runs — scale with dataset)")
        form.addRow("", uncap)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.Accepted:
            return
        preset = tp.TrainPreset(
            name=name_edit.text(), subject_type=subject.currentData(),
            optimizer=optimizer.currentData(), learning_rate=lr.value(),
            network_dim=dim.value(), network_alpha=alpha.value(),
            target_steps=steps.value(), uncap_steps=uncap.isChecked())
        try:
            self._app.set(tp.SETTINGS_KEY, tp.add_custom(self._app.get(tp.SETTINGS_KEY), preset))
        except ValueError as e:
            QMessageBox.warning(self, "Cannot save preset", str(e))
            return
        self._refresh_presets_list()

    def _delete_selected_preset(self):
        from core import train_presets as tp
        item = self._presets_list.currentItem()
        if item is None:
            return
        name = item.text().split("   —   ")[0].strip()
        if QMessageBox.question(self, "Delete preset", f"Delete preset “{name}”?",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) != QMessageBox.Yes:
            return
        self._app.set(tp.SETTINGS_KEY, tp.remove_custom(self._app.get(tp.SETTINGS_KEY), name))
        self._refresh_presets_list()

    def _build_describe_group(self) -> QGroupBox:
        """Which model writes the .nl captions, and where its weights live.

        This is a machine-level setting, not a per-run one — it belongs here for the
        same reason the sd-scripts path does. Home shows the choice read-only and
        links back here, so no control is duplicated off the front.
        """
        g = QGroupBox("Describe Model (natural-language captions)")
        v = QVBoxLayout(g)
        v.setSpacing(8)
        hint = QLabel(
            "The model that writes the .nl sentence for each image. If you already have "
            "it in your Hugging Face cache — including through a junction — it is used "
            "as-is, with no re-download.")
        hint.setObjectName("label_field")
        hint.setWordWrap(True)
        v.addWidget(hint)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._describe_combo = QComboBox()
        for label, _repo, _vram, _quant in DESCRIBE_MODELS:
            self._describe_combo.addItem(label)
        self._describe_combo.addItem("Custom (Hugging Face repo ID)…")
        model_row.addWidget(self._describe_combo, 1)
        v.addLayout(model_row)

        custom_row = QHBoxLayout()
        custom_row.addWidget(QLabel("Repo ID:"))
        self._describe_custom = QLineEdit()
        self._describe_custom.setPlaceholderText("org/model-name")
        custom_row.addWidget(self._describe_custom, 1)
        v.addLayout(custom_row)

        cache_row = QHBoxLayout()
        cache_row.addWidget(QLabel("Model folder:"))
        self._describe_cache = QLineEdit()
        self._describe_cache.setPlaceholderText("Auto — Hugging Face cache")
        cache_browse = QPushButton("Browse…")
        cache_browse.setFixedWidth(108)
        cache_browse.clicked.connect(self._browse_describe_cache)
        cache_clear = QPushButton("Clear")
        cache_clear.setFixedWidth(80)
        cache_clear.clicked.connect(self._clear_describe_cache)
        cache_row.addWidget(self._describe_cache, 1)
        cache_row.addWidget(cache_browse)
        cache_row.addWidget(cache_clear)
        v.addLayout(cache_row)

        self._describe_status = QLabel("")
        self._describe_status.setObjectName("label_field")
        self._describe_status.setWordWrap(True)
        v.addWidget(self._describe_status)

        self._describe_combo.currentIndexChanged.connect(self._on_describe_changed)
        self._describe_custom.editingFinished.connect(self._on_describe_changed)
        self._describe_cache.editingFinished.connect(self._on_describe_changed)

        self._load_describe_settings()
        return g

    def _load_describe_settings(self):
        idx = self._app.get("describe_model_index")
        custom_row = self._describe_combo.count() - 1
        self._describe_combo.setCurrentIndex(custom_row if idx == CUSTOM_INDEX else idx)
        self._describe_custom.setText(self._app.get("describe_model_custom_id"))
        self._describe_cache.setText(self._app.get("describe_cache_dir"))
        self._on_describe_changed()

    def _browse_describe_cache(self):
        d = QFileDialog.getExistingDirectory(
            self, "Describe model folder", self._describe_cache.text().strip())
        if d:
            self._describe_cache.setText(d)
            self._on_describe_changed()

    def _clear_describe_cache(self):
        self._describe_cache.clear()
        self._on_describe_changed()

    def _on_describe_changed(self):
        """Persist the picker, then report what a run would actually load."""
        custom_row = self._describe_combo.count() - 1
        is_custom = self._describe_combo.currentIndex() == custom_row
        self._describe_custom.setEnabled(is_custom)
        index = CUSTOM_INDEX if is_custom else self._describe_combo.currentIndex()

        custom_id = self._describe_custom.text().strip()
        if is_custom and custom_id and not valid_repo_id(custom_id):
            self._describe_status.setText(
                "⚠ A Hugging Face repo ID looks like 'org/model-name'.")
            return

        explicit = self._describe_cache.text().strip()
        self._app.set("describe_model_index", index)
        self._app.set("describe_model_custom_id", custom_id)
        self._app.set("describe_cache_dir", explicit)

        repo_id, cache_dir, _label = read_describe_defaults(self._app)
        where = cache_dir or "Hugging Face cache (HF_HUB_CACHE / HF_HOME / ~/.cache)"
        parts = [f"{repo_id} → {where}",
                 "Already downloaded."
                 if model_is_present(repo_id, explicit) else "Downloads on first use."]

        entry = entry_for_repo(repo_id)
        free_mb = free_vram_mb()
        if entry and free_mb is not None and free_mb < entry[2] * 1024:
            parts.append(f"⚠ {free_mb // 1024}GB VRAM free; this model wants "
                         f"{entry[2]}GB. It may be slow or fail to load.")
        self._describe_status.setText("   ".join(parts))
        self.describe_model_changed.emit()

    def _build_forge_group(self) -> QGroupBox:
        g = QGroupBox("Forge / Stable Diffusion API (deliver + test-render)")
        v = QVBoxLayout(g)
        v.setSpacing(8)
        hint = QLabel("Optional. After training, deliver the LoRA into Forge and/or auto test-render it. "
                      "Start Forge with --api enabled.")
        hint.setObjectName("label_field")
        hint.setWordWrap(True)
        v.addWidget(hint)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("API URL:"))
        self._forge_url_edit = QLineEdit()
        self._forge_url_edit.setPlaceholderText("http://127.0.0.1:7860")
        r1.addWidget(self._forge_url_edit)
        tb = QPushButton("Test connection")
        tb.setFixedWidth(152)
        tb.clicked.connect(self._test_forge)
        r1.addWidget(tb)
        v.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("LoRA folder:"))
        self._forge_lora_edit = QLineEdit()
        self._forge_lora_edit.setPlaceholderText(".../Forge/models/Lora")
        r2.addWidget(self._forge_lora_edit)
        lb = QPushButton("Browse…")
        lb.setFixedWidth(108)
        lb.clicked.connect(self._browse_forge_lora)
        r2.addWidget(lb)
        v.addLayout(r2)
        self._forge_deliver_check = QCheckBox("Auto-deliver LoRA to Forge after training")
        self._forge_test_check = QCheckBox("Auto test-render after training")
        self._startup_update_check = QCheckBox("Check for updates at startup")
        v.addWidget(self._forge_deliver_check)
        v.addWidget(self._forge_test_check)
        v.addWidget(self._startup_update_check)
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("ComfyUI LoRA folder:"))
        self._comfy_lora_edit = QLineEdit()
        self._comfy_lora_edit.setPlaceholderText(".../ComfyUI/models/loras")
        self._comfy_lora_edit.setToolTip(
            "Optional — enables 'Deliver to ComfyUI' (a plain copy; ComfyUI has no render API to target)")
        r3.addWidget(self._comfy_lora_edit)
        cb = QPushButton("Browse…")
        cb.setFixedWidth(108)
        cb.clicked.connect(self._browse_comfy_lora)
        r3.addWidget(cb)
        v.addLayout(r3)
        self._forge_status = QLabel("")
        self._forge_status.setWordWrap(True)
        self._forge_status.setStyleSheet("font-size: 11px; color: #8a8a93;")
        v.addWidget(self._forge_status)
        return g

    def _build_advanced_group(self) -> QGroupBox:
        g = QGroupBox("Advanced Training  (leave default unless you know)")
        v = QVBoxLayout(g)
        v.setSpacing(6)
        fr = QHBoxLayout()
        fr.addWidget(QLabel("Flow weighting:"))
        self._weighting_combo = QComboBox()
        self._weighting_combo.addItems(["sigmoid", "logit_normal", "mode", "cosmap", "uniform"])
        fr.addWidget(self._weighting_combo)
        fr.addWidget(QLabel("logit μ:"))
        self._logit_mean_spin = QDoubleSpinBox()
        self._logit_mean_spin.setRange(-5.0, 5.0)
        self._logit_mean_spin.setSingleStep(0.1)
        fr.addWidget(self._logit_mean_spin)
        fr.addWidget(QLabel("logit σ:"))
        self._logit_std_spin = QDoubleSpinBox()
        self._logit_std_spin.setRange(0.1, 5.0)
        self._logit_std_spin.setSingleStep(0.1)
        self._logit_std_spin.setValue(1.0)
        fr.addWidget(self._logit_std_spin)
        fr.addStretch()
        v.addLayout(fr)
        dr = QHBoxLayout()
        dr.addWidget(QLabel("Caption dropout:"))
        self._capdrop_spin = QDoubleSpinBox()
        self._capdrop_spin.setRange(0.0, 0.5)
        self._capdrop_spin.setSingleStep(0.05)
        dr.addWidget(self._capdrop_spin)
        dr.addWidget(QLabel("Network dropout:"))
        self._netdrop_spin = QDoubleSpinBox()
        self._netdrop_spin.setRange(0.0, 0.5)
        self._netdrop_spin.setSingleStep(0.05)
        dr.addWidget(self._netdrop_spin)
        dr.addStretch()
        v.addLayout(dr)
        self._flip_check = QCheckBox("flip_aug (horizontal flip — avoid for asymmetric subjects or text)")
        v.addWidget(self._flip_check)
        vr = QHBoxLayout()
        vr.addWidget(QLabel("Warn if free VRAM below:"))
        self._min_vram_spin = QSpinBox()
        self._min_vram_spin.setRange(0, 24000)
        self._min_vram_spin.setSingleStep(500)
        self._min_vram_spin.setSuffix(" MB")
        self._min_vram_spin.setToolTip(
            "Before training, warn if free VRAM is below this (0 disables). "
            "Catches another app (e.g. Forge) holding the GPU.")
        vr.addWidget(self._min_vram_spin)
        vr.addStretch()
        v.addLayout(vr)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #2a2a1e;")
        v.addWidget(sep)
        qr = QHBoxLayout()
        qr.addWidget(QLabel("Quality prefix:"))
        self._sample_quality_edit = QLineEdit()
        self._sample_quality_edit.setPlaceholderText("masterpiece, best quality, score_7, safe — clear for raw output")
        self._sample_quality_edit.setToolTip(
            "Prepended to every preview prompt (after the trigger/anchor) so progress images "
            "aren't rendered in Anima's plain unrefined-base style. Clear to preview raw LoRA output."
        )
        qr.addWidget(self._sample_quality_edit, 1)
        v.addLayout(qr)
        sr = QHBoxLayout()
        sr.addWidget(QLabel("Sampler:"))
        self._sample_sampler_edit = QLineEdit()
        self._sample_sampler_edit.setText("euler_a")
        self._sample_sampler_edit.setFixedWidth(120)
        sr.addWidget(self._sample_sampler_edit)
        self._sample_first_check = QCheckBox("also at first")
        sr.addWidget(self._sample_first_check)
        sr.addStretch()
        v.addLayout(sr)
        return g

    def _build_defaults_group(self) -> QGroupBox:
        g = QGroupBox("App Defaults  (new runs start from these)")
        v = QVBoxLayout(g)
        v.setSpacing(6)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Dim:"))
        self._def_dim_spin = QSpinBox()
        self._def_dim_spin.setRange(1, 128)
        self._def_dim_spin.setValue(16)
        r1.addWidget(self._def_dim_spin)
        r1.addWidget(QLabel("Alpha:"))
        self._def_alpha_spin = QSpinBox()
        self._def_alpha_spin.setRange(1, 128)
        self._def_alpha_spin.setValue(8)
        r1.addWidget(self._def_alpha_spin)
        r1.addStretch()
        v.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Target steps:"))
        self._def_steps_spin = QSpinBox()
        self._def_steps_spin.setRange(500, 40000)
        self._def_steps_spin.setSingleStep(100)
        self._def_steps_spin.setValue(500)
        r2.addWidget(self._def_steps_spin)
        r2.addWidget(QLabel("Caption order:"))
        self._def_order_combo = QComboBox()
        self._def_order_combo.addItems(["NL then tags", "Tags then NL"])
        r2.addWidget(self._def_order_combo)
        r2.addStretch()
        v.addLayout(r2)
        # UI font — swaps the four decorative faces (headlines/nav/taglines/body)
        # for readability in non-Latin locales. Functional text always stays on
        # the system UI stack, so a bad pick is always recoverable here.
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("UI font:"))
        self._font_mode_combo = QComboBox()
        self._font_mode_combo.addItems(["Forge fonts (default)", "System font", "Custom…"])
        r3.addWidget(self._font_mode_combo)
        self._font_family_combo = QFontComboBox()
        self._font_family_combo.setVisible(False)
        r3.addWidget(self._font_family_combo)
        r3.addStretch()
        v.addLayout(r3)
        from PySide6.QtWidgets import QButtonGroup, QRadioButton
        from core.caption_policy import ASK, KEEP, OVERWRITE
        pol = QVBoxLayout()
        pol.addWidget(QLabel("When a dataset already has captions:"))
        self._policy_group = QButtonGroup(self)
        self._policy_buttons = {}
        for key, text in (
            (ASK, "Ask me (recommended)"),
            (KEEP, "Keep them — caption only the images that have none"),
            (OVERWRITE, "Overwrite them without asking"),
        ):
            rb = QRadioButton(text)
            self._policy_group.addButton(rb)
            self._policy_buttons[key] = rb
            pol.addWidget(rb)
        v.addLayout(pol)

        # Caption find/replace rules — banning a tag IS find/replace with an empty
        # replacement ("1boy" -> ""), so one table covers both misgendering fixes
        # ("man" -> "woman") and tag bans. Applied to the caption body only, when the
        # .txt is built (see core/caption_rules.py + core/dataset_manager.combine_caption).
        sep_rules = QFrame()
        sep_rules.setFrameShape(QFrame.HLine)
        sep_rules.setStyleSheet("color: #2a2a1e;")
        v.addWidget(sep_rules)
        v.addWidget(QLabel("Caption find/replace rules:"))
        self._rules_table = QTableWidget(0, 2)
        self._rules_table.setHorizontalHeaderLabels(["Find", "Replace with"])
        self._rules_table.horizontalHeader().setStretchLastSection(True)
        self._rules_table.setMinimumHeight(120)
        self._rules_table.setSelectionBehavior(QTableWidget.SelectRows)
        v.addWidget(self._rules_table)
        rules_btn_row = QHBoxLayout()
        add_rule_btn = QPushButton("Add rule")
        add_rule_btn.clicked.connect(self._add_rule_row)
        remove_rule_btn = QPushButton("Remove selected")
        remove_rule_btn.clicked.connect(self._remove_selected_rule)
        rules_btn_row.addWidget(add_rule_btn)
        rules_btn_row.addWidget(remove_rule_btn)
        rules_btn_row.addStretch()
        v.addLayout(rules_btn_row)
        rules_hint = QLabel(
            "Whole words only — “man” won't match “woman” or “human”. "
            "Leave Replace with empty to delete the term: that bans a tag cleanly, but in a "
            "written description it leaves a gap (“a man stands” → “a stands”). For prose, "
            "replacing beats deleting."
        )
        rules_hint.setObjectName("label_field")
        rules_hint.setWordWrap(True)
        v.addWidget(rules_hint)
        return g

    def _add_rule_row(self):
        row = self._rules_table.rowCount()
        self._rules_table.insertRow(row)
        self._rules_table.setItem(row, 0, QTableWidgetItem(""))
        self._rules_table.setItem(row, 1, QTableWidgetItem(""))
        self._save_rules_from_table()

    def _remove_selected_rule(self):
        rows = sorted({idx.row() for idx in self._rules_table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for row in rows:
            self._rules_table.removeRow(row)
        self._save_rules_from_table()

    def _load_rules_into_table(self):
        """Populate the table from AppSettings. Called from _bind_app_widgets BEFORE the
        itemChanged connection is registered, so populating never triggers a write-back."""
        rules = parse_rules(self._app.get("caption_rules_json"))
        self._rules_table.setRowCount(0)
        for find, repl in rules:
            row = self._rules_table.rowCount()
            self._rules_table.insertRow(row)
            self._rules_table.setItem(row, 0, QTableWidgetItem(find))
            self._rules_table.setItem(row, 1, QTableWidgetItem(repl))

    def _rules_from_table(self) -> list:
        """Read the table into [(find, replace)]. Cells can be None (row just inserted,
        item not yet created for that column) so every read is guarded."""
        rules = []
        for row in range(self._rules_table.rowCount()):
            find_item = self._rules_table.item(row, 0)
            repl_item = self._rules_table.item(row, 1)
            find = find_item.text().strip() if find_item is not None else ""
            repl = repl_item.text() if repl_item is not None else ""
            if find:
                rules.append((find, repl))
        return rules

    def _save_rules_from_table(self, _item=None):
        self._app.set("caption_rules_json", dump_rules(self._rules_from_table()))

    def _bind_app_widgets(self):
        """Load values from AppSettings into the new widgets, then save on change."""
        a = self._app
        self._scan_edit.setText(a.first_run_scan_default())
        self._forge_url_edit.setText(a.get("forge_api_url"))
        self._forge_lora_edit.setText(a.get("forge_lora_dir"))
        self._comfy_lora_edit.setText(a.get("comfyui_lora_dir"))
        self._forge_deliver_check.setChecked(a.get("forge_auto_deliver"))
        self._forge_test_check.setChecked(a.get("forge_auto_test"))
        self._startup_update_check.setChecked(a.get("startup_update_check"))
        self._force_libuv_check.setChecked(a.get("force_use_libuv_off"))
        self._weighting_combo.setCurrentText(a.get("weighting_scheme"))
        self._logit_mean_spin.setValue(a.get("logit_mean"))
        self._logit_std_spin.setValue(a.get("logit_std"))
        self._capdrop_spin.setValue(a.get("caption_dropout_rate"))
        self._netdrop_spin.setValue(a.get("network_dropout"))
        self._flip_check.setChecked(a.get("flip_aug"))
        self._min_vram_spin.setValue(a.get("min_free_vram_mb"))
        self._sample_quality_edit.setText(a.get("sample_quality_prefix"))
        self._sample_sampler_edit.setText(a.get("sample_sampler"))
        self._sample_first_check.setChecked(a.get("sample_at_first"))
        self._def_dim_spin.setValue(a.get("default_network_dim"))
        self._def_alpha_spin.setValue(a.get("default_network_alpha"))
        self._def_steps_spin.setValue(a.get("default_target_steps"))
        self._def_order_combo.setCurrentIndex(0 if a.get("default_caption_order") == "nl_first" else 1)
        _font_modes = ("forge", "system", "custom")
        mode = a.get("ui_font_mode")
        self._font_mode_combo.setCurrentIndex(
            _font_modes.index(mode) if mode in _font_modes else 0)
        fam_name = a.get("ui_font_family")
        if fam_name:
            from PySide6.QtGui import QFont
            self._font_family_combo.setCurrentFont(QFont(fam_name))
        self._font_family_combo.setVisible(mode == "custom")
        from core.caption_policy import ASK
        policy = a.get("caption_existing_policy")
        self._policy_buttons.get(policy, self._policy_buttons[ASK]).setChecked(True)
        self._load_rules_into_table()
        # Save on change (connected AFTER loading so load doesn't trigger writes)
        self._scan_edit.textChanged.connect(lambda t: a.set("model_scan_dir", t))
        self._forge_url_edit.textChanged.connect(lambda t: a.set("forge_api_url", t))
        self._forge_lora_edit.textChanged.connect(lambda t: a.set("forge_lora_dir", t))
        self._comfy_lora_edit.textChanged.connect(lambda t: a.set("comfyui_lora_dir", t))
        self._forge_deliver_check.toggled.connect(lambda b: a.set("forge_auto_deliver", b))
        self._forge_test_check.toggled.connect(lambda b: a.set("forge_auto_test", b))
        self._startup_update_check.toggled.connect(lambda b: a.set("startup_update_check", b))
        self._force_libuv_check.toggled.connect(self._on_force_libuv_toggled)
        self._weighting_combo.currentTextChanged.connect(lambda t: a.set("weighting_scheme", t))
        self._logit_mean_spin.valueChanged.connect(lambda val: a.set("logit_mean", val))
        self._logit_std_spin.valueChanged.connect(lambda val: a.set("logit_std", val))
        self._capdrop_spin.valueChanged.connect(lambda val: a.set("caption_dropout_rate", val))
        self._netdrop_spin.valueChanged.connect(lambda val: a.set("network_dropout", val))
        self._flip_check.toggled.connect(lambda b: a.set("flip_aug", b))
        self._min_vram_spin.valueChanged.connect(lambda val: a.set("min_free_vram_mb", val))
        self._sample_quality_edit.textChanged.connect(lambda t: a.set("sample_quality_prefix", t))
        self._sample_sampler_edit.textChanged.connect(lambda t: a.set("sample_sampler", t))
        self._sample_first_check.toggled.connect(lambda b: a.set("sample_at_first", b))
        self._def_dim_spin.valueChanged.connect(lambda val: a.set("default_network_dim", val))
        self._def_alpha_spin.valueChanged.connect(lambda val: a.set("default_network_alpha", val))
        self._def_steps_spin.valueChanged.connect(lambda val: a.set("default_target_steps", val))
        self._def_order_combo.currentIndexChanged.connect(
            lambda i: a.set("default_caption_order", "nl_first" if i == 0 else "tags_first"))
        self._font_mode_combo.currentIndexChanged.connect(self._on_font_mode_changed)
        self._font_family_combo.currentFontChanged.connect(self._on_font_family_changed)
        for key, rb in self._policy_buttons.items():
            rb.toggled.connect(
                lambda checked, k=key: checked and a.set("caption_existing_policy", k))
        self._rules_table.itemChanged.connect(self._save_rules_from_table)

    def _on_font_mode_changed(self, idx: int) -> None:
        from utils.fonts import apply_app_font
        self._app.set("ui_font_mode", ("forge", "system", "custom")[idx])
        self._font_family_combo.setVisible(idx == 2)
        if idx == 2 and not self._app.get("ui_font_family"):
            # Seed from whatever the combo shows so custom mode is never "empty".
            self._app.set("ui_font_family",
                          self._font_family_combo.currentFont().family())
        apply_app_font()

    def _on_font_family_changed(self, font) -> None:
        from utils.fonts import apply_app_font
        self._app.set("ui_font_family", font.family())
        if self._app.get("ui_font_mode") == "custom":
            apply_app_font()

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _on_text_changed(self):
        self._save_timer.stop()
        self._save_timer.start()
        self.settings_changed.emit()

    def _save_settings(self):
        # Install-internal paths are stored relative to the app root (QSettings is
        # machine-global, so absolute paths leak between side-by-side copies).
        from core.paths import to_portable
        self._settings.setValue("sdscripts_path", to_portable(self._sdscripts_edit.text()))
        self._settings.setValue("dit_path", to_portable(self._dit_edit.text()))
        self._settings.setValue("qwen3_path", to_portable(self._qwen3_edit.text()))
        self._settings.setValue("vae_path", to_portable(self._vae_edit.text()))
        self._settings.setValue("moss_dit_path", to_portable(self._moss_dit_edit.text()))
        self._settings.setValue("moss_te_path", to_portable(self._moss_te_edit.text()))
        self._settings.setValue("moss_vae_path", to_portable(self._moss_vae_edit.text()))
        self._settings.setValue("output_dir", to_portable(self._output_edit.text()))
        self._settings.sync()

    def _load_settings(self):
        from core.paths import app_root, from_portable, in_other_install
        root = app_root()

        # sd-scripts: this install's bundled copy wins over an invalid path OR a path
        # that leaked from another Lora Forge copy on the same machine. A user's own
        # external sd-scripts clone is still honored.
        default_sd = str(root / "sd-scripts")
        saved_sd = from_portable(self._settings.value("sdscripts_path", "", type=str))
        if (not saved_sd
                or not (Path(saved_sd) / "anima_train_network.py").is_file()
                or in_other_install(saved_sd)):
            saved_sd = default_sd
        self._sdscripts_edit.setText(saved_sd)

        self._dit_edit.setText(from_portable(self._settings.value("dit_path", "", type=str)))
        self._qwen3_edit.setText(from_portable(self._settings.value("qwen3_path", "", type=str)))
        self._vae_edit.setText(from_portable(self._settings.value("vae_path", "", type=str)))
        self._moss_dit_edit.setText(from_portable(self._settings.value("moss_dit_path", "", type=str)))
        self._moss_te_edit.setText(from_portable(self._settings.value("moss_te_path", "", type=str)))
        self._moss_vae_edit.setText(from_portable(self._settings.value("moss_vae_path", "", type=str)))

        from core.paths import DEFAULT_OUTPUT_DIR
        default_out = DEFAULT_OUTPUT_DIR
        saved_out = from_portable(self._settings.value("output_dir", default_out, type=str))
        if not saved_out or in_other_install(saved_out):
            saved_out = default_out
        self._output_edit.setText(saved_out)

    # ------------------------------------------------------------------
    # Install dialog
    # ------------------------------------------------------------------

    def _show_install_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Lora Forge Setup Instructions")
        dlg.setMinimumWidth(620)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)
        title = QLabel("Manual Setup Instructions")
        title.setStyleSheet("font-size: 15px; font-weight: 700; color: #d4af37;")
        layout.addWidget(title)
        instructions = QTextEdit()
        instructions.setReadOnly(True)
        instructions.setMinimumHeight(360)
        instructions.setPlainText(
            "Step 1 — Prerequisites\n"
            "  • Python 3.10 or 3.11, Git, and an NVIDIA GPU with CUDA 12.1 drivers\n\n"
            "Step 2 — sd-scripts (must support Anima and MOSS-Image)\n"
            "    git clone https://github.com/kohya-ss/sd-scripts.git\n"
            "    cd sd-scripts && git pull   (ensure anima_train_network.py exists)\n\n"
            "Step 3 — Virtual environment\n"
            "    python -m venv venv\n"
            "    venv\\Scripts\\activate\n\n"
            "Step 4 — Install PyTorch 2.5+ (Anima requires it; older = NaN loss)\n"
            "    pip install \"torch>=2.5\" torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "  RTX 50-series (Blackwell) cards need the CUDA 12.8 build instead:\n"
            "    pip install \"torch>=2.7\" torchvision --index-url https://download.pytorch.org/whl/cu128\n"
            "  (or use the 'Upgrade PyTorch' button above)\n\n"
            "Step 5 — sd-scripts dependencies\n"
            "    pip install -r requirements.txt\n"
            "    pip install prodigy-plus-schedule-free   (Prodigy+ ScheduleFree optimizer)\n"
            "    pip install accelerate && accelerate config\n\n"
            "Step 6 — Anima model files (or use Auto-detect)\n"
            "  From https://huggingface.co/circlestone-labs/Anima, under split_files/:\n"
            f"    • DiT checkpoint:    diffusion_models/{PREFERRED_DIT}   (~3.9 GB)\n"
            f"    • Qwen3 text encoder: text_encoders/{QWEN3_FILENAME}   (~1.1 GB)\n"
            f"    • Qwen-Image VAE:     vae/{QWEN_VAE_FILENAME}   (~242 MB)\n"
            "  Put them anywhere under your model scan folder — Auto-detect reads each\n"
            "  file's contents, so the folder names and any renames don't matter.\n\n"
            "  NOT interchangeable: an Anima checkpoint downloaded from elsewhere may be\n"
            "  saved in ComfyUI's wrapped layout ('model.diffusion_model.…' weights).\n"
            "  This trainer needs the 'net.…' layout, i.e. the split_files copy above.\n\n"
            "Step 7 — Describe model (natural-language captions)\n"
            "    Pick it in 'Describe Model' above. If you already have the model in your\n"
            "    Hugging Face cache (including via a junction), it is used as-is.\n\n"
            "Note: xformers is NOT used by Anima (it uses SDPA).\n"
        )
        layout.addWidget(instructions)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec()
