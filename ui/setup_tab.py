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
from core.env import subprocess_python
from core.settings import AppSettings
from ui.forge_modal import ForgeModal
from utils.proc import apply_no_window

# Shared QSettings ids (single source of truth in core.settings).
from core.settings import SETTINGS_ORG, SETTINGS_APP  # noqa: E402

QWEN3_FILENAME = "qwen_3_06b_base.safetensors"
QWEN_VAE_FILENAME = "qwen_image_vae.safetensors"
# Official Anima DiT filename; auto-detect also matches any 'anima*.safetensors' rename.
PREFERRED_DIT = "anima-base-v1.0.safetensors"


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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(1000)
        self._save_timer.timeout.connect(self._save_settings)
        self._torch_process = None
        self._pytorch_ok = False
        self._app = AppSettings()
        self._build_ui()
        self._load_settings()
        self._bind_app_widgets()

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

    def get_output_dir(self) -> str:
        return self._output_edit.text().strip()

    def get_lmstudio_url(self) -> str:
        return self._lms_url_edit.text().strip() or "http://localhost:1234/v1"

    def get_lmstudio_model(self) -> str:
        return self._lms_model_edit.text().strip()

    def is_environment_valid(self) -> tuple:
        """Returns (is_valid: bool, message: str)."""
        sd = self.get_sdscripts_path()
        out = self.get_output_dir()

        if not sd:
            return False, "sd-scripts path is not set."
        if not Path(sd).is_dir():
            return False, "sd-scripts path does not exist."
        if not (Path(sd) / "anima_train_network.py").is_file():
            return False, "sd-scripts path does not contain anima_train_network.py (update sd-scripts)."

        for label, path in (
            ("Anima DiT checkpoint", self.get_dit_path()),
            ("Qwen3 text encoder", self.get_qwen3_path()),
            ("Qwen-Image VAE", self.get_vae_path()),
        ):
            if not path:
                return False, f"{label} is not set."
            if not Path(path).is_file():
                return False, f"{label} file does not exist."
            if not path.lower().endswith((".safetensors", ".pth", ".ckpt")):
                return False, f"{label} must be a .safetensors/.pth file."

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
        sd_group = QGroupBox("Kohya sd-scripts (Anima)")
        sd_layout = QVBoxLayout(sd_group)
        sd_layout.setSpacing(8)
        sd_hint = QLabel("Path to your local Kohya sd-scripts folder (must include anima_train_network.py).")
        sd_hint.setObjectName("label_field")
        sd_hint.setWordWrap(True)
        sd_layout.addWidget(sd_hint)
        sd_row = QHBoxLayout()
        self._sdscripts_edit = QLineEdit()
        self._sdscripts_edit.setPlaceholderText("e.g. .../AnimaForge/sd-scripts")
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

        # LM Studio group (caption refinement)
        lms_group = QGroupBox("LM Studio (caption refinement)")
        lms_layout = QVBoxLayout(lms_group)
        lms_layout.setSpacing(8)
        lms_hint = QLabel(
            "Vision-LLM endpoint used by the Dataset tab's 'LLM Pass' caption refiner. "
            "Start the LM Studio server and load an uncensored vision model (e.g. Gemma-4)."
        )
        lms_hint.setObjectName("label_field")
        lms_hint.setWordWrap(True)
        lms_layout.addWidget(lms_hint)

        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("Base URL:"))
        self._lms_url_edit = QLineEdit()
        self._lms_url_edit.setPlaceholderText("http://localhost:1234/v1")
        url_row.addWidget(self._lms_url_edit)
        lms_layout.addLayout(url_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model id:"))
        self._lms_model_edit = QLineEdit()
        self._lms_model_edit.setPlaceholderText("(blank = server's loaded model)")
        self._lms_model_edit.setToolTip(
            "Optional. Must be a VISION model for caption refine. Recommended: "
            "qwen2.5-vl-7b-instruct (e.g. huihui-ai's abliterated build for adult content). "
            "Leave blank to use whatever model LM Studio currently has loaded."
        )
        model_row.addWidget(self._lms_model_edit)
        test_btn = QPushButton("Test connection")
        test_btn.setFixedWidth(152)
        test_btn.clicked.connect(self._test_lmstudio)
        model_row.addWidget(test_btn)
        lms_layout.addLayout(model_row)

        self._lms_status = QLabel("")
        self._lms_status.setWordWrap(True)
        self._lms_status.setStyleSheet("font-size: 11px; color: #8a8a93;")
        lms_layout.addWidget(self._lms_status)

        self._lms_in_process_check = QCheckBox(
            "Include AI Refine step in one-click Process & auto-pipeline runs"
        )
        self._lms_in_process_check.setToolTip(
            "Off by default. The ✨ Refine button on the Dataset tab still works regardless. "
            "When on, ▶ Process and the Home auto-pipeline add the LM Studio refine pass — "
            "LM Studio must be running with a vision model."
        )
        lms_layout.addWidget(self._lms_in_process_check)
        layout.addWidget(lms_group)

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

        # The Bench — optional service (stays visible, like LM Studio).
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
                     self._vae_edit, self._output_edit, self._lms_url_edit, self._lms_model_edit):
            edit.textChanged.connect(self._on_text_changed)

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
        found = []
        missing = []
        scan_root = Path(self._scan_edit.text().strip() or self._app.first_run_scan_default())

        qwen3 = scan_root / "text_encoder" / QWEN3_FILENAME
        if qwen3.is_file():
            self._qwen3_edit.setText(str(qwen3))
            found.append("Qwen3 encoder")
        else:
            missing.append("Qwen3 encoder")

        vae = scan_root / "VAE" / QWEN_VAE_FILENAME
        if vae.is_file():
            self._vae_edit.setText(str(vae))
            found.append("Qwen-Image VAE")
        else:
            missing.append("Qwen-Image VAE")

        ckpt_dir = scan_root / "Stable-diffusion"
        dit = None
        if ckpt_dir.is_dir():
            preferred = ckpt_dir / PREFERRED_DIT
            if preferred.is_file():
                dit = preferred
            else:
                anima_ckpts = sorted(
                    f for f in ckpt_dir.glob("*.safetensors") if "anima" in f.name.lower()
                )
                if anima_ckpts:
                    dit = anima_ckpts[0]
        if dit:
            self._dit_edit.setText(str(dit))
            found.append(f"DiT ({dit.name})")
        else:
            missing.append("Anima DiT checkpoint")

        parts = []
        if found:
            parts.append("Found: " + ", ".join(found) + ".")
        if missing:
            parts.append("Not found: " + ", ".join(missing) + " — set manually.")
        if not found and not missing:
            parts.append("Model scan folder not found — set it above.")
        self._autodetect_label.setText(" ".join(parts))
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

    def _check_torch_version(self):
        py = self._venv_python()
        if not py or not Path(py).is_file():
            self._torch_log.append("Could not find a training python interpreter. Check the install (.venv) or set the sd-scripts path.")
            return
        if self._torch_process is not None and self._torch_process.state() != QProcess.NotRunning:
            self._torch_log.append("A PyTorch operation is already running.")
            return
        self._torch_log.append("Checking PyTorch version…")
        proc = QProcess(self)
        apply_no_window(proc)  # no console window pop-up on Windows
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda p=proc: self._handle_torch_check_output(p))
        proc.start(py, ["-c", "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"])

    def _handle_torch_check_output(self, p):
        text = bytes(p.readAllStandardOutput()).decode("utf-8", "replace").strip()
        if text:
            self._torch_log.append(text)
        m = re.search(r"torch\s+(\d+)\.(\d+)", text)
        if m:
            self._pytorch_ok = (int(m.group(1)), int(m.group(2))) >= (2, 5)
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
            self._torch_log.append("✔ PyTorch upgrade finished. Click 'Check PyTorch Version' to confirm.")
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

    def _test_lmstudio(self):
        import json
        import urllib.request
        url = self.get_lmstudio_url().rstrip("/") + "/models"
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read().decode("utf-8"))
            ids = [m.get("id") for m in data.get("data", [])]
            if ids:
                self._lms_status.setText("✔ Connected. Loaded: " + ", ".join(str(i) for i in ids))
                self._lms_status.setStyleSheet("font-size: 11px; color: #d4af37;")
            else:
                self._lms_status.setText("Reachable, but no model loaded — load a vision model.")
                self._lms_status.setStyleSheet("font-size: 11px; color: #d4972b;")
        except Exception as e:
            self._lms_status.setText(f"✘ Cannot reach LM Studio: {e}")
            self._lms_status.setStyleSheet("font-size: 11px; color: #d9534f;")

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
        v.addWidget(self._forge_deliver_check)
        v.addWidget(self._forge_test_check)
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
            "Catches LM Studio / Forge holding the GPU.")
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
            "Leave Replace with empty to delete the term (that's how you ban a tag)."
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
        self._settings.setValue("output_dir", to_portable(self._output_edit.text()))
        self._settings.setValue("lmstudio_url", self._lms_url_edit.text())
        self._settings.setValue("lmstudio_model", self._lms_model_edit.text())
        self._settings.setValue("lmstudio_refine_in_process", self._lms_in_process_check.isChecked())
        self._settings.sync()

    def _load_settings(self):
        from core.paths import app_root, from_portable, in_other_install
        root = app_root()

        # sd-scripts: this install's bundled copy wins over an invalid path OR a path
        # that leaked from another AnimaForge copy on the same machine. A user's own
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

        default_out = str(root / "output")
        saved_out = from_portable(self._settings.value("output_dir", default_out, type=str))
        if not saved_out or in_other_install(saved_out):
            saved_out = default_out
        self._output_edit.setText(saved_out)

        self._lms_url_edit.setText(
            self._settings.value("lmstudio_url", "http://localhost:1234/v1", type=str)
            or "http://localhost:1234/v1"
        )
        self._lms_model_edit.setText(
            self._settings.value("lmstudio_model", "qwen2.5-vl-7b-instruct", type=str)
        )
        self._lms_in_process_check.setChecked(
            self._settings.value("lmstudio_refine_in_process", False, type=bool)
        )

    # ------------------------------------------------------------------
    # Install dialog
    # ------------------------------------------------------------------

    def _show_install_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("AnimaForge Setup Instructions")
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
            "Step 2 — sd-scripts (must support Anima)\n"
            "    git clone https://github.com/kohya-ss/sd-scripts.git\n"
            "    cd sd-scripts && git pull   (ensure anima_train_network.py exists)\n\n"
            "Step 3 — Virtual environment\n"
            "    python -m venv venv\n"
            "    venv\\Scripts\\activate\n\n"
            "Step 4 — Install PyTorch 2.5+ (Anima requires it; older = NaN loss)\n"
            "    pip install \"torch>=2.5\" torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "  RTX 50-series (Blackwell) cards need the CUDA 12.8 build instead:\n"
            "    pip install \"torch>=2.7\" torchvision --index-url https://download.pytorch.org/whl/cu128\n"
            "  RTX 50-series (Blackwell) cards need the CUDA 12.8 build instead:\n"
            "    pip install \"torch>=2.7\" torchvision --index-url https://download.pytorch.org/whl/cu128\n"
            "  (or use the 'Upgrade PyTorch' button above)\n\n"
            "Step 5 — sd-scripts dependencies\n"
            "    pip install -r requirements.txt\n"
            "    pip install prodigy-plus-schedule-free   (Prodigy+ ScheduleFree optimizer)\n"
            "    pip install accelerate && accelerate config\n\n"
            "Step 6 — Anima model files (or use Auto-detect from Forge Neo)\n"
            "    • Anima DiT checkpoint (e.g. anima_baseV10.safetensors)\n"
            "    • Qwen3 text encoder: qwen_3_06b_base.safetensors\n"
            "    • Qwen-Image VAE: qwen_image_vae.safetensors\n"
            "    From https://huggingface.co/circlestone-labs/Anima\n\n"
            "Step 7 — JoyCaption (natural-language captions)\n"
            "    Already uses transformers in the venv; the model downloads on first use.\n\n"
            "Note: xformers is NOT used by Anima (it uses SDPA).\n"
        )
        layout.addWidget(instructions)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec()
