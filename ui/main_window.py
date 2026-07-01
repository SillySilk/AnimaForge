from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ui.home_tab import HomeTab
from ui.setup_tab import SetupTab
from ui.dataset_tab import DatasetTab
from ui.characters_tab import CharactersTab
from ui.train_tab import TrainTab
from ui.batch_tab import BatchTab
from ui.progress_rail import ProgressRail
from utils.styles import asset_url


class NavButton(QPushButton):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("nav_button")
        self.setCheckable(False)
        self._selected = False
        self.setFixedHeight(48)
        self.setFlat(True)
        self.setCursor(Qt.PointingHandCursor)

    def set_selected(self, selected: bool):
        self._selected = selected
        self.setProperty("selected", "true" if selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Anima Forge LoRA Trainer with Auto-Captioning")
        self.setMinimumSize(1100, 720)
        self.resize(1280, 800)

        self._build_ui()
        self._connect_signals()

        # Initialise train tab environment + defaults from saved settings
        self._sync_environment_to_train()
        self._train_tab.apply_defaults(self._setup_tab.get_app_settings())

        # Offer to recover a run that was interrupted (crash/kill) last session
        from core import sets
        rd = sets.interrupted_run()
        if rd is not None:
            self._train_tab.show_recovery_banner(rd)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Central widget with horizontal layout
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ---- Sidebar ----
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(216)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # App branding
        brand_widget = QWidget()
        brand_widget.setObjectName("sidebar")
        brand_layout = QVBoxLayout(brand_widget)
        brand_layout.setContentsMargins(0, 0, 0, 0)
        brand_layout.setSpacing(0)

        emblem = QLabel()
        _pm = QPixmap(asset_url("emblem.png"))
        if not _pm.isNull():
            emblem.setPixmap(_pm.scaledToHeight(64, Qt.SmoothTransformation))
        emblem.setAlignment(Qt.AlignHCenter)
        emblem.setStyleSheet("background-color: #08080a; padding-top: 14px;")
        brand_layout.addWidget(emblem)

        app_title = QLabel("ANIMA FORGE")
        app_title.setObjectName("app_title")
        app_title.setAlignment(Qt.AlignLeft)
        brand_layout.addWidget(app_title)

        app_subtitle = QLabel("LoRA Trainer · Auto-Captioning")
        app_subtitle.setObjectName("app_subtitle")
        app_subtitle.setAlignment(Qt.AlignLeft)
        app_subtitle.setWordWrap(True)
        brand_layout.addWidget(app_subtitle)

        sidebar_layout.addWidget(brand_widget)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("color: #2a2a1e; margin: 0 8px;")
        sidebar_layout.addWidget(div)

        # Primary nav buttons. Setup is intentionally NOT here — it's a set-once concern,
        # reached via the ⚙ Settings button pinned at the bottom (stack index 1).
        self._nav_buttons = []  # list of (NavButton, stack_index)
        nav_items = [
            ("  Home", 0, "home"),
            ("  Dataset", 2, "dataset"),
            ("  Characters", 3, "characters"),
            ("  Train", 4, "train"),
            ("  Batch", 5, "batch"),
        ]
        for label, index, icon_name in nav_items:
            btn = NavButton(label)
            _icon = QIcon(asset_url(f"nav/{icon_name}.png"))
            if not _icon.isNull():
                btn.setIcon(_icon)
                btn.setIconSize(QSize(22, 22))
            btn.clicked.connect(lambda checked=False, i=index: self._switch_tab(i))
            sidebar_layout.addWidget(btn)
            self._nav_buttons.append((btn, index))

        sidebar_layout.addStretch()

        # ⚙ Settings (the demoted Setup tab) pinned at the bottom.
        gear = NavButton("  ⚙ Settings")
        _gear_icon = QIcon(asset_url("nav/setup.png"))
        if not _gear_icon.isNull():
            gear.setIcon(_gear_icon)
            gear.setIconSize(QSize(22, 22))
        gear.clicked.connect(lambda checked=False: self._switch_tab(1))
        sidebar_layout.addWidget(gear)
        self._nav_buttons.append((gear, 1))

        # Version label at bottom of sidebar
        ver_label = QLabel("v1.0.0")
        ver_label.setAlignment(Qt.AlignCenter)
        ver_label.setStyleSheet("color: #4a4a44; font-size: 10px; padding: 8px;")
        sidebar_layout.addWidget(ver_label)

        root_layout.addWidget(sidebar)

        # ---- Content area (ember backdrop + header bar + stack) ----
        content = QWidget()
        content.setObjectName("app_bg")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        header = QWidget()
        header.setObjectName("app_header")
        hb = QHBoxLayout(header)
        hb.setContentsMargins(0, 0, 0, 0)
        self._header_title = QLabel("Home")
        self._header_title.setObjectName("header_title")
        hb.addWidget(self._header_title)
        hb.addStretch()
        content_layout.addWidget(header)

        # Global Load → Name → Caption → Train rail (advisory orientation + navigation).
        self._rail = ProgressRail()
        content_layout.addWidget(self._rail)

        self._stack = QStackedWidget()
        content_layout.addWidget(self._stack)
        root_layout.addWidget(content)

        # Create tabs
        self._home_tab = HomeTab()
        self._setup_tab = SetupTab()
        self._dataset_tab = DatasetTab()
        self._characters_tab = CharactersTab()
        self._train_tab = TrainTab()
        self._batch_tab = BatchTab()

        self._stack.addWidget(self._home_tab)        # index 0
        self._stack.addWidget(self._setup_tab)       # index 1
        self._stack.addWidget(self._dataset_tab)     # index 2
        self._stack.addWidget(self._characters_tab)  # index 3
        self._stack.addWidget(self._train_tab)       # index 4
        self._stack.addWidget(self._batch_tab)       # index 5

        # ---- Status bar ----
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

        # Select first tab
        self._switch_tab(0)

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        # When dataset is loaded, sync to train tab
        self._dataset_tab.dataset_loaded.connect(self._on_dataset_loaded)

        # Progress rail: navigation + readiness refresh on every step-changing event.
        self._rail.navigate.connect(self._on_rail_navigate)
        self._dataset_tab.dataset_loaded.connect(lambda *_: self._refresh_rail())
        self._dataset_tab.characters_changed.connect(self._refresh_rail)
        self._dataset_tab.caption_finished.connect(self._refresh_rail)
        self._characters_tab.characters_changed.connect(self._refresh_rail)
        self._train_tab.subject_type_changed.connect(self._refresh_rail)

        # When setup settings change, sync to train tab
        self._setup_tab.settings_changed.connect(self._sync_environment_to_train)

        # Train tab status messages
        self._train_tab.status_message.connect(self._status_bar.showMessage)

        # Add-to-batch from the Train tab populates the Batch queue
        self._train_tab.add_to_batch_requested.connect(self._batch_tab.add_run)

        # Loading a set restores its dataset + trigger into the Dataset tab
        self._train_tab.load_set_requested.connect(self._on_load_set_requested)

        # Characters tab <-> Dataset tab two-way resync (shared per-folder JSON)
        self._characters_tab.characters_changed.connect(self._dataset_tab.reload_characters)
        self._dataset_tab.characters_changed.connect(self._characters_tab.reload_characters)
        self._characters_tab.status_message.connect(self._status_bar.showMessage)
        # Fix-names on the Characters tab re-combines the .txt so names land in the prompt
        self._characters_tab.names_validated.connect(self._dataset_tab.rebuild_captions_after_naming)
        # "Name your characters?" prompt after tagging jumps to the Characters tab
        self._dataset_tab.open_characters_requested.connect(lambda: self._switch_tab(3))

        # Dashboard quick actions
        self._home_tab.navigate.connect(self._switch_tab)
        self._home_tab.autodetect_requested.connect(self._on_home_autodetect)
        self._home_tab.recover_requested.connect(self._on_home_recover)

        # Home "Quick Run" cockpit → drive the real tabs (Train owns the relocated widgets)
        self._home_tab.folder_chosen.connect(self._on_home_folder_chosen)
        self._home_tab.name_changed.connect(self._train_tab.set_lora_name)
        self._home_tab.trigger_changed.connect(self._on_home_trigger_changed)
        self._home_tab.prefix_changed.connect(self._dataset_tab.set_prefix)
        self._home_tab.type_changed.connect(self._on_home_type_changed)
        self._home_tab.anchor_changed.connect(self._characters_tab.set_style_anchor)
        self._home_tab.run_requested.connect(self._on_home_run)
        # The Step Calculator lives on Home now; its subject combo drives the Style @anchor
        # field's visibility (only meaningful for Style runs).
        self._train_tab.subject_type_changed.connect(self._sync_home_anchor_visibility)
        # Caption + training controls are relocated from the Dataset/Train tabs onto Home
        # (single source of truth). Their buttons stay wired to their owning tab's engine, so
        # mounting is a display-only move — no rewiring. The back tabs keep only their
        # previews (Dataset: gallery; Train: progress + log + sample images).
        self._home_tab.mount_caption_controls(self._dataset_tab.caption_controls())
        self._home_tab.mount_step_calculator(self._train_tab.step_calculator())
        self._home_tab.mount_train_controls(self._train_tab.control_panel())
        # Mirror Train's progress onto Home's identical RunProgress widget
        self._train_tab.run_progress.connect(self._home_tab.apply_run_progress)
        # Headless caption chain (Home pipeline) completion → advance the pipeline
        self._dataset_tab.auto_caption_finished.connect(self._qr_caption_done)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _switch_tab(self, index: int):
        self._stack.setCurrentIndex(index)
        for btn, idx in self._nav_buttons:
            btn.set_selected(idx == index)

        tab_names = ["Home", "Settings", "Dataset", "Characters", "Train", "Batch"]
        self._header_title.setText(tab_names[index])
        self._status_bar.showMessage(f"{tab_names[index]} tab active")
        if index == 0:
            self._home_tab.refresh(self._collect_home_context())
        self._refresh_rail()

    def _refresh_rail(self):
        """Recompute the progress rail from the active dataset folder + current tab."""
        from core import workflow
        folder = self._dataset_tab.get_folder_path()
        load = workflow.dataset_state(folder)
        name = workflow.naming_state(folder)
        name["applicable"] = not self._train_tab.is_style_subject()
        caption = workflow.caption_state(folder)
        idx = self._stack.currentIndex()
        if idx == 4:
            current = "train"
        elif idx == 3:
            current = "name"
        elif idx == 2:
            current = "load" if not load["done"] else "caption"
        else:
            current = None
        self._rail.set_state({"load": load, "name": name,
                              "caption": caption, "current": current})

    def _on_rail_navigate(self, key: str):
        """Rail segment clicked: Load/Caption → Dataset, Name → Name Cast, Train → Train."""
        if key in ("load", "caption"):
            self._switch_tab(2)
        elif key == "name":
            if self._dataset_tab.get_folder_path():
                self._dataset_tab._open_name_validator()
            else:
                self._switch_tab(2)
        elif key == "train":
            self._switch_tab(4)

    def _collect_home_context(self):
        s = self._setup_tab
        try:
            torch_ok = bool(s.is_pytorch_ok())
        except Exception:
            torch_ok = False
        return {
            "sdscripts": s.get_sdscripts_path(),
            "dit": s.get_dit_path(),
            "qwen3": s.get_qwen3_path(),
            "vae": s.get_vae_path(),
            "output": s.get_output_dir(),
            "torch_ok": torch_ok,
            "dataset_folder": self._dataset_tab.get_folder_path(),
            "image_count": self._dataset_tab.get_image_count(),
            "lms_url": s.get_lmstudio_url(),
            "lms_ok": None,
            # Quick Run cockpit mirror (Train tab is the source of truth)
            "lora_name": self._train_tab.get_lora_name(),
            "trigger_word": self._dataset_tab.get_trigger_word(),
            "quality_prefix": self._dataset_tab.get_prefix(),
            "subject_type": self._train_tab.get_subject_type(),
            "target_steps": self._train_tab.get_target_steps(),
            "style_anchor": self._train_tab._dataset_style_anchor(),
        }

    def _on_home_folder_chosen(self, folder: str):
        """Home picked a dataset folder — load it (cascades into Train, whose Step Calculator
        is shown on Home and auto-recalculates the suggested steps)."""
        if folder:
            self._dataset_tab.load_folder_path(folder)

    def _on_home_trigger_changed(self, trigger: str):
        """Front-page trigger edit → the Dataset tab (source of truth) + Train mirror."""
        self._dataset_tab.set_trigger_word(trigger)
        self._train_tab.set_trigger_word(trigger)

    def _on_home_type_changed(self, key: str):
        """Filename auto-detect on Home picked a subject type → drive the relocated Step
        Calculator (Train owns it); its subject_type_changed signal updates anchor visibility."""
        self._train_tab.set_subject_type(key)

    def _sync_home_anchor_visibility(self):
        """Show Home's Style @anchor field only when the relocated Step Calculator is on Style."""
        self._home_tab.set_style_anchor_visible(self._train_tab.is_style_subject())

    def _on_home_run(self):
        """Home Run → unattended pipeline: (detect names) → (caption) → train.

        Phases are decided up front (pure planner) then executed one at a time. The chain
        runs hands-off and only interrupts with a dialog on error.
        """
        from core import workflow, quick_run, characters as ch
        folder = self._dataset_tab.get_folder_path()
        if not folder:
            QMessageBox.warning(self, "No dataset", "Choose a dataset folder on Home first.")
            return
        if not self._train_tab.get_lora_name():
            QMessageBox.warning(self, "Name required",
                                "Name the LoRA first (Home → LoRA name).")
            return
        subject = self._train_tab.get_subject_type()
        has_roster = any(c.token.strip() for c in ch.load(folder).roster)
        captioned = bool(workflow.caption_state(folder).get("done"))
        self._qr_phases = quick_run.plan_phases(subject, has_roster, captioned)
        self._home_tab.apply_run_progress({"kind": "reset"})  # zero the bar before each run
        self._switch_tab(4)  # surface the live training log + sample previews
        self._qr_advance()

    def _on_home_caption(self):
        folder = self._dataset_tab.get_folder_path()
        if not folder:
            QMessageBox.warning(self, "No dataset", "Choose a dataset folder on Home first.")
            return
        self._home_tab.apply_run_progress({"kind": "reset"})
        self._switch_tab(2)  # Dataset tab — its caption log + progress
        self._dataset_tab.start_auto_caption()

    def _on_home_train(self):
        folder = self._dataset_tab.get_folder_path()
        if not folder:
            QMessageBox.warning(self, "No dataset", "Choose a dataset folder on Home first.")
            return
        if not self._train_tab.get_lora_name():
            QMessageBox.warning(self, "Name required", "Name the LoRA first (Home → LoRA name).")
            return
        self._home_tab.apply_run_progress({"kind": "reset"})
        self._switch_tab(4)  # Train tab — its progress mirrors to Home
        self._train_tab.start_from_cockpit()

    def _qr_advance(self):
        """Execute the next Quick Run phase. Synchronous phases recurse; the caption
        phase returns and resumes from auto_caption_finished → _qr_caption_done."""
        from core import quick_run
        phases = getattr(self, "_qr_phases", [])
        if not phases:
            return
        phase = phases[0]
        if phase == quick_run.DETECT:
            self._home_tab.apply_run_progress({"kind": "indeterminate", "label": "Detecting names…"})
            self._characters_tab.auto_detect_from_filenames()
            self._qr_phases.pop(0)
            self._qr_advance()
        elif phase == quick_run.CAPTION:
            self._home_tab.apply_run_progress({"kind": "indeterminate", "label": "Captioning…"})
            self._status_bar.showMessage("Quick Run: captioning dataset…")
            if not self._dataset_tab.start_auto_caption():
                self._qr_phases = []
                self._home_tab.apply_run_progress({"kind": "error", "label": "Caption error"})
                QMessageBox.warning(
                    self, "Cannot caption",
                    "Could not start captioning. Check the sd-scripts path (Settings) and "
                    "that the dataset folder has images.")
            # else: wait for auto_caption_finished → _qr_caption_done
        elif phase == quick_run.TRAIN:
            self._qr_phases.pop(0)
            self._train_tab.start_from_cockpit()

    def _qr_caption_done(self, success: bool):
        from core import quick_run
        phases = getattr(self, "_qr_phases", [])
        if not phases or phases[0] != quick_run.CAPTION:
            return  # not a pipeline-driven caption run
        if not success:
            self._qr_phases = []
            self._home_tab.apply_run_progress({"kind": "error", "label": "Caption failed"})
            QMessageBox.warning(
                self, "Captioning failed",
                "Captioning stopped with an error. Open the Dataset tab to review the log.")
            return
        self._qr_phases.pop(0)
        self._qr_advance()

    def _on_home_autodetect(self):
        """Run the model scan, then show Settings (where the results land) + a status summary.

        Previously this fired the scan silently while the user stood on Home, so the filled-in
        fields and found/missing summary were invisible. Jumping to Settings makes it land.
        """
        self._setup_tab._auto_detect_models()
        self._switch_tab(1)
        msg = self._setup_tab._autodetect_label.text().strip() or "Model auto-detect complete."
        self._status_bar.showMessage(msg, 8000)

    def _on_home_recover(self):
        from core import sets
        rd = sets.interrupted_run()
        if rd is not None:
            self._switch_tab(4)
            self._train_tab.show_recovery_banner(rd)

    def _on_load_set_requested(self, dataset_folder: str, trigger: str):
        if dataset_folder:
            self._dataset_tab.load_folder_path(dataset_folder)
        if trigger:
            self._dataset_tab.set_trigger_word(trigger)

    def _on_dataset_loaded(self, folder_path: str, image_count: int):
        self._train_tab.set_dataset(folder_path, image_count)
        self._train_tab.set_trigger_word(self._dataset_tab.get_trigger_word())
        self._characters_tab.set_dataset(folder_path)
        self._status_bar.showMessage(
            f"Dataset loaded: {image_count} images from {folder_path}"
        )

    def _sync_environment_to_train(self):
        sd_path = self._setup_tab.get_sdscripts_path()
        self._train_tab.set_environment(
            sdscripts_path=sd_path,
            dit_path=self._setup_tab.get_dit_path(),
            qwen3_path=self._setup_tab.get_qwen3_path(),
            vae_path=self._setup_tab.get_vae_path(),
            output_dir=self._setup_tab.get_output_dir(),
        )
        self._dataset_tab.set_sdscripts_path(sd_path)
        lms_url = self._setup_tab.get_lmstudio_url()
        lms_model = self._setup_tab.get_lmstudio_model()
        self._dataset_tab.set_lmstudio_config(lms_url, lms_model)
        self._train_tab.set_lmstudio_config(lms_url, lms_model)
        # Reflect the LM Studio Refine-in-Process toggle on the Dataset tab live
        self._dataset_tab._refresh_refine_reflection()
