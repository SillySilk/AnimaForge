from PySide6.QtCore import Qt, QSize, QSettings, Signal
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
    """Forge-shell nav item: icon + typewriter label, 46px tall, collapsible.

    Collapsing hides the label and centres the icon so the sidebar becomes a
    76px icon-only rail.
    """

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("af_nav")
        self.setCheckable(False)
        self._selected = False
        self._label = text
        self.setFixedHeight(46)
        self.setFlat(True)
        self.setCursor(Qt.PointingHandCursor)

    def set_selected(self, selected: bool):
        self._selected = selected
        self.setProperty("selected", "true" if selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_collapsed(self, collapsed: bool):
        self.setText("" if collapsed else self._label)
        self.setToolTip(self._label.strip() if collapsed else "")


class MainWindow(QMainWindow):
    # Updater worker threads → UI (queued back onto the main thread)
    _update_check_done = Signal(str)            # remote version ("" = unreachable)
    _update_apply_done = Signal(str, bool, str)  # (version, requirements_changed, error)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Anima Forge LoRA Trainer with Auto-Captioning")
        self.setMinimumSize(1100, 720)
        self.resize(1280, 800)
        self._update_busy = False

        from core import train_presets
        self._preset_name = train_presets.DEFAULT_NAME  # "Person"

        self._build_ui()
        self._connect_signals()

        # Initialise train tab environment + defaults from saved settings
        self._sync_environment_to_train()
        self._train_tab.apply_defaults(self._setup_tab.get_app_settings())

        # Offer to recover a run that was interrupted (crash/kill) last session.
        # The per-run output/<lora>/run.json manifest is tried first (it survives a
        # copied/moved output folder or a second run started on top of an old one);
        # sets.interrupted_run()'s single global marker is the fallback for installs
        # that predate the manifest.
        from core import run_manifest, sets
        rd = run_manifest.find_resumable(self._setup_tab.get_output_dir())
        if rd is None:
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

        # ---- Sidebar (forge shell — flame emblem, blackletter wordmark, nav,
        #      decor block, pinned Setup + version; collapses to a 76px rail) ----
        sidebar = QWidget()
        sidebar.setObjectName("af_sidebar")
        sidebar.setFixedWidth(250)
        self._sidebar = sidebar
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Collapse toggle — top-right of the sidebar (flips ‹ / ›).
        collapse_row = QWidget()
        cr = QHBoxLayout(collapse_row)
        cr.setContentsMargins(0, 11, 11, 0)
        cr.addStretch()
        self._collapse_btn = QPushButton("‹")
        self._collapse_btn.setObjectName("af_collapse_btn")
        self._collapse_btn.setFixedSize(26, 26)
        self._collapse_btn.setCursor(Qt.PointingHandCursor)
        self._collapse_btn.clicked.connect(self._toggle_sidebar)
        cr.addWidget(self._collapse_btn)
        sidebar_layout.addWidget(collapse_row)

        # Branding — emblem + blackletter wordmark + "Heretics Only" eyebrow.
        brand_widget = QWidget()
        brand_widget.setObjectName("af_sidebar")
        brand_layout = QVBoxLayout(brand_widget)
        brand_layout.setContentsMargins(0, 6, 0, 12)
        brand_layout.setSpacing(6)

        # The brand badge fills the sidebar's full width (one badge per page; the
        # decor seal further down uses the older drawn forge mark instead).
        self._emblem = QLabel()
        self._emblem_pm = QPixmap(asset_url("emblem.png"))
        if not self._emblem_pm.isNull():
            self._emblem.setPixmap(self._emblem_pm.scaledToWidth(250, Qt.SmoothTransformation))
        self._emblem.setAlignment(Qt.AlignHCenter)
        self._emblem.setStyleSheet("background-color: transparent;")
        brand_layout.addWidget(self._emblem)

        self._wordmark = QLabel("AnimaForge")
        self._wordmark.setObjectName("af_wordmark")
        self._wordmark.setAlignment(Qt.AlignHCenter)
        brand_layout.addWidget(self._wordmark)

        self._brand_eyebrow = QLabel("HERETICS ONLY")
        self._brand_eyebrow.setObjectName("af_eyebrow")
        self._brand_eyebrow.setAlignment(Qt.AlignHCenter)
        brand_layout.addWidget(self._brand_eyebrow)

        sidebar_layout.addWidget(brand_widget)

        # Gold hairline rule.
        self._brand_rule = QFrame()
        self._brand_rule.setObjectName("af_rule")
        self._brand_rule.setFixedHeight(2)
        rule_wrap = QWidget()
        rw = QHBoxLayout(rule_wrap)
        rw.setContentsMargins(18, 6, 18, 12)
        rw.addWidget(self._brand_rule)
        sidebar_layout.addWidget(rule_wrap)

        # Primary nav. Setup is pinned at the bottom (set-once concern), stack index 1.
        self._nav_buttons = []  # list of (NavButton, stack_index)
        nav_wrap = QWidget()
        nav_wrap.setObjectName("af_sidebar")
        nav_layout = QVBoxLayout(nav_wrap)
        nav_layout.setContentsMargins(10, 0, 10, 0)
        nav_layout.setSpacing(2)
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
                btn.setIconSize(QSize(18, 18))
            btn.clicked.connect(lambda checked=False, i=index: self._switch_tab(i))
            nav_layout.addWidget(btn)
            self._nav_buttons.append((btn, index))

        # Presets — not a screen: jumps Home and opens the Train Presets modal
        # (optimizer preset, sample previews, run options). Second way into the same
        # Home-owned panel, so the single-source-of-truth rule holds.
        presets_btn = NavButton("  Presets")
        _presets_icon = QIcon(asset_url("nav/presets.png"))
        if not _presets_icon.isNull():
            presets_btn.setIcon(_presets_icon)
            presets_btn.setIconSize(QSize(18, 18))
        presets_btn.clicked.connect(self._open_presets_modal)
        nav_layout.addWidget(presets_btn)
        sidebar_layout.addWidget(nav_wrap)

        # Decor block — marker scrawl + wax-seal emblem + members line.
        self._decor = QWidget()
        self._decor.setObjectName("af_sidebar")
        decor_layout = QVBoxLayout(self._decor)
        decor_layout.setContentsMargins(24, 22, 24, 8)
        decor_layout.setSpacing(12)
        quote = QLabel("“Forge it in fire.\nWalk away.”")
        quote.setObjectName("af_decor_quote")
        quote.setWordWrap(True)
        decor_layout.addWidget(quote)
        seal = QLabel()
        seal_pm = QPixmap(asset_url("forge_seal.png"))  # the old flame mark, not the badge
        if not seal_pm.isNull():
            seal.setPixmap(seal_pm.scaledToHeight(46, Qt.SmoothTransformation))
        seal.setAlignment(Qt.AlignHCenter)
        seal.setStyleSheet("background-color: transparent;")
        decor_layout.addWidget(seal)
        members = QLabel("MEMBERS · EST. NOWHERE")
        members.setObjectName("af_decor_meta")
        members.setAlignment(Qt.AlignHCenter)
        members.setWordWrap(True)
        decor_layout.addWidget(members)
        sidebar_layout.addWidget(self._decor)

        sidebar_layout.addStretch()

        # Pinned bottom — Setup nav + version stamp.
        btm_wrap = QWidget()
        btm_wrap.setObjectName("af_sidebar")
        btm_layout = QVBoxLayout(btm_wrap)
        btm_layout.setContentsMargins(10, 0, 10, 10)
        btm_layout.setSpacing(2)
        gear = NavButton("  Setup")
        _gear_icon = QIcon(asset_url("nav/setup.png"))
        if not _gear_icon.isNull():
            gear.setIcon(_gear_icon)
            gear.setIconSize(QSize(18, 18))
        gear.clicked.connect(lambda checked=False: self._switch_tab(1))
        btm_layout.addWidget(gear)
        self._nav_buttons.append((gear, 1))

        from core.version import __version__
        self._ver_label = QLabel(f"v{__version__} · EST. NOWHERE")
        self._ver_label.setObjectName("af_ver")
        self._ver_label.setAlignment(Qt.AlignCenter)
        btm_layout.addWidget(self._ver_label)
        sidebar_layout.addWidget(btm_wrap)

        root_layout.addWidget(sidebar)

        # Restore the persisted collapse state.
        self._sidebar_collapsed = False
        try:
            self._sidebar_collapsed = QSettings().value(
                "ui/sidebar_collapsed", False, type=bool)
        except Exception:
            self._sidebar_collapsed = False

        # ---- Content area (ember backdrop + header bar + stack) ----
        content = QWidget()
        content.setObjectName("app_bg")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        header = QWidget()
        header.setObjectName("app_header")
        header.setFixedHeight(66)
        hb = QHBoxLayout(header)
        hb.setContentsMargins(34, 0, 34, 0)
        hb.setSpacing(14)
        self._header_title = QLabel("Home")
        self._header_title.setObjectName("af_screen_title")
        hb.addWidget(self._header_title, 0, Qt.AlignVCenter)
        self._header_eyebrow = QLabel("THE BENCH")
        self._header_eyebrow.setObjectName("af_screen_eyebrow")
        hb.addWidget(self._header_eyebrow, 0, Qt.AlignBottom)
        self._header_eyebrow.setContentsMargins(0, 0, 0, 16)
        hb.addStretch()
        content_layout.addWidget(header)

        # Global Load → Name → Caption → Train rail (advisory orientation + navigation).
        # The forge redesign folds this orientation into the sidebar nav + per-screen
        # header eyebrows, so the rail is kept (wiring intact) but hidden.
        self._rail = ProgressRail()
        self._rail.setVisible(False)
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

        # Apply any persisted sidebar collapse state.
        self._apply_sidebar_collapsed()

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
        # Fresh captions → fill the (empty) sample-prompt box so previews launch with
        # real dataset prompts; load-time autofill fires before captions exist.
        self._dataset_tab.caption_finished.connect(
            self._train_tab.refresh_sample_prompts_from_captions)
        self._dataset_tab.auto_caption_finished.connect(
            self._train_tab.refresh_sample_prompts_from_captions)
        # Home's stage chips: per-caption ticks while an engine runs, authoritative
        # sidecar counts the moment each step lands.
        self._dataset_tab.caption_tick.connect(self._home_tab.apply_caption_tick)
        self._dataset_tab.caption_finished.connect(
            lambda: self._home_tab.set_stage_counts(
                *self._dataset_tab.caption_stage_counts()))
        self._characters_tab.characters_changed.connect(self._refresh_rail)
        self._train_tab.subject_type_changed.connect(self._refresh_rail)
        # Live OPTIMIZER tile: the Train Presets modal floats over Home, so reflect
        # preset flips immediately rather than waiting for the next full refresh.
        self._train_tab.optimizer_changed.connect(
            lambda label: self._home_tab.set_train_summary(optimizer=label))
        # Closing a Presets/Step Calculator modal re-pulls the whole Home summary
        # (steps, dim/alpha, optimizer) — it used to go stale until the next tab switch.
        self._home_tab.presets_closed.connect(
            lambda: self._home_tab.refresh(self._collect_home_context()))

        # When setup settings change, sync to train tab
        self._setup_tab.settings_changed.connect(self._sync_environment_to_train)

        # Train tab status messages
        self._train_tab.status_message.connect(self._status_bar.showMessage)

        # Add-to-batch from the Train tab populates the Batch queue
        self._train_tab.add_to_batch_requested.connect(self._batch_tab.add_run)

        # Home's "➕ Add to Batch" (Lever band) queues the current cockpit-mirrored run
        self._home_tab.add_to_batch_requested.connect(self._train_tab.add_current_to_batch)

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

        # PRESET button on the Set card → the picker (MainWindow owns preset data)
        self._home_tab.preset_pick_requested.connect(self._open_preset_picker)
        # 📜 beside Start → the same config preview the Options modal offers
        self._home_tab.preview_config_requested.connect(self._train_tab._preview_config)

        # Dashboard quick actions
        self._home_tab.navigate.connect(self._switch_tab)
        self._home_tab.autodetect_requested.connect(self._on_home_autodetect)
        self._home_tab.recover_requested.connect(self._on_home_recover)
        self._home_tab.update_check_requested.connect(self._on_update_check)
        self._update_check_done.connect(self._show_update_result)
        self._update_apply_done.connect(self._on_update_applied)

        # Home "Quick Run" cockpit → drive the real tabs (Train owns the relocated widgets)
        self._home_tab.folder_chosen.connect(self._on_home_folder_chosen)
        self._home_tab.name_changed.connect(self._train_tab.set_lora_name)
        self._home_tab.trigger_changed.connect(self._on_home_trigger_changed)
        self._home_tab.prefix_changed.connect(self._dataset_tab.set_prefix)
        self._home_tab.prefix_changed.connect(self._train_tab.set_quality_prefix)
        self._home_tab.type_changed.connect(self._on_home_type_changed)
        self._home_tab.anchor_changed.connect(self._characters_tab.set_style_anchor)
        self._home_tab.run_requested.connect(self._on_home_run)
        # Pillar primary buttons drive the real engines (options live in the modals).
        self._home_tab.run_caption_requested.connect(self._dataset_tab._process_clicked)
        self._home_tab.start_train_requested.connect(
            lambda: self._train_tab._start_training(confirm=True))
        # Stop rides next to Start on the front; Train's engine owns the confirm dialog.
        self._home_tab.stop_train_requested.connect(self._train_tab._stop_training)
        self._train_tab.training_active.connect(self._home_tab.set_training_active)
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
        # Caption milestones → autosave the project under the LoRA name
        self._dataset_tab.caption_stage_done.connect(self._on_caption_stage_done)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Sidebar collapse
    # ------------------------------------------------------------------

    def _toggle_sidebar(self):
        self._sidebar_collapsed = not self._sidebar_collapsed
        self._apply_sidebar_collapsed()
        try:
            QSettings().setValue("ui/sidebar_collapsed", self._sidebar_collapsed)
        except Exception:
            pass

    def _apply_sidebar_collapsed(self):
        """Switch the sidebar between the 250px full panel and the 76px icon rail."""
        collapsed = self._sidebar_collapsed
        self._sidebar.setFixedWidth(76 if collapsed else 250)
        self._collapse_btn.setText("›" if collapsed else "‹")
        self._collapse_btn.setToolTip("Expand sidebar" if collapsed else "Collapse sidebar")
        for widget in (self._wordmark, self._brand_eyebrow, self._brand_rule,
                       self._decor, self._ver_label):
            widget.setVisible(not collapsed)
        # Shrink the emblem and hide nav labels in the collapsed rail.
        if not self._emblem_pm.isNull():
            self._emblem.setPixmap(self._emblem_pm.scaledToWidth(
                64 if collapsed else 250, Qt.SmoothTransformation))
        for btn, _ in self._nav_buttons:
            btn.set_collapsed(collapsed)

    def _switch_tab(self, index: int):
        self._stack.setCurrentIndex(index)
        for btn, idx in self._nav_buttons:
            btn.set_selected(idx == index)

        tab_names = ["Home", "Setup", "Dataset", "Characters", "Train", "Batch"]
        eyebrows = ["The Bench", "The Workshop", "The Cutting Room",
                    "The Roster", "The Furnace", "The Line"]
        self._header_title.setText(tab_names[index])
        self._header_eyebrow.setText(eyebrows[index].upper())
        self._status_bar.showMessage(f"{tab_names[index]} tab active")
        if index == 0:
            self._home_tab.refresh(self._collect_home_context())
        self._refresh_rail()

    def _open_presets_modal(self):
        """Sidebar Presets item: jump to Home and open the Training Presets picker."""
        self._switch_tab(0)
        self._open_preset_picker()

    # ------------------------------------------------------------------
    # Training presets (intent profiles)
    # ------------------------------------------------------------------

    def _presets_store(self) -> str:
        return self._setup_tab.get_app_settings().get("train_presets_json")

    def _open_preset_picker(self):
        """List + explicit Select — nothing applies from hovering or scrolling
        (an earlier combo flipped values on stray mouse-wheel; radios that replaced
        it couldn't grow). Double-click also selects (still a deliberate act)."""
        from core import train_presets as tp
        from ui.forge_modal import ForgeModal
        from PySide6.QtWidgets import QListWidget, QListWidgetItem
        presets = tp.all_presets(self._presets_store())
        modal = ForgeModal(
            self, title="Training Presets", eyebrow="The Set · Preset",
            subtitle="Pick the training intent — nothing changes until you hit Select.",
            max_width=560)
        lst = QListWidget()
        lst.setCursor(Qt.PointingHandCursor)
        for p in presets:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, p.name)
            # Three-deck row: name / settings / the step math in small print. The
            # built-ins read identically without that last line — the exposure
            # formula IS the difference between Person, Object and Style.
            row = QWidget()
            rv = QVBoxLayout(row)
            rv.setContentsMargins(10, 7, 10, 7)
            rv.setSpacing(1)
            name_lbl = QLabel(p.name)
            name_lbl.setStyleSheet(
                "background: transparent; color: #e8e0c8; font-size: 14px; font-weight: 700;")
            summ_lbl = QLabel(tp.summary_line(p))
            summ_lbl.setStyleSheet(
                "background: transparent; color: #8a8a93; font-size: 11px;")
            math_lbl = QLabel(tp.formula_line(p))
            math_lbl.setStyleSheet(
                "background: transparent; color: #a8925a; font-size: 10px; font-style: italic;")
            for lbl in (name_lbl, summ_lbl, math_lbl):
                rv.addWidget(lbl)
            row.setStyleSheet("background: transparent;")
            item.setSizeHint(row.sizeHint())
            lst.addItem(item)
            lst.setItemWidget(item, row)
            if p.name == self._preset_name:
                lst.setCurrentItem(item)
        lst.setMinimumHeight(min(420, 68 * lst.count() + 12))
        modal.body.addWidget(lst)
        hint = QLabel("Add your own in Setup → Training Presets.")
        hint.setObjectName("af_eyebrow_mute")
        modal.body.addWidget(hint)
        # Bridge to the deeper knobs (optimizer/network/sample/run options) — a tester
        # looked for them here first. Same Home-owned modal, one click away.
        options_btn = modal.add_footer_button("Train Options…")
        options_btn.clicked.connect(
            lambda: (modal.close_modal(), self._home_tab.open_train_presets()))
        modal.add_footer_button("Cancel").clicked.connect(modal.close_modal)
        select_btn = modal.add_footer_button("Select", primary=True)

        def _select():
            item = lst.currentItem()
            if item is not None:
                self._apply_preset_by_name(item.data(Qt.UserRole))
            modal.close_modal()

        select_btn.clicked.connect(_select)
        lst.itemDoubleClicked.connect(lambda _i: _select())
        modal.open()

    def _apply_preset_by_name(self, name: str):
        from core import train_presets as tp
        p = tp.find(self._presets_store(), name)
        if p is None:
            return
        self._preset_name = p.name
        self._train_tab.apply_preset(p)
        self._home_tab.set_preset_label(p.name, p.subject_type)
        self._home_tab.refresh(self._collect_home_context())
        self._status_bar.showMessage(f"Preset: {p.name} — {tp.summary_line(p)}", 6000)

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
            "preset_name": self._preset_name,
            "target_steps": self._train_tab.get_target_steps(),
            "style_anchor": self._train_tab._dataset_style_anchor(),
            # live pillar readouts (Home condenses these into the two step cards)
            "caption_stage_counts": self._dataset_tab.caption_stage_counts(),
            "optimizer_label": self._train_tab.optimizer_label(),
            "net_dim": self._train_tab._dim_spin.value(),
            "net_alpha": self._train_tab._alpha_spin.value(),
            "net_res": 1024,
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
        """Show the Style @anchor (Home + Characters) only when the subject type is Style.

        Also keeps the PRESET button honest: when the subject drifts away from the
        active preset (filename auto-detect, a loaded Saved Set), the label falls back
        to the built-in intent for that subject rather than lying about what's applied.
        """
        from core import train_presets as tp
        is_style = self._train_tab.is_style_subject()
        self._home_tab.set_style_anchor_visible(is_style)
        subject = self._train_tab.get_subject_type()
        current = tp.find(self._presets_store(), self._preset_name)
        if current is None or current.subject_type != subject:
            self._preset_name = tp.builtin_for_subject(subject).name
        self._home_tab.set_preset_label(self._preset_name, subject)
        self._characters_tab.set_anchor_gate(is_style)

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
                if self._dataset_tab.start_cancelled_by_user():
                    # The user clicked Cancel in the existing-captions dialog — not an
                    # error, so no popup and no "Caption error" label.
                    self._home_tab.apply_run_progress({"kind": "reset"})
                    self._status_bar.showMessage("Quick Run: captioning cancelled.")
                else:
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

    def _on_caption_stage_done(self, stage: str):
        """Autosave the project (Set + caption snapshot) named after the LoRA.

        Fires at two milestones of every caption chain — 'captioned' (raw passes
        done) and 'processed' (final .txt built) — so a failure or a later image
        readjustment can be rolled back. Silent besides a status-bar note; must
        never interrupt the running chain."""
        from core import sets
        name = self._train_tab.get_lora_name()
        folder = self._dataset_tab.get_folder_path()
        if not name or not folder:
            self._status_bar.showMessage(
                "Autosave skipped — name the LoRA to enable project autosave.", 6000)
            return
        rd, msg = self._train_tab.build_run_definition()
        if rd is None:
            # Settings aren't complete enough for a Set (e.g. model paths missing)
            # — still keep the captions safe.
            n = sets.snapshot_captions(folder, name, stage)
            self._status_bar.showMessage(
                f"Captions autosaved ('{name}', {stage} — {n} file(s)); "
                f"settings not saved: {msg}", 8000)
            return
        ok, note = sets.autosave_project(name, rd, folder, stage)
        self._status_bar.showMessage(note, 8000)
        if ok:
            self._train_tab.refresh_sets()

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
        from core import run_manifest, sets
        rd = run_manifest.find_resumable(self._setup_tab.get_output_dir())
        if rd is None:
            rd = sets.interrupted_run()
        if rd is not None:
            self._switch_tab(4)
            self._train_tab.show_recovery_banner(rd)

    # ---- self-update (Home footer "⟳ Updates") ----

    def _on_update_check(self):
        """Check GitHub main for a newer version — network runs off the UI thread."""
        if self._update_busy:
            return
        self._update_busy = True
        self._status_bar.showMessage("Checking GitHub for updates…")
        import threading
        from core import updater

        def work():
            self._update_check_done.emit(updater.fetch_remote_version() or "")

        threading.Thread(target=work, daemon=True).start()

    def _show_update_result(self, remote: str):
        from core import updater
        from core.version import __version__
        self._update_busy = False
        self._status_bar.clearMessage()
        if not remote:
            QMessageBox.warning(
                self, "Check for Updates",
                "Could not reach GitHub — check your connection and try again.")
            return
        if not updater.is_newer(remote, __version__):
            QMessageBox.information(
                self, "Check for Updates", f"You're up to date (v{__version__}).")
            return
        # Subtle sidebar hint even if they decline the update now.
        self._ver_label.setText(f"v{__version__} · UPDATE AVAILABLE")
        if QMessageBox.question(
            self, "Update Available",
            f"AnimaForge v{remote} is available (you have v{__version__}).\n\n"
            "Download and update now? Your sets, settings, datasets, and models "
            "are untouched.",
        ) != QMessageBox.Yes:
            return
        self._update_busy = True
        self._status_bar.showMessage(f"Downloading v{remote} from GitHub…")
        import tempfile
        import threading
        from pathlib import Path
        from core import updater as up
        app_root = Path(__file__).resolve().parents[1]

        def work():
            try:
                with tempfile.TemporaryDirectory() as td:
                    new_root = up.download_and_extract(td)
                    req_changed = up.requirements_changed(new_root, app_root)
                    up.apply_update(new_root, app_root)
                self._update_apply_done.emit(remote, req_changed, "")
            except Exception as e:  # noqa: BLE001 — any failure = install untouched
                self._update_apply_done.emit(remote, False, str(e))

        threading.Thread(target=work, daemon=True).start()

    def _on_update_applied(self, version: str, req_changed: bool, error: str):
        self._update_busy = False
        if error:
            self._status_bar.clearMessage()
            QMessageBox.warning(
                self, "Update Failed",
                f"The update was not applied: {error}\n\nYour install is unchanged.")
            return
        self._status_bar.showMessage(f"Updated to v{version} — restart to finish.")
        self._ver_label.setText(f"v{version} · RESTART TO APPLY")
        extra = ("\n\nrequirements.txt changed — run install.bat before the next "
                 "launch." if req_changed else "")
        QMessageBox.information(
            self, "Update Applied",
            f"AnimaForge v{version} is in place.\nRestart AnimaForge to finish.{extra}")

    def _on_load_set_requested(self, dataset_folder: str, trigger: str, quality_prefix: str):
        if dataset_folder:
            self._dataset_tab.load_folder_path(dataset_folder)
        if trigger:
            self._dataset_tab.set_trigger_word(trigger)
        # Unconditional (not `if quality_prefix:`): a loaded set with an empty prefix must
        # clear whatever the previous dataset left in the field — that's the exact
        # stale-live-UI bug this restore path exists to prevent. Restore both the visible
        # Home/Dataset field and TrainTab's own copy so the UI never disagrees with what
        # a queued run would actually use.
        self._dataset_tab.set_prefix(quality_prefix)
        self._train_tab.set_quality_prefix(quality_prefix)

    def _on_dataset_loaded(self, folder_path: str, image_count: int):
        self._train_tab.set_dataset(folder_path, image_count)
        self._train_tab.set_trigger_word(self._dataset_tab.get_trigger_word())
        self._train_tab.set_quality_prefix(self._dataset_tab.get_prefix())
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
