import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from utils.styles import asset_url
from ui.run_progress import RunProgress


class _LmsPing(QThread):
    """Background reachability check for an LM Studio (OpenAI-compatible) server."""
    done = Signal(bool)

    def __init__(self, url, parent=None):
        super().__init__(parent)
        self._url = url

    def run(self):
        ok = False
        try:
            import urllib.request
            base = self._url.rstrip("/")
            with urllib.request.urlopen(base + "/models", timeout=2.5) as r:
                ok = 200 <= getattr(r, "status", r.getcode()) < 500
        except Exception:
            ok = False
        self.done.emit(ok)


class HomeTab(QWidget):
    navigate = Signal(int)
    autodetect_requested = Signal()
    recover_requested = Signal()

    # Quick Run cockpit intents — MainWindow translates these into the real tabs.
    # Caption/train/batch are no longer Home signals: those controls are the relocated
    # Dataset/Train panels, wired directly to their owning tab's engine.
    folder_chosen = Signal(str)
    name_changed = Signal(str)
    trigger_changed = Signal(str)  # the set's trigger word (single source; drives Dataset+Train)
    prefix_changed = Signal(str)   # quality prefix baked at Combine (single source; drives Dataset)
    type_changed = Signal(str)    # "character" / "concept" / "style" (auto-detect → Train)
    anchor_changed = Signal(str)
    run_requested = Signal()

    _GLYPH = {"ok": "✓", "idle": "–", "err": "✗"}
    _OBJ = {"ok": "ready_row_ok", "idle": "ready_row_idle", "err": "ready_row_err"}
    # Subject type keys, parallel to the cockpit combo entries.
    _TYPE_KEYS = ["character", "concept", "style"]
    _TYPE_LABELS = ["Character", "Object / Concept", "Style"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ready_labels = {}
        self._recent_label = None
        self._recover_btn = None
        self._lms_thread = None
        self._build_ui()

    # ---- pure logic (unit-tested) ----
    def _readiness_rows(self, ctx):
        def has(p):
            return "ok" if p and os.path.exists(p) else "idle"
        rows = [
            ("sd-scripts", has(ctx.get("sdscripts"))),
            ("DiT model", has(ctx.get("dit"))),
            ("Qwen3 encoder", has(ctx.get("qwen3"))),
            ("VAE", has(ctx.get("vae"))),
            ("Output folder", has(ctx.get("output"))),
            ("PyTorch 2.5+", "ok" if ctx.get("torch_ok") else "idle"),
            ("Dataset", "ok" if (ctx.get("dataset_folder") and ctx.get("image_count", 0) > 0) else "idle"),
        ]
        lms_ok = ctx.get("lms_ok")
        if ctx.get("lms_url"):
            rows.append(("LM Studio", "ok" if lms_ok else ("err" if lms_ok is False else "idle")))
        else:
            rows.append(("LM Studio", "idle"))
        return rows

    def _recent_outputs(self, output_dir, limit=5):
        if not output_dir or not os.path.isdir(output_dir):
            return []
        items = []
        for root, _dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith(".safetensors"):
                    p = os.path.join(root, f)
                    items.append((os.path.getmtime(p), f))
        items.sort(reverse=True)
        return [name for _m, name in items[:limit]]

    @staticmethod
    def suggest_name_from_folder(folder: str) -> str:
        """A first-guess LoRA name from a dataset folder (basename, spaces -> underscores)."""
        if not folder or not folder.strip():
            return ""
        base = os.path.basename(os.path.normpath(folder)).strip()
        if base in (".", ".."):
            return ""
        return base.replace(" ", "_")

    # ---- construction ----
    def _build_ui(self):
        # Wrap content in a scroll area (like the other tabs) so that when the
        # window is shorter than the cockpit's natural height the content
        # scrolls instead of compressing children past their minimums and
        # overlapping (e.g. Run buttons landing on top of the Type/steps row).
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        root = QVBoxLayout(content)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        hero = QWidget()
        hero.setObjectName("hero")
        hero.setFixedHeight(150)
        hl = QHBoxLayout(hero)
        hl.setContentsMargins(20, 0, 20, 0)
        emblem = QLabel()
        pm = QPixmap(asset_url("emblem.png"))
        if not pm.isNull():
            emblem.setPixmap(pm.scaledToHeight(110, Qt.SmoothTransformation))
        emblem.setStyleSheet("background: transparent;")
        hl.addWidget(emblem)
        title = QLabel("Anima Forge LoRA Trainer\nwith Auto-Captioning")
        title.setObjectName("hero_title")
        hl.addWidget(title)
        hl.addStretch()
        root.addWidget(hero)

        body = QHBoxLayout()
        body.setSpacing(16)

        # readiness card
        ready_card = QFrame()
        ready_card.setObjectName("card")
        rc = QVBoxLayout(ready_card)
        rl = QLabel("FORGE READINESS")
        rl.setObjectName("label_section")
        rc.addWidget(rl)
        for label, _ in self._readiness_rows({}):
            row = QLabel()
            row.setObjectName("ready_row_idle")
            self._ready_labels[label] = row
            rc.addWidget(row)
        rc.addStretch()
        body.addWidget(ready_card, 1)

        # ---- Quick Run cockpit (the heart of Home) ----
        body.addWidget(self._build_quick_run_card(), 2)

        # recent outputs card
        rec_card = QFrame()
        rec_card.setObjectName("card")
        recl = QVBoxLayout(rec_card)
        rh = QLabel("RECENT OUTPUTS")
        rh.setObjectName("label_section")
        recl.addWidget(rh)
        self._recent_label = QLabel("No runs yet.")
        self._recent_label.setWordWrap(True)
        recl.addWidget(self._recent_label)
        recl.addStretch()
        body.addWidget(rec_card, 1)

        root.addLayout(body)
        root.addStretch()

    def _build_quick_run_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        ac = QVBoxLayout(card)
        al = QLabel("QUICK RUN")
        al.setObjectName("label_section")
        ac.addWidget(al)

        hint = QLabel("Point at a folder, name it, pick a type, and Run — captioning and naming "
                      "happen automatically. Drop into the Train tab only to fine-tune.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8a8a93; font-size: 11px;")
        ac.addWidget(hint)

        # Run progress — a prominent, isolated band at the top of the cockpit so its
        # phase/counter labels never crowd (or overlap) the form fields below.
        self._run_progress = RunProgress()
        ac.addWidget(self._run_progress)
        prog_sep = QFrame()
        prog_sep.setFrameShape(QFrame.HLine)
        prog_sep.setStyleSheet("color: #2a2a1e;")
        ac.addWidget(prog_sep)

        # Dataset folder
        ac.addWidget(self._field_label("Dataset folder"))
        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setReadOnly(True)
        self._folder_edit.setPlaceholderText("Choose the folder with your LoRA images…")
        browse = QPushButton("Browse…")
        browse.setFixedWidth(84)
        browse.clicked.connect(self._browse_folder)
        folder_row.addWidget(self._folder_edit)
        folder_row.addWidget(browse)
        ac.addLayout(folder_row)

        # LoRA name
        ac.addWidget(self._field_label("LoRA name"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Set when you choose a folder — editable here")
        self._name_edit.textEdited.connect(self.name_changed.emit)
        ac.addWidget(self._name_edit)

        # Trigger word — mirrors the Dataset tab's box so the set's trigger can be
        # set here on the front page, where most processing is driven from.
        ac.addWidget(self._field_label("Trigger word (optional)"))
        self._trigger_edit = QLineEdit()
        self._trigger_edit.setPlaceholderText("e.g. mycharacter (optional)")
        self._trigger_edit.textEdited.connect(self.trigger_changed.emit)
        ac.addWidget(self._trigger_edit)

        # Quality prefix — baked into the final captions when you Combine.
        ac.addWidget(self._field_label("Quality prefix (optional)"))
        self._prefix_edit = QLineEdit()
        self._prefix_edit.setPlaceholderText("optional, e.g. masterpiece, best quality")
        self._prefix_edit.textEdited.connect(self.prefix_changed.emit)
        ac.addWidget(self._prefix_edit)

        # Step Calculator (subject type, target steps, uncap, readout) is relocated here from
        # the Train tab so the front page is the single place to pick the type and tune steps.
        # MainWindow mounts the real widget (TrainTab.step_calculator()) into this slot.
        self._stepcalc_mount = QVBoxLayout()
        self._stepcalc_mount.setContentsMargins(0, 0, 0, 0)
        ac.addLayout(self._stepcalc_mount)

        # Style @anchor (only meaningful for Style runs)
        self._anchor_label = self._field_label("Style @anchor (optional)")
        ac.addWidget(self._anchor_label)
        self._anchor_edit = QLineEdit()
        self._anchor_edit.setPlaceholderText("@mystyle")
        self._anchor_edit.textEdited.connect(self.anchor_changed.emit)
        ac.addWidget(self._anchor_edit)
        self._set_anchor_visible(False)

        # Captioning controls are relocated here from the Dataset tab so the front page is
        # the single place to drive captioning. MainWindow mounts the real panel
        # (DatasetTab.caption_controls()) into this slot — captions the images only.
        self._caption_mount = QVBoxLayout()
        self._caption_mount.setContentsMargins(0, 0, 0, 0)
        ac.addLayout(self._caption_mount)

        # Training controls (Sample Previews, Start/Stop, Add to Batch, Deliver/Test in
        # Forge, Low VRAM, and the Advanced set-once block) are relocated here from the Train
        # tab so the front page is the single place to drive a run. MainWindow mounts the real
        # panel (TrainTab.control_panel()) into this slot.
        self._train_mount = QVBoxLayout()
        self._train_mount.setContentsMargins(0, 0, 0, 0)
        ac.addLayout(self._train_mount)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #2a2a1e;")
        ac.addWidget(sep)

        # Secondary shortcuts
        links = QHBoxLayout()
        for text, idx in [("\U0001f5bc  Dataset", 2), ("\U0001f4e6  Batch", 5)]:
            b = QPushButton(text)
            b.setObjectName("quick_action")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _c=False, i=idx: self._go(i))
            links.addWidget(b)
        ac.addLayout(links)

        detect = QPushButton("Auto-detect models")
        detect.clicked.connect(self.autodetect_requested.emit)
        ac.addWidget(detect)

        self._recover_btn = QPushButton("♻  Recover last run")
        self._recover_btn.setObjectName("btn_primary")
        self._recover_btn.clicked.connect(self.recover_requested.emit)
        self._recover_btn.setVisible(False)
        ac.addWidget(self._recover_btn)

        ac.addStretch()

        # Configure Setup demoted to the very bottom — it's a set-once concern.
        cfg = QPushButton("⚙  Configure Setup")
        cfg.setObjectName("quick_action")
        cfg.setCursor(Qt.PointingHandCursor)
        cfg.clicked.connect(lambda: self._go(1))
        ac.addWidget(cfg)
        return card

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setStyleSheet("color: #9a9aa2; font-size: 11px; font-weight: 600; padding-top: 4px;")
        return lab

    def _set_anchor_visible(self, visible: bool):
        self._anchor_label.setVisible(visible)
        self._anchor_edit.setVisible(visible)

    # ---- cockpit slots ----
    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder", self._folder_edit.text() or "")
        if not folder:
            return
        self._folder_edit.setText(folder)
        self.folder_chosen.emit(folder)
        self._autoset_type(folder)
        self._prompt_lora_name(folder)

    def _prompt_lora_name(self, folder: str):
        """A LoRA can't train unnamed — ask for the name the moment a folder is set."""
        suggested = self.suggest_name_from_folder(folder)
        name, ok = QInputDialog.getText(
            self, "Name this LoRA",
            "Name this LoRA (required before training):",
            text=suggested)
        name = (name or "").strip()
        if ok and name:
            self._name_edit.setText(name)
            self.name_changed.emit(name)

    def _autoset_type(self, folder):
        """Pick the subject type from the dataset's filenames (naming convention) and push it
        to the relocated Step Calculator via the type_changed intent (MainWindow → Train)."""
        try:
            from core.dataset_manager import scan_folder
            from core import naming
            stems = [Path(d["image_path"]).stem for d in scan_folder(folder)]
            dt = naming.project_category(stems)
        except Exception:
            return
        key = {"Character": "character", "Style": "style", "Object": "concept"}.get(dt or "")
        if key in self._TYPE_KEYS:
            self.type_changed.emit(key)

    def mount_caption_controls(self, widget):
        """Host the Dataset tab's caption control panel on Home (single source of truth)."""
        self._caption_mount.addWidget(widget)

    def mount_train_controls(self, widget):
        """Host the Train tab's run controls (previews/actions/advanced) on Home."""
        self._train_mount.addWidget(widget)

    def mount_step_calculator(self, widget):
        """Host the Train tab's Step Calculator (type + target steps) on Home."""
        self._stepcalc_mount.addWidget(widget)

    def set_style_anchor_visible(self, visible: bool):
        """Show the Style @anchor field only for Style runs (driven by Train's subject type)."""
        self._set_anchor_visible(bool(visible))

    def apply_run_progress(self, payload):
        self._run_progress.apply(payload)

    # ---- refresh + nav ----
    def _go(self, index):
        self.navigate.emit(index)

    def _set_row(self, label, state, suffix=""):
        lab = self._ready_labels.get(label)
        if lab is None:
            return
        lab.setText(f"  {self._GLYPH[state]}   {label}{suffix}")
        lab.setObjectName(self._OBJ[state])
        lab.style().unpolish(lab)
        lab.style().polish(lab)

    def _sync_cockpit(self, context):
        """Pull current run state from the app context into the cockpit (no signal echo)."""
        folder = context.get("dataset_folder", "") or ""
        self._folder_edit.setText(folder)

        self._name_edit.blockSignals(True)
        self._name_edit.setText(context.get("lora_name", "") or "")
        self._name_edit.blockSignals(False)

        self._trigger_edit.blockSignals(True)
        self._trigger_edit.setText(context.get("trigger_word", "") or "")
        self._trigger_edit.blockSignals(False)

        self._prefix_edit.blockSignals(True)
        self._prefix_edit.setText(context.get("quality_prefix", "") or "")
        self._prefix_edit.blockSignals(False)

        # Subject type + target steps now live in the relocated Step Calculator (Train owns
        # the widgets); here we only reflect the Style @anchor visibility from the context.
        key = (context.get("subject_type") or "character").lower()
        self._set_anchor_visible(key == "style")

        self._anchor_edit.blockSignals(True)
        self._anchor_edit.setText(context.get("style_anchor", "") or "")
        self._anchor_edit.blockSignals(False)

    def refresh(self, context):
        from core import sets
        for label, state in self._readiness_rows(context):
            self._set_row(label, state)
        # live LM Studio reachability (non-blocking background ping)
        url = context.get("lms_url")
        if url:
            self._set_row("LM Studio", "idle", "  (checking…)")
            self._start_lms_ping(url)
        names = self._recent_outputs(context.get("output", ""))
        self._recent_label.setText("\n".join(f"•  {n}" for n in names) if names else "No runs yet.")
        self._sync_cockpit(context)
        try:
            self._recover_btn.setVisible(sets.interrupted_run() is not None)
        except Exception:
            self._recover_btn.setVisible(False)

    def _start_lms_ping(self, url):
        prev = self._lms_thread
        if prev is not None and prev.isRunning():
            return  # a check is already in flight
        t = _LmsPing(url, self)
        t.done.connect(self._on_lms_ping)
        self._lms_thread = t
        t.start()

    def _on_lms_ping(self, ok):
        self._set_row("LM Studio", "ok" if ok else "err")
