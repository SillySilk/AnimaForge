import os
import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ui.forge_modal import ForgeModal


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


class _ClickTile(QFrame):
    """A stat tile that acts as a button: click anywhere on it to open its editor."""

    def __init__(self, on_click, parent=None):
        super().__init__(parent)
        self._on_click = on_click

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._on_click is not None:
            self._on_click()
        super().mousePressEvent(event)


class HomeTab(QWidget):
    """Home — "The Bench": the one-click cockpit.

    Point at a folder, name it, then the two moves — Caption, then Train — with
    everything else popped out (Ready checklist modal, Save/Load Project).

    Home is the single source of truth for the run definition. The heavy
    controls (caption panel, step calculator, train panel) are the *real*
    widgets owned by the Dataset/Train tabs and mounted here by MainWindow, so
    all engine wiring stays intact — this screen only reframes them in the forge
    layout.
    """

    navigate = Signal(int)
    autodetect_requested = Signal()
    recover_requested = Signal()

    # Quick Run cockpit intents — MainWindow translates these into the real tabs.
    folder_chosen = Signal(str)
    name_changed = Signal(str)
    trigger_changed = Signal(str)  # the set's trigger word (single source; drives Dataset+Train)
    prefix_changed = Signal(str)   # quality prefix baked at Combine (single source; drives Dataset)
    type_changed = Signal(str)     # "character" / "concept" / "style" (auto-detect → Train)
    anchor_changed = Signal(str)
    run_requested = Signal()          # "Forge It" — the unattended caption→train pipeline
    run_caption_requested = Signal()  # pillar "Run Captioning" button
    start_train_requested = Signal()  # pillar "Start Training" button
    stop_train_requested = Signal()   # pillar "Stop" button (enabled while a run is live)
    presets_closed = Signal()         # an Options/Step Calculator modal closed → re-pull summary
    preset_pick_requested = Signal()  # PRESET button — MainWindow opens the picker (owns the data)

    _GLYPH = {"ok": "✓", "idle": "–", "err": "✗"}
    _OBJ = {"ok": "ready_row_ok", "idle": "ready_row_idle", "err": "ready_row_err"}
    # Subject type keys, parallel to the cockpit combo entries.
    _TYPE_KEYS = ["character", "concept", "style"]
    _TYPE_LABELS = ["Character", "Object / Concept", "Style"]
    # The six "core envt" rows surfaced in the header Ready pill / checklist modal.
    _READY_CORE = ["sd-scripts", "DiT model", "Qwen3 encoder", "VAE",
                   "PyTorch 2.5+", "LM Studio"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ready_labels = {}
        self._recent_label = None
        self._recover_btn = None
        self._lms_thread = None
        self._last_ctx = {}
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

    @staticmethod
    def validate_lora_name(raw: str):
        """Inline name check → (ok, message). Rules per the design handoff."""
        raw = (raw or "").strip()
        if not raw:
            return False, "Name is empty"
        if any(c.isspace() for c in raw):
            return False, "No spaces — use _ instead"
        if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in raw):
            return False, "Letters, numbers, . _ - only"
        if len(raw) > 64:
            return False, "Too long (max 64)"
        return True, "Good — safe to forge"

    # ---- construction ----
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        # centred column, capped ~1120px per the mock
        wrap = QHBoxLayout(content)
        wrap.setContentsMargins(34, 22, 34, 40)
        wrap.addStretch()
        col_host = QWidget()
        col_host.setMaximumWidth(1120)
        wrap.addWidget(col_host, 1)
        wrap.addStretch()

        root = QVBoxLayout(col_host)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(22)

        # detached readiness labels (surfaced in the Ready modal; kept live for
        # _set_row / the LM Studio ping regardless of whether the modal is open)
        for label, _ in self._readiness_rows({}):
            row = QLabel()
            row.setObjectName("ready_row_idle")
            self._ready_labels[label] = row

        root.addWidget(self._build_action_row())
        root.addWidget(self._build_set_card())
        root.addWidget(self._build_pillars())
        root.addWidget(self._build_lever())
        root.addWidget(self._build_footer())
        root.addStretch()

    # ---- action row (Save / Load / Ready pill) ----
    def _build_action_row(self) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(12)
        h.addStretch()

        save = QPushButton("  Save Project")
        save.setObjectName("af_btn_ghost")
        save.setMinimumHeight(36)
        save.setCursor(Qt.PointingHandCursor)
        save.clicked.connect(self._save_project)
        h.addWidget(save)

        load = QPushButton("  Load Project")
        load.setObjectName("af_btn_ghost")
        load.setMinimumHeight(36)
        load.setCursor(Qt.PointingHandCursor)
        load.clicked.connect(self._load_project)
        h.addWidget(load)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFixedHeight(22)
        h.addWidget(sep)

        self._ready_pill = QPushButton("READY · 0 / 6")
        self._ready_pill.setObjectName("af_pill_ok")
        self._ready_pill.setMinimumHeight(36)
        self._ready_pill.setCursor(Qt.PointingHandCursor)
        self._ready_pill.clicked.connect(self._open_ready_modal)
        h.addWidget(self._ready_pill)

        info = QPushButton("i")
        info.setObjectName("af_icon_btn")
        info.setFixedSize(36, 36)
        info.setCursor(Qt.PointingHandCursor)
        info.setToolTip("Ready checklist")
        info.clicked.connect(self._open_ready_modal)
        h.addWidget(info)
        return row

    # ---- The Set ----
    def _build_set_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("af_card")
        c = QVBoxLayout(card)
        c.setContentsMargins(20, 18, 20, 16)
        c.setSpacing(10)

        # dataset folder header row
        top = QHBoxLayout()
        lbl = QLabel("DATASET FOLDER")
        lbl.setObjectName("af_eyebrow_mute")
        top.addWidget(lbl)
        top.addStretch()
        self._folder_status = QLabel("")
        self._folder_status.setObjectName("af_eyebrow_mute")
        self._folder_status.setStyleSheet("color:#8fa86b;")
        top.addWidget(self._folder_status)
        c.addLayout(top)

        folder_row = QHBoxLayout()
        folder_row.setSpacing(8)
        self._folder_edit = QLineEdit()
        self._folder_edit.setReadOnly(True)
        self._folder_edit.setMinimumHeight(42)
        self._folder_edit.setPlaceholderText("Choose the folder with your LoRA images…")
        browse = QPushButton("Browse")
        browse.setObjectName("af_btn_ghost")
        browse.setMinimumHeight(42)
        browse.setFixedWidth(96)
        browse.setCursor(Qt.PointingHandCursor)
        browse.clicked.connect(self._browse_folder)
        folder_row.addWidget(self._folder_edit)
        folder_row.addWidget(browse)
        c.addLayout(folder_row)

        # slim tagline (half-height hero)
        tagline = QHBoxLayout()
        tagline.setSpacing(12)
        big = QLabel("Point it at a folder. Name it. Forge.")
        big.setObjectName("af_display_gold4")
        tagline.addWidget(big)
        mk = QLabel("caption, then train.")
        mk.setObjectName("af_marker")
        mk.setStyleSheet("color:#a89c7e;")
        tagline.addWidget(mk)
        tagline.addStretch()
        c.addSpacing(4)
        c.addLayout(tagline)

        rule = QFrame()
        rule.setObjectName("af_rule")
        rule.setFixedHeight(2)
        c.addWidget(rule)

        # name (+ Check) / trigger / prefix
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        grid.addWidget(self._field_label("LORA NAME"), 0, 0)
        grid.addWidget(self._field_label("TRIGGER WORD (OPTIONAL)"), 0, 1)
        grid.addWidget(self._field_label("QUALITY PREFIX (OPTIONAL)"), 0, 2)

        name_row = QWidget()
        nr = QHBoxLayout(name_row)
        nr.setContentsMargins(0, 0, 0, 0)
        nr.setSpacing(8)
        self._name_edit = QLineEdit()
        self._name_edit.setMinimumHeight(42)
        self._name_edit.setPlaceholderText("Set when you choose a folder")
        self._name_edit.textEdited.connect(self._on_name_edited)
        check = QPushButton("  Check")
        check.setObjectName("af_btn_ghost")
        check.setMinimumHeight(42)
        check.setCursor(Qt.PointingHandCursor)
        check.clicked.connect(self._validate_name)
        nr.addWidget(self._name_edit)
        nr.addWidget(check)
        grid.addWidget(name_row, 1, 0)

        self._trigger_edit = QLineEdit()
        self._trigger_edit.setMinimumHeight(42)
        self._trigger_edit.setPlaceholderText("e.g. mycharacter")
        self._trigger_edit.textEdited.connect(self.trigger_changed.emit)
        grid.addWidget(self._trigger_edit, 1, 1)

        self._prefix_edit = QLineEdit()
        self._prefix_edit.setMinimumHeight(42)
        self._prefix_edit.setPlaceholderText("e.g. masterpiece, best quality")
        self._prefix_edit.textEdited.connect(self.prefix_changed.emit)
        grid.addWidget(self._prefix_edit, 1, 2)

        grid.setColumnStretch(0, 135)
        grid.setColumnStretch(1, 100)
        grid.setColumnStretch(2, 100)
        c.addSpacing(4)
        c.addLayout(grid)

        # inline name-check status
        self._name_msg = QLabel("")
        self._name_msg.setObjectName("af_eyebrow_mute")
        self._name_msg.setVisible(False)
        c.addWidget(self._name_msg)

        # Training preset — ONE labeled control instead of a knob row (radios were too
        # restrictive once presets became user-extensible; a bare combo flipped values
        # on stray scrolls in an earlier iteration). The picker modal lists presets and
        # applies nothing until its explicit Select button. Gear = Step Calculator.
        c.addSpacing(6)
        sub_row = QHBoxLayout()
        sub_row.setSpacing(10)
        sub_lbl = QLabel("PRESET")
        sub_lbl.setObjectName("af_eyebrow_mute")
        sub_row.addWidget(sub_lbl)
        self._preset_btn = QPushButton("👤  Person  ▾")
        self._preset_btn.setObjectName("af_btn_ghost")
        self._preset_btn.setMinimumHeight(36)
        self._preset_btn.setCursor(Qt.PointingHandCursor)
        self._preset_btn.setToolTip(
            "Training preset — subject, optimizer, network size & steps in one pick.\n"
            "Add your own in Setup → Training Presets.")
        self._preset_btn.clicked.connect(self.preset_pick_requested.emit)
        sub_row.addWidget(self._preset_btn)
        sub_row.addStretch()
        gear = QPushButton("⚙")
        gear.setObjectName("af_icon_btn")
        gear.setFixedSize(34, 34)
        gear.setCursor(Qt.PointingHandCursor)
        gear.setToolTip("Target steps & advanced step settings")
        gear.clicked.connect(self._open_stepcalc_modal)
        sub_row.addWidget(gear)
        c.addLayout(sub_row)

        # Style @anchor (only meaningful for Style runs)
        self._anchor_label = self._field_label("STYLE @ANCHOR (OPTIONAL)")
        c.addWidget(self._anchor_label)
        self._anchor_edit = QLineEdit()
        self._anchor_edit.setMinimumHeight(38)
        self._anchor_edit.setPlaceholderText("@mystyle")
        self._anchor_edit.textEdited.connect(self.anchor_changed.emit)
        c.addWidget(self._anchor_edit)
        self._set_anchor_visible(False)

        return card

    # ---- two pillars ----
    def _build_pillars(self) -> QWidget:
        host = QWidget()
        g = QGridLayout(host)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(0)

        # CAPTION pillar
        cap = QFrame()
        cap.setObjectName("af_card")
        cl = QVBoxLayout(cap)
        cl.setContentsMargins(22, 0, 22, 22)
        cl.setSpacing(0)
        cl.addWidget(self._pillar_accent())
        cl.addSpacing(18)
        cl.addWidget(self._pillar_head("STEP 01", "Caption",
                                       action_label="Options",
                                       on_action=self._open_caption_modal))
        # (Train pillar's action is also "Options" now — "Presets" is reserved for
        # the intent presets picker on the Set card.)
        marker = QLabel("Tag, describe & combine — training-ready text for every image.")
        marker.setObjectName("af_marker")
        marker.setWordWrap(True)
        cl.addWidget(marker)
        cl.addSpacing(14)
        cl.addLayout(self._chip_row(["Auto-Tag", "Describe", "Combine"]))
        cl.addSpacing(14)
        # live status line (N / M captioned)
        status_row = QHBoxLayout()
        sl = QLabel("STATUS")
        sl.setObjectName("af_eyebrow_mute")
        status_row.addWidget(sl)
        status_row.addStretch()
        self._caption_status = QLabel("— not captioned")
        self._caption_status.setObjectName("af_stat_value")
        status_row.addWidget(self._caption_status)
        cl.addLayout(status_row)
        cl.addStretch()
        # primary action; the mass of options lives in the Options modal
        self._run_caption_btn = QPushButton("📝  Run Captioning")
        self._run_caption_btn.setObjectName("btn_start")
        self._run_caption_btn.setMinimumHeight(48)
        self._run_caption_btn.setCursor(Qt.PointingHandCursor)
        self._run_caption_btn.clicked.connect(self.run_caption_requested.emit)
        cl.addWidget(self._run_caption_btn)
        g.addWidget(cap, 0, 0)

        # connector
        conn = QWidget()
        conn.setFixedWidth(54)
        cc = QVBoxLayout(conn)
        cc.setContentsMargins(0, 0, 0, 0)
        cc.addStretch()
        ingot = QLabel("→")
        ingot.setAlignment(Qt.AlignCenter)
        ingot.setFixedSize(38, 38)
        ingot.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #f6c453, stop:1 #b8860b);"
            "color:#1a1206; border-radius:19px; font-size:19px;")
        cc.addWidget(ingot, 0, Qt.AlignHCenter)
        then = QLabel("then")
        then.setObjectName("af_marker_gold")
        then.setAlignment(Qt.AlignCenter)
        cc.addWidget(then)
        cc.addStretch()
        g.addWidget(conn, 0, 1)

        # TRAIN pillar
        tr = QFrame()
        tr.setObjectName("af_card")
        tl = QVBoxLayout(tr)
        tl.setContentsMargins(22, 0, 22, 22)
        tl.setSpacing(0)
        tl.addWidget(self._pillar_accent())
        tl.addSpacing(18)
        tl.addWidget(self._pillar_head("STEP 02", "Train",
                                       action_label="Options",
                                       on_action=self._open_train_modal))
        marker2 = QLabel("Anima-tuned settings, already dialed in. Just pull the lever.")
        marker2.setObjectName("af_marker")
        marker2.setWordWrap(True)
        tl.addWidget(marker2)
        tl.addSpacing(14)
        tl.addLayout(self._stat_tiles())
        tl.addSpacing(10)
        # compact network readout (dim · alpha · resolution)
        self._network_line = QLabel("dim 16 · alpha 8 · 1024px")
        self._network_line.setObjectName("af_eyebrow_mute")
        tl.addWidget(self._network_line)
        tl.addStretch()
        # primary action; the mass of settings lives in the Presets modal. Stop rides
        # alongside (the front owns ALL run controls — it went missing in the Bench
        # redesign) and lights up only while a run is live.
        start_stop = QHBoxLayout()
        start_stop.setSpacing(8)
        self._start_train_btn = QPushButton("🚀  Start Training")
        self._start_train_btn.setObjectName("btn_start")
        self._start_train_btn.setMinimumHeight(48)
        self._start_train_btn.setCursor(Qt.PointingHandCursor)
        self._start_train_btn.clicked.connect(self.start_train_requested.emit)
        start_stop.addWidget(self._start_train_btn, 1)
        self._stop_train_btn = QPushButton("■  Stop")
        self._stop_train_btn.setObjectName("btn_stop")
        self._stop_train_btn.setMinimumHeight(48)
        self._stop_train_btn.setCursor(Qt.PointingHandCursor)
        self._stop_train_btn.setEnabled(False)
        self._stop_train_btn.setToolTip("Stop the current training run (asks to confirm)")
        self._stop_train_btn.clicked.connect(self.stop_train_requested.emit)
        start_stop.addWidget(self._stop_train_btn)
        tl.addLayout(start_stop)
        g.addWidget(tr, 0, 2)

        g.setColumnStretch(0, 1)
        g.setColumnStretch(2, 1)
        return host

    def _pillar_accent(self) -> QFrame:
        acc = QFrame()
        acc.setObjectName("af_pillar_accent")
        acc.setFixedHeight(2)
        return acc

    def _pillar_head(self, step: str, title: str, action_label=None, on_action=None) -> QWidget:
        head = QWidget()
        h = QHBoxLayout(head)
        h.setContentsMargins(0, 0, 0, 0)
        col = QVBoxLayout()
        col.setSpacing(2)
        eb = QLabel(step)
        eb.setObjectName("af_eyebrow_flame")
        col.addWidget(eb)
        t = QLabel(title)
        t.setObjectName("af_display_gold")
        col.addWidget(t)
        h.addLayout(col)
        h.addStretch()
        if action_label and on_action is not None:
            btn = QPushButton("⚙  " + action_label)
            btn.setObjectName("af_btn_ghost")
            btn.setMinimumHeight(32)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(f"{action_label} — the full options open in a modal")
            btn.clicked.connect(on_action)
            h.addWidget(btn, 0, Qt.AlignTop)
        return head

    def _chip_row(self, labels) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        for i, text in enumerate(labels):
            chip = QLabel(text.upper())
            chip.setObjectName("af_chip")
            chip.setAlignment(Qt.AlignCenter)
            row.addWidget(chip, 1)
            if i < len(labels) - 1:
                arr = QLabel("→")
                arr.setStyleSheet("color:#8a5a12;")
                row.addWidget(arr)
        return row

    def _stat_tiles(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        self._tile_values = {}
        # Each tile opens the modal that edits its value — a readout you can't click
        # was the #1 "how do I change the steps?" stumble (user feedback).
        tile_defs = [
            ("steps", "TARGET STEPS", "—",
             "Click to set the target steps (Step Calculator)", self._open_stepcalc_modal),
            ("optimizer", "OPTIMIZER", "Prodigy",
             "Click to change the optimizer & network (Train Options)", self._open_train_modal),
        ]
        for key, cap_text, val, tip, on_click in tile_defs:
            tile = _ClickTile(on_click)
            tile.setObjectName("af_well")
            tile.setCursor(Qt.PointingHandCursor)
            tile.setToolTip(tip)
            tv = QVBoxLayout(tile)
            tv.setContentsMargins(12, 9, 12, 9)
            tv.setSpacing(2)
            cl = QLabel(cap_text)
            cl.setObjectName("af_eyebrow_mute")
            tv.addWidget(cl)
            vv = QLabel(val)
            vv.setObjectName("af_stat_value")
            tv.addWidget(vv)
            self._tile_values[key] = vv
            row.addWidget(tile, 1)
        return row

    # ---- live readouts (fed by MainWindow.refresh) ----
    def set_caption_status(self, done: int, total: int):
        if total:
            self._caption_status.setText(f"{done} / {total} captioned")
        else:
            self._caption_status.setText("— not captioned")

    def set_train_summary(self, steps=None, optimizer=None, dim=None, alpha=None, res=None):
        if steps is not None:
            self._tile_values["steps"].setText(str(steps))
        if optimizer:
            self._tile_values["optimizer"].setText(str(optimizer))
        if dim is not None and alpha is not None and res is not None:
            self._network_line.setText(f"dim {dim} · alpha {alpha} · {res}px")

    # ---- The Lever ----
    def _build_lever(self) -> QFrame:
        band = QFrame()
        band.setObjectName("af_lever")
        h = QHBoxLayout(band)
        h.setContentsMargins(22, 16, 22, 16)
        h.setSpacing(20)
        col = QVBoxLayout()
        col.setSpacing(4)
        eb = QLabel("UNATTENDED PIPELINE")
        eb.setObjectName("af_eyebrow_flame")
        eb.setStyleSheet("color:#f4d160; letter-spacing:3px;")
        col.addWidget(eb)
        mk = QLabel("Forge it — caption, then train. Walk away.")
        mk.setObjectName("af_marker")
        mk.setStyleSheet("color:#e9e0cc; font-size:18px;")
        col.addWidget(mk)
        h.addLayout(col)
        h.addStretch()

        forge = QPushButton("⚒  Forge It")
        forge.setObjectName("af_btn_forge")
        forge.setMinimumHeight(52)
        forge.setMinimumWidth(200)
        forge.setCursor(Qt.PointingHandCursor)
        forge.clicked.connect(self.run_requested.emit)
        h.addWidget(forge)
        return band

    # ---- footer (recover / auto-detect / recent) ----
    def _build_footer(self) -> QWidget:
        foot = QWidget()
        v = QVBoxLayout(foot)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)

        row = QHBoxLayout()
        row.setSpacing(12)
        self._recover_btn = QPushButton("♻  Recover last run")
        self._recover_btn.setObjectName("btn_primary")
        self._recover_btn.setCursor(Qt.PointingHandCursor)
        self._recover_btn.clicked.connect(self.recover_requested.emit)
        self._recover_btn.setVisible(False)
        row.addWidget(self._recover_btn)
        row.addStretch()
        detect = QPushButton("Auto-detect models")
        detect.setObjectName("af_btn_ghost")
        detect.setMinimumHeight(34)
        detect.setCursor(Qt.PointingHandCursor)
        detect.clicked.connect(self.autodetect_requested.emit)
        row.addWidget(detect)
        v.addLayout(row)

        self._recent_label = QLabel("No runs yet.")
        self._recent_label.setObjectName("af_eyebrow_mute")
        self._recent_label.setWordWrap(True)
        v.addWidget(self._recent_label)
        return foot

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("af_eyebrow_mute")
        return lab

    def _set_anchor_visible(self, visible: bool):
        self._anchor_label.setVisible(visible)
        self._anchor_edit.setVisible(visible)

    # ---- name validation ----
    def _on_name_edited(self, text: str):
        self._name_msg.setVisible(False)
        self.name_changed.emit(text)

    def _validate_name(self):
        ok, msg = self.validate_lora_name(self._name_edit.text())
        self._name_msg.setText(("✓  " if ok else "⚠  ") + msg)
        self._name_msg.setStyleSheet(
            "color:#8fa86b;" if ok else "color:#ff9a5c;")
        self._name_msg.setVisible(True)

    # ---- Save / Load Project ----
    def _save_project(self):
        name = (self._name_edit.text() or "animaforge").strip() or "animaforge"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", f"{name}.forge.json",
            "AnimaForge Project (*.forge.json *.json)")
        if not path:
            return
        data = {
            "folder": self._folder_edit.text(),
            "loraName": self._name_edit.text(),
            "triggerWord": self._trigger_edit.text(),
            "qualityPrefix": self._prefix_edit.text(),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "",
            "AnimaForge Project (*.forge.json *.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            return
        folder = d.get("folder") or ""
        name = d.get("loraName") or ""
        trigger = d.get("triggerWord") or ""
        prefix = d.get("qualityPrefix") or ""
        if name:
            self._name_edit.setText(name)
            self.name_changed.emit(name)
        if trigger:
            self._trigger_edit.setText(trigger)
            self.trigger_changed.emit(trigger)
        if prefix:
            self._prefix_edit.setText(prefix)
            self.prefix_changed.emit(prefix)
        if folder:
            self._folder_edit.setText(folder)
            # loading the folder re-derives subject type + steps downstream
            self.folder_chosen.emit(folder)
            self._autoset_type(folder)

    # ---- Ready checklist modal ----
    def _open_ready_modal(self):
        rows = self._readiness_rows(self._last_ctx or {})
        ok = sum(1 for k, s in rows if k in self._READY_CORE and s == "ok")
        modal = ForgeModal(
            self.window(), title="Ready to Forge",
            subtitle="What's lit, and what's still cold.", max_width=440)
        for label, state in rows:
            r = QLabel(f"{self._GLYPH.get(state, '–')}   {label}"
                       + ("   (optional)" if label == "LM Studio" else ""))
            r.setObjectName(self._OBJ.get(state, "ready_row_idle"))
            modal.body.addWidget(r)
        ctx = self._last_ctx or {}
        note = QLabel(
            f"{ctx.get('image_count', 0)} images · "
            f"{ok} / {len(self._READY_CORE)} core requirements ready")
        note.setObjectName("af_eyebrow_mute")
        note.setContentsMargins(0, 12, 0, 0)
        modal.body.addWidget(note)
        done = modal.add_footer_button("Close", primary=True)
        done.clicked.connect(modal.close_modal)
        modal.open()

    def _update_ready_pill(self, ctx):
        rows = dict(self._readiness_rows(ctx))
        ok = sum(1 for k in self._READY_CORE if rows.get(k) == "ok")
        self._ready_pill.setText(f"READY · {ok} / {len(self._READY_CORE)}")

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
        """Stash the Dataset tab's caption panel — shown in the Caption Options modal."""
        self._caption_panel = widget
        widget.setParent(self)
        widget.setVisible(False)

    def mount_train_controls(self, widget):
        """Stash the Train tab's control panel — shown in the Train Presets modal."""
        self._train_panel = widget
        widget.setParent(self)
        widget.setVisible(False)

    # ---- Options / Presets modals (host the stashed real panels) ----
    def _restash(self, panel):
        """Reparent a panel back onto Home (hidden) before its modal is destroyed, so the
        panel and its live state survive close/reopen."""
        if panel is not None:
            panel.setParent(self)
            panel.setVisible(False)

    def _open_caption_modal(self):
        panel = getattr(self, "_caption_panel", None)
        if panel is None:
            return
        modal = ForgeModal(
            self.window(), title="Captioning", eyebrow="Step 01 · Options",
            subtitle="Run the passes, or fire an individual step.", max_width=560)
        panel.setVisible(True)
        modal.body.addWidget(panel)
        modal.closed.connect(lambda p=panel: self._restash(p))
        modal.add_footer_button("Close", primary=True).clicked.connect(modal.close_modal)
        modal.open()

    def open_train_presets(self):
        """Back-compat public entry — opens the Train Options modal."""
        self._open_train_modal()

    def _open_train_modal(self):
        panel = getattr(self, "_train_panel", None)
        if panel is None:
            return
        modal = ForgeModal(
            self.window(), title="Train Options", eyebrow="Step 02 · Options",
            subtitle="Sample previews, optimizer & network, run options — all here.",
            max_width=960)
        panel.setVisible(True)
        modal.body.addWidget(panel)
        modal.closed.connect(lambda p=panel: self._restash(p))
        # Home's steps/optimizer/dim·alpha readouts otherwise refresh only on tab
        # switch — closing the editor is the moment they must be current.
        modal.closed.connect(self.presets_closed.emit)
        modal.add_footer_button("Close", primary=True).clicked.connect(modal.close_modal)
        modal.open()

    def mount_step_calculator(self, widget):
        """Stash the Train tab's Step Calculator — shown in the gear modal (numeric settings)."""
        self._stepcalc_panel = widget
        widget.setParent(self)
        widget.setVisible(False)

    def _open_stepcalc_modal(self):
        panel = getattr(self, "_stepcalc_panel", None)
        if panel is None:
            return
        panel.setVisible(True)
        modal = ForgeModal(
            self.window(), title="Step Calculator", eyebrow="Fine Tuning",
            subtitle="Target steps and the exposure cap — tuned to the subject type.",
            max_width=520)
        modal.body.addWidget(panel)
        modal.closed.connect(lambda p=panel: self._restash(p))
        modal.closed.connect(self.presets_closed.emit)
        modal.add_footer_button("Done", primary=True).clicked.connect(modal.close_modal)
        modal.open()

    _TYPE_ICON = {"character": "👤", "concept": "📦", "style": "🎨"}

    def set_preset_label(self, name: str, subject_type: str = ""):
        """Reflect the active training preset on the PRESET button (display only)."""
        icon = self._TYPE_ICON.get((subject_type or "").lower(), "👤")
        self._preset_btn.setText(f"{icon}  {name}  ▾")

    def set_style_anchor_visible(self, visible: bool):
        """Show the Style @anchor field only for Style runs (driven by Train's subject type)."""
        self._set_anchor_visible(bool(visible))

    def apply_run_progress(self, payload):
        # The mounted Train control panel carries the live RunProgress; Home mirrors it there,
        # so this remains a no-op-safe hook for MainWindow's existing wiring.
        pass

    def set_training_active(self, active: bool):
        """Mirror the run state onto the front-page Start/Stop pair."""
        self._start_train_btn.setEnabled(not active)
        self._stop_train_btn.setEnabled(bool(active))

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

        count = context.get("image_count", 0) or 0
        self._folder_status.setText(f"✓ {count} images" if (folder and count) else "")

        self._name_edit.blockSignals(True)
        self._name_edit.setText(context.get("lora_name", "") or "")
        self._name_edit.blockSignals(False)

        self._trigger_edit.blockSignals(True)
        self._trigger_edit.setText(context.get("trigger_word", "") or "")
        self._trigger_edit.blockSignals(False)

        self._prefix_edit.blockSignals(True)
        self._prefix_edit.setText(context.get("quality_prefix", "") or "")
        self._prefix_edit.blockSignals(False)

        # Active preset + subject type drive the PRESET button and @anchor visibility.
        key = (context.get("subject_type") or "character").lower()
        if context.get("preset_name"):
            self.set_preset_label(context["preset_name"], key)
        self._set_anchor_visible(key == "style")

        self._anchor_edit.blockSignals(True)
        self._anchor_edit.setText(context.get("style_anchor", "") or "")
        self._anchor_edit.blockSignals(False)

    def refresh(self, context):
        from core import sets
        self._last_ctx = dict(context)
        for label, state in self._readiness_rows(context):
            self._set_row(label, state)
        # live LM Studio reachability (non-blocking background ping)
        url = context.get("lms_url")
        if url:
            self._set_row("LM Studio", "idle", "  (checking…)")
            self._start_lms_ping(url)
        names = self._recent_outputs(context.get("output", ""))
        self._recent_label.setText(
            "Recent:  " + "   ·   ".join(names) if names else "No runs yet.")
        self._sync_cockpit(context)
        self._update_ready_pill(context)
        # live pillar readouts
        done, total = context.get("caption_counts", (0, 0))
        self.set_caption_status(done, total)
        self.set_train_summary(
            steps=context.get("target_steps"),
            optimizer=context.get("optimizer_label") or "Prodigy",
            dim=context.get("net_dim"), alpha=context.get("net_alpha"),
            res=context.get("net_res"))
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
