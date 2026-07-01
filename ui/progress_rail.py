"""Global workflow progress rail — Load → Name → Caption → Train.

A slim, always-visible stepper pinned above the main stacked content. It is a pure
view: `set_state(...)` hands it the readiness dict computed by `core.workflow` (plus
the active tab and whether naming applies), and it renders markers + counts and emits
`navigate(key)` when a segment is clicked. It is advisory only — it never gates Start.
See docs/superpowers/specs/2026-06-24-workflow-progress-rail-and-train-streamline-design.md.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

# Segment text colors by state (dark gold theme).
_DONE = "#7ed957"      # green check — step complete
_CURRENT = "#f4d160"   # bright gold — the active step
_TODO = "#8a8a93"      # dim grey — not done yet
_OPTIONAL = "#9a8f5f"  # muted gold — optional / not-applicable


class ProgressRail(QWidget):
    navigate = Signal(str)  # "load" | "name" | "caption" | "train"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("progress_rail")
        self.setStyleSheet(
            "#progress_rail{background:#100e08;border-bottom:1px solid #2a2a1e;}")
        row = QHBoxLayout(self)
        row.setContentsMargins(18, 6, 18, 6)
        row.setSpacing(6)

        self._segs = {}
        order = [("load", "Load"), ("name", "Name"),
                 ("caption", "Caption"), ("train", "Train")]
        for i, (key, _label) in enumerate(order):
            btn = QPushButton()
            btn.setObjectName("rail_seg")
            btn.setFlat(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, k=key: self.navigate.emit(k))
            self._segs[key] = btn
            row.addWidget(btn)
            if i < len(order) - 1:
                conn = QLabel("──")
                conn.setStyleSheet(f"color:{_TODO};")
                row.addWidget(conn)
        row.addStretch()

        self.set_state({})  # initial empty render

    def set_state(self, state: dict):
        """Render from a state dict:
            {
              'load':    {'images': int, 'done': bool},
              'name':    {'named': int, 'done': bool, 'applicable': bool},
              'caption': {'captioned': int, 'images': int, 'done': bool},
              'current': 'load'|'name'|'caption'|'train'|None,
            }
        Missing keys render as not-started, so a partial dict is always safe."""
        state = state or {}
        current = state.get("current")
        self._render("load", self._load_text(state.get("load", {})),
                     state.get("load", {}).get("done", False), current)
        self._render_name(state.get("name", {}), current)
        self._render("caption", self._caption_text(state.get("caption", {})),
                     state.get("caption", {}).get("done", False), current)
        # Train has no "done" of its own — it's complete only while it's the active step.
        self._render("train", "Train", False, current)

    # ---- per-segment rendering ----
    def _render(self, key, label, done, current):
        is_current = current == key
        marker = "●" if is_current else ("✓" if done else "○")
        if is_current:
            color, weight = _CURRENT, "700"
        elif done:
            color, weight = _DONE, "600"
        else:
            color, weight = _TODO, "500"
        self._apply(key, f"{marker}  {label}", color, weight)

    def _render_name(self, st, current):
        applicable = st.get("applicable", True)
        done = st.get("done", False)
        named = st.get("named", 0)
        is_current = current == "name"
        if not applicable:
            label, color = "Name · not needed for styles", _OPTIONAL
        elif done:
            label, color = f"Name · {named}", (_CURRENT if is_current else _DONE)
        else:
            label, color = "Name (optional)", (_CURRENT if is_current else _OPTIONAL)
        marker = "●" if is_current else ("✓" if (applicable and done) else "○")
        weight = "700" if is_current else ("600" if done else "500")
        self._apply("name", f"{marker}  {label}", color, weight)

    def _apply(self, key, text, color, weight):
        btn = self._segs[key]
        btn.setText(text)
        btn.setStyleSheet(
            f"#rail_seg{{color:{color};font-size:12px;font-weight:{weight};"
            "border:none;background:transparent;padding:2px 6px;text-align:left;}"
            "#rail_seg:hover{text-decoration:underline;}")

    # ---- text helpers ----
    @staticmethod
    def _load_text(st):
        images = st.get("images", 0)
        return f"Load · {images}" if images else "Load"

    @staticmethod
    def _caption_text(st):
        images = st.get("images", 0)
        captioned = st.get("captioned", 0)
        if images and captioned and captioned < images:
            return f"Caption · {captioned}/{images}"
        return "Caption"
