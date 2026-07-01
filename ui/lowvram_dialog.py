"""Low-VRAM mode dialog — an opt-in, acknowledged, quality-neutral training recipe.

OFF by default. The user must tick BOTH "Enable" and the acknowledgment before Apply is
allowed; Apply writes the recipe to the in-memory (non-persistent) holder in core.lowvram.
See docs/superpowers/specs/2026-06-24-lowvram-and-settings-gear-design.md.
"""
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QRadioButton, QSpinBox, QVBoxLayout, QWidget,
)

from core import lowvram
from ui.collapsible import CollapsibleBox

_TARGETS = (16, 12, 10, 8)


class LowVramDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Low-VRAM Mode")
        self.setMinimumWidth(440)
        root = QVBoxLayout(self)

        blurb = QLabel(
            "For GPUs that can't otherwise fit Anima training. It keeps quality identical "
            "(same precision, resolution and effective batch) and only runs slower. Off by "
            "default and only for this session."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color: #c9c08a;")
        root.addWidget(blurb)

        self._enable_chk = QCheckBox("Enable Low-VRAM mode")
        root.addWidget(self._enable_chk)

        self._ack_chk = QCheckBox(
            "I understand: Low-VRAM mode trades speed for fit (same quality), this session only."
        )
        root.addWidget(self._ack_chk)

        # VRAM target radios
        tgt_row = QHBoxLayout()
        tgt_row.addWidget(QLabel("Your VRAM:"))
        self._tgt_group = QButtonGroup(self)
        for gb in _TARGETS:
            rb = QRadioButton(f"{gb} GB")
            self._tgt_group.addButton(rb, gb)
            tgt_row.addWidget(rb)
        tgt_row.addStretch()
        root.addLayout(tgt_row)
        self._tgt_group.idClicked.connect(self._apply_preset)

        # Advanced
        adv = CollapsibleBox("Advanced")
        av = adv.content_layout()
        self._micro = self._spin(av, "Micro-batch", 1, 4, 1)
        self._accum = self._spin(av, "Gradient accumulation", 1, 16, 4)
        self._swap = self._spin(av, "Blocks to swap (of 28, max 26)", 0, lowvram.MAX_BLOCKS_TO_SWAP, 8)
        self._fp8_chk = QCheckBox("Use fp8 base model (slightly lower quality — only if still out of memory)")
        av.addWidget(self._fp8_chk)
        root.addWidget(adv)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self._apply_btn = self._buttons.addButton("Apply", QDialogButtonBox.AcceptRole)
        self._buttons.accepted.connect(self._on_apply)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

        self._enable_chk.toggled.connect(self._update_apply_enabled)
        self._ack_chk.toggled.connect(self._update_apply_enabled)

        self._prefill_from_holder()
        self._update_apply_enabled()

    # ---- helpers ------------------------------------------------------
    def _spin(self, layout, label, lo, hi, val):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        sb = QSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(val)
        row.addWidget(sb)
        row.addStretch()
        holder = QWidget()
        holder.setLayout(row)
        layout.addWidget(holder)
        return sb

    def _apply_preset(self, gb):
        r = lowvram.recipe_for(gb)
        self._micro.setValue(r["micro_batch"])
        self._accum.setValue(r["grad_accum"])
        self._swap.setValue(r["blocks_to_swap"])

    def _update_apply_enabled(self, *_):
        self._apply_btn.setEnabled(self._enable_chk.isChecked() and self._ack_chk.isChecked())

    def _prefill_from_holder(self):
        cur = lowvram.get_current()
        if not cur:
            return
        self._enable_chk.setChecked(True)
        self._ack_chk.setChecked(True)
        self._micro.setValue(cur.get("micro_batch", 1))
        self._accum.setValue(cur.get("grad_accum", 4))
        self._swap.setValue(cur.get("blocks_to_swap", 8))
        self._fp8_chk.setChecked(bool(cur.get("fp8_base", False)))

    def _on_apply(self):
        if self._enable_chk.isChecked() and self._ack_chk.isChecked():
            lowvram.set_current({
                "micro_batch": self._micro.value(),
                "grad_accum": self._accum.value(),
                "blocks_to_swap": self._swap.value(),
                "fp8_base": self._fp8_chk.isChecked(),
            })
        else:
            lowvram.clear()
        self.accept()
