import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from ui.home_tab import HomeTab

_app = QApplication.instance() or QApplication([])


def _ctx(**over):
    base = dict(sdscripts="", dit="", qwen3="", vae="", output="",
                torch_ok=False, dataset_folder="", image_count=0,
                lms_url="", lms_ok=None)
    base.update(over)
    return base


def test_readiness_idle_when_unset():
    t = HomeTab()
    rows = dict(t._readiness_rows(_ctx()))
    assert rows["sd-scripts"] == "idle"
    assert rows["DiT model"] == "idle"
    assert rows["Dataset"] == "idle"


def test_readiness_ok_when_set(tmp_path: Path):
    dit = tmp_path / "d.safetensors"; dit.write_bytes(b"x")
    ds = tmp_path / "ds"; ds.mkdir()
    t = HomeTab()
    rows = dict(t._readiness_rows(_ctx(
        sdscripts=str(tmp_path), dit=str(dit), torch_ok=True,
        dataset_folder=str(ds), image_count=20)))
    assert rows["sd-scripts"] == "ok"
    assert rows["DiT model"] == "ok"
    assert rows["PyTorch 2.5+"] == "ok"
    assert rows["Dataset"] == "ok"


def test_recent_outputs_lists_newest(tmp_path: Path):
    out = tmp_path / "out"; out.mkdir()
    for i, n in enumerate(["a.safetensors", "b.safetensors", "c.txt"]):
        f = out / n; f.write_bytes(b"x"); os.utime(f, (1000 + i, 1000 + i))
    t = HomeTab()
    names = t._recent_outputs(str(out))
    assert names[0] == "b.safetensors"   # newest .safetensors first
    assert "c.txt" not in names          # only safetensors
    assert t._recent_outputs("") == []   # no dir -> empty


def test_lms_ping_updates_row():
    t = HomeTab()
    t._on_lms_ping(True)
    assert t._ready_labels["LM Studio"].objectName() == "ready_row_ok"
    t._on_lms_ping(False)
    assert t._ready_labels["LM Studio"].objectName() == "ready_row_err"


def test_refresh_runs_and_navigate_signal():
    t = HomeTab()
    t.refresh(_ctx())          # must not raise
    assert hasattr(t, "navigate")
    captured = []
    t.navigate.connect(lambda i: captured.append(i))
    t._go(3)
    assert captured == [3]


def test_suggest_name_from_folder():
    assert HomeTab.suggest_name_from_folder(r"C:\data\My Character") == "My_Character"
    assert HomeTab.suggest_name_from_folder("/x/y/margaux") == "margaux"
    assert HomeTab.suggest_name_from_folder("") == ""


def test_cockpit_sync_and_anchor_visibility():
    # Subject type + target steps now live in the relocated Step Calculator (Train owns the
    # widgets, shown on Home). Home still mirrors name/anchor and reflects Style-anchor
    # visibility from the context's subject_type.
    t = HomeTab()
    t.refresh(_ctx(lora_name="Foo", subject_type="style", target_steps=820,
                   style_anchor="@bat"))
    assert t._name_edit.text() == "Foo"
    assert t._anchor_edit.text() == "@bat"
    assert t._anchor_edit.isHidden() is False          # style context → anchor shown
    # set_style_anchor_visible is the public hook MainWindow drives from Train's subject type.
    t.set_style_anchor_visible(False)
    assert t._anchor_edit.isHidden() is True


def test_cockpit_trigger_sync_and_signal():
    t = HomeTab()
    t.refresh(_ctx(trigger_word="mychar"))
    assert t._trigger_edit.text() == "mychar"          # mirrored from context
    emitted = []
    t.trigger_changed.connect(lambda s: emitted.append(s))
    t._trigger_edit.textEdited.emit("newtrig")         # simulate a user edit
    assert emitted == ["newtrig"]


def test_stage_counts_from_context():
    t = HomeTab()
    t.refresh(_ctx(caption_stage_counts=(98, 42, 0, 98)))
    assert t._stage_count_labels["tag"].text() == "98 / 98"
    assert t._stage_count_labels["describe"].text() == "42 / 98"
    assert t._stage_count_labels["combine"].text() == "0 / 98"
    # empty dataset -> placeholder, no fake zero-of-zero
    t.refresh(_ctx())
    assert t._stage_count_labels["describe"].text() == "—"


def test_caption_tick_updates_chip_live():
    t = HomeTab()
    t.apply_caption_tick("Describe", 5, 98)
    assert t._stage_count_labels["describe"].text() == "5 / 98"
    assert t._stage_chips["describe"].objectName() == "af_stage_chip_live"
    assert t._stage_chips["tag"].objectName() == "af_stage_chip"
    # Refine ticks ride the DESCRIBE chip (it reworks the natural-language draft)
    t.apply_caption_tick("Refine", 7, 98)
    assert t._stage_count_labels["describe"].text() == "7 / 98"
    # unknown phases are ignored, authoritative counts clear the live highlight
    t.apply_caption_tick("Nonsense", 1, 2)
    t.set_stage_counts(98, 98, 98, 98)
    assert t._stage_chips["describe"].objectName() == "af_stage_chip"
    assert t._stage_count_labels["combine"].text() == "98 / 98"


def test_run_progress_shows_live_steps_on_front():
    t = HomeTab()
    t.apply_run_progress({"kind": "progress", "step": 143, "total": 800})
    assert t._train_progress._counter.text() == "143 / 800"
    t.apply_run_progress({"kind": "reset"})
    assert t._train_progress._counter.text() == ""


def test_home_emits_add_to_batch_requested():
    h = HomeTab()
    seen = []
    h.add_to_batch_requested.connect(lambda: seen.append(True))
    h._add_batch_btn.click()
    assert seen == [True]
