import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from ui.dataset_tab import DatasetTab

_app = QApplication.instance() or QApplication([])


def test_step_status_counts(tmp_path: Path):
    from PIL import Image
    for n in ("a", "b", "c"):
        Image.new("RGB", (8, 8), (10, 10, 10)).save(tmp_path / f"{n}.png")
    (tmp_path / "a.tags").write_text("tag1, tag2", encoding="utf-8")
    (tmp_path / "b.tags").write_text("tag3", encoding="utf-8")
    (tmp_path / "a.nl").write_text("a description", encoding="utf-8")
    (tmp_path / "a.txt").write_text("final caption", encoding="utf-8")
    (tmp_path / "c.txt").write_text("", encoding="utf-8")           # empty -> not counted
    t = DatasetTab()
    c = t._step_status_counts(str(tmp_path))
    assert c["total"] == 3
    assert c["tags"] == 2
    assert c["nl"] == 1
    assert c["txt"] == 1


def test_step_status_counts_missing_folder():
    t = DatasetTab()
    assert t._step_status_counts("") == {"tags": 0, "nl": 0, "txt": 0, "total": 0}


def test_should_prompt_naming_truth_table():
    assert DatasetTab._should_prompt_naming(has_roster=False, enabled=True) is True
    assert DatasetTab._should_prompt_naming(has_roster=True, enabled=True) is False
    assert DatasetTab._should_prompt_naming(has_roster=False, enabled=False) is False


def test_dataset_tab_builds_process_panel():
    t = DatasetTab()
    # The caption controls are still built here (engine wiring intact) but the panel is now
    # exposed via caption_controls() for relocation onto Home, not parented into the splitter.
    for attr in ("_process_btn", "_phase_label", "_autotag_btn", "_describe_btn",
                 "_llm_btn", "_combine_btn", "_stop_caption_btn",
                 "_step1_status", "_step2_status", "_step3_status", "_step4_status",
                 "_hsplit", "_vsplit"):
        assert hasattr(t, attr), attr
    assert t._hsplit.count() == 1          # gallery only — caption panel relocated to Home
    assert t.caption_controls() is t._caption_side_panel
    assert t._vsplit.count() == 2          # content + log
    assert t._process_btn.text().startswith("📝")
    # Validate names stays on the Dataset tab as the lone fine-tune control.
    assert hasattr(t, "_validate_names_btn")
    captured = []
    t.open_characters_requested.connect(lambda: captured.append(1))
    t.open_characters_requested.emit()
    assert captured == [1]


def test_refresh_step_status_no_folder():
    _set_refine_in_process(False)
    t = DatasetTab()
    t._refresh_step_status()
    assert t._step1_status.text() == "— not run"
    assert "manual only" in t._step3_status.text().lower()


def test_process_steps_and_phase_text():
    from ui.dataset_tab import PROCESS_STEPS, STEP_NAMES, phase_text
    assert PROCESS_STEPS == ["tag", "describe", "refine", "combine"]
    assert STEP_NAMES["refine"] == "Refine"
    assert phase_text("tag") == "Step 1/4 · Tag…"
    assert phase_text("combine") == "Step 4/4 · Combine…"


def test_phase_text_numbers_against_actual_chain():
    from ui.dataset_tab import phase_text
    three = ["tag", "describe", "combine"]
    assert phase_text("tag", three) == "Step 1/3 · Tag…"
    assert phase_text("combine", three) == "Step 3/3 · Combine…"


def _set_refine_in_process(value: bool):
    from PySide6.QtCore import QSettings
    from core.settings import SETTINGS_ORG, SETTINGS_APP
    QSettings(SETTINGS_ORG, SETTINGS_APP).setValue("lmstudio_refine_in_process", value)


def test_build_process_chain_default_skips_refine():
    _set_refine_in_process(False)
    t = DatasetTab()
    assert t._build_process_chain() == ["tag", "describe", "combine"]


def test_build_process_chain_includes_refine_when_enabled():
    _set_refine_in_process(True)
    t = DatasetTab()
    assert t._build_process_chain() == ["tag", "describe", "refine", "combine"]
    _set_refine_in_process(False)


def test_refine_reflection_label_off_by_default():
    _set_refine_in_process(False)
    t = DatasetTab()
    t._refresh_refine_reflection()
    assert "manual only" in t._step3_status.text().lower()


def test_refine_reflection_label_when_enabled():
    _set_refine_in_process(True)
    t = DatasetTab()
    t._refresh_refine_reflection()
    assert "in process" in t._step3_status.text().lower()
    _set_refine_in_process(False)


def test_read_tagger_defaults_fallbacks():
    from PySide6.QtCore import QSettings
    from ui.dataset_tab import read_tagger_defaults
    s = QSettings("PonyExpress", "LoRATrainer")
    s.remove("tagger_model_index")
    s.remove("tagger_threshold")
    s.remove("tagger_overwrite")
    idx, thr, ow = read_tagger_defaults()
    assert idx == 0 and abs(thr - 0.35) < 1e-6 and ow is False


def test_process_chain_runs_steps_in_order():
    t = DatasetTab()
    calls = []
    t._start_tag_with_defaults = lambda: calls.append("tag")
    t._start_describe = lambda: calls.append("describe")
    t._start_refine = lambda: calls.append("refine")
    t._rebuild_txt_from_sidecars = lambda: (3, 0)
    t._refresh_step_status = lambda: None
    t._chain_finish_ok = lambda *a: calls.append("done")
    t._chain = list(__import__("ui.dataset_tab", fromlist=["PROCESS_STEPS"]).PROCESS_STEPS)
    t._chain_active = True
    t._chain_start_next()                       # tag
    t._chain_step_done("tag", True, "x")        # -> describe
    t._chain_step_done("describe", True, "x")   # -> refine
    t._chain_step_done("refine", True, "x")     # -> combine (sync) -> done
    assert calls == ["tag", "describe", "refine", "done"]
    assert t._chain == [] and t._chain_active is False


def test_process_chain_stops_on_failure():
    t = DatasetTab()
    failed = []
    t._chain_fail = lambda key, reason: failed.append(key)
    t._chain = ["describe", "refine", "combine"]
    t._chain_active = True
    t._chain_step_done("describe", False, "LM Studio not reachable")
    assert failed == ["describe"]


def test_chain_cancelled_resets():
    t = DatasetTab()
    t._chain = ["refine", "combine"]
    t._chain_active = True
    t._chain_cancelled()
    assert t._chain == [] and t._chain_active is False


def test_dataset_tab_has_validate_names_button():
    t = DatasetTab()
    assert hasattr(t, "_validate_names_btn")
    assert "Validate names" in t._validate_names_btn.text()


def test_validate_names_recombines_txt_with_name_first(tmp_path: Path):
    from PIL import Image
    from core import naming
    Image.new("RGB", (8, 8), (10, 10, 10)).save(tmp_path / "Homer_001_Character.png")
    (tmp_path / "Homer_001_Character.tags").write_text("1boy, bald", encoding="utf-8")
    (tmp_path / "Homer_001_Character.nl").write_text("a man", encoding="utf-8")
    naming.write_characters_from_names(str(tmp_path), ["Homer_001_Character.png"])
    t = DatasetTab()
    t._folder_path = str(tmp_path)
    t._on_names_validated()                      # what fires when Validate Names finishes
    txt = (tmp_path / "Homer_001_Character.txt").read_text(encoding="utf-8")
    assert txt.startswith("Homer,")             # name hoisted to the front of the caption


def test_rebuild_captions_after_naming_is_public(tmp_path: Path):
    # the Characters-tab "Fix names" path drives the same re-combine through this hook
    t = DatasetTab()
    assert hasattr(t, "rebuild_captions_after_naming")
