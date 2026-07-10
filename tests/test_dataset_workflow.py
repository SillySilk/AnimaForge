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


def test_process_run_hands_job_to_runner(tmp_path: Path):
    # The chain sequencing moved into CaptionRunner (covered by test_caption_runner). The
    # tab's job now is to snapshot settings into a CaptionJob and hand it to the runner.
    from PIL import Image
    from core.caption_policy import OVERWRITE
    _set_refine_in_process(False)
    Image.new("RGB", (8, 8), (10, 10, 10)).save(tmp_path / "a.png")
    t = DatasetTab()
    t._folder_path = str(tmp_path)
    t._sdscripts_path = "C:/sd"
    t._image_data = [{"image_path": str(tmp_path / "a.png")}]
    captured = {}
    t._runner.start = lambda job: (captured.__setitem__("job", job), True)[1]
    assert t.start_auto_caption() is True
    assert captured["job"].chain == ["tag", "describe", "combine"]
    assert captured["job"].policy == OVERWRITE   # this task always overwrites
    assert t._auto_mode is True                  # stays set until the runner finishes


def test_runner_finished_false_ends_auto_pipeline():
    # A mid-chain failure or a user Stop both reach the tab as runner.finished(False).
    # In the Home pipeline it must surface as auto_caption_finished(False), silently.
    t = DatasetTab()
    t._auto_mode = True
    got = []
    t.auto_caption_finished.connect(got.append)
    t._runner.finished.emit(False)
    assert got == [False]
    assert t._auto_mode is False
    assert t._process_btn.isEnabled()
    assert t._phase_label.text() == "Idle"


def test_runner_finished_true_emits_processed_and_finishes():
    # A clean finish emits the 'processed' autosave milestone, then auto_caption_finished(True).
    t = DatasetTab()
    t._auto_mode = True
    stages, auto = [], []
    t.caption_stage_done.connect(stages.append)
    t.auto_caption_finished.connect(auto.append)
    t._runner.finished.emit(True)
    assert stages == ["processed"]
    assert auto == [True]
    assert t._auto_mode is False
    assert t._process_btn.isEnabled()


def test_dataset_tab_has_validate_names_button():
    t = DatasetTab()
    assert hasattr(t, "_validate_names_btn")
    assert "Validate Names" in t._validate_names_btn.text()
    # the segmented filter + search are present; captioning is NOT launched from here
    assert set(t._seg_buttons) == {"all", "captioned", "needs"}
    assert hasattr(t, "_search_edit")


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
