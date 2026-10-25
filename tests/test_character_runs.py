import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from core import character_runs as CR, characters as C
from core.batch import BatchRunner, RunDefinition, FAILED

_app = QApplication.instance() or QApplication([])


def _set(folder: Path, solo: dict, groups=(), captioned=True):
    """A dataset with `solo` = {token: n} solo images and group images of token tuples."""
    folder.mkdir(parents=True, exist_ok=True)
    data = C.DatasetCharacters(roster=[C.Character(t) for t in solo])
    def add(name, present):
        (folder / name).write_bytes(b"\x89PNG\r\n\x1a\n")
        if captioned:
            (folder / name).with_suffix(".txt").write_text(
                "Fam, " + ", ".join(present) + ", a caption", encoding="utf-8")
        data.assignments[name] = {"present": list(present), "oneoffs": []}
    for token, n in solo.items():
        for i in range(n):
            add(f"{token}_{i:03d}_Character.png", [token])
    for i, present in enumerate(groups):
        add(f"{'-'.join(present)}_{i:03d}_Character.png", present)
    C.save(str(folder), data)
    return data


def _main(folder, out, **kw):
    base = dict(lora_name="Fam", dataset_folder=str(folder), image_count=40,
                trigger_word="Fam", target_steps=2000, output_dir=str(out),
                sample_prompts=["Fam, a caption"], network_weights="old.safetensors")
    base.update(kw)
    return RunDefinition(**base)


def test_solo_images_leave_out_group_shots(tmp_path):
    data = _set(tmp_path / "set", {"Jay Pritchett": 6, "Gloria": 2},
                groups=[("Jay Pritchett", "Gloria")])
    solo = CR.solo_images(data, CR.image_names(str(tmp_path / "set")))
    assert len(solo["Jay Pritchett"]) == 6 and len(solo["Gloria"]) == 2


def test_plan_makes_one_short_run_per_character_and_skips_thin_ones(tmp_path):
    data = _set(tmp_path / "set", {"Jay Pritchett": 6, "Gloria": 2})
    main = _main(tmp_path / "set", tmp_path / "out")
    runs, skipped = CR.plan(main, data, CR.image_names(str(tmp_path / "set")), steps=500)
    assert skipped == [("Gloria", 2)]
    (r,) = runs
    assert r.lora_name == "Fam_Jay_Pritchett"
    assert r.trigger_word == "Jay Pritchett" and r.character_token == "Jay Pritchett"
    assert r.target_steps == 500 and r.image_count == 6
    assert r.sample_prompts == [] and r.network_weights == ""
    assert r.character_source == str(tmp_path / "set")
    assert Path(r.dataset_folder) == tmp_path / "out" / "Fam_Jay_Pritchett" / "dataset"
    assert main.target_steps == 2000  # the main run is untouched


def test_build_folder_copies_solo_images_with_captions(tmp_path):
    data = _set(tmp_path / "set", {"Jay": 5}, groups=[("Jay", "Manny")])
    (run,), _ = CR.plan(_main(tmp_path / "set", tmp_path / "out"), data,
                        CR.image_names(str(tmp_path / "set")))
    stale = Path(run.dataset_folder)
    stale.mkdir(parents=True)
    (stale / "old.png").write_bytes(b"x")
    assert CR.build_folder(run) == 5
    files = sorted(p.name for p in stale.iterdir())
    assert "old.png" not in files and len(files) == 10
    assert all("Manny" not in f for f in files)


def test_build_folder_refuses_uncaptioned_set(tmp_path):
    data = _set(tmp_path / "set", {"Jay": 5}, captioned=False)
    (run,), _ = CR.plan(_main(tmp_path / "set", tmp_path / "out"), data,
                        CR.image_names(str(tmp_path / "set")))
    with pytest.raises(ValueError, match="no caption"):
        CR.build_folder(run)


def test_character_run_round_trips_through_the_queue(tmp_path):
    data = _set(tmp_path / "set", {"Jay": 5})
    (run,), _ = CR.plan(_main(tmp_path / "set", tmp_path / "out"), data,
                        CR.image_names(str(tmp_path / "set")))
    assert RunDefinition.from_dict(run.to_dict()) == run


def test_batch_builds_the_folder_then_trains_without_captioning(tmp_path, monkeypatch):
    data = _set(tmp_path / "set", {"Jay": 5})
    (run,), _ = CR.plan(_main(tmp_path / "set", tmp_path / "out"), data,
                        CR.image_names(str(tmp_path / "set")))
    runner = BatchRunner()
    trained = []
    monkeypatch.setattr(runner, "_start_training", lambda r: trained.append(r) or True)
    monkeypatch.setattr(runner._captioner, "start",
                        lambda job: pytest.fail("a character run must not caption"))
    runner.start([run])
    assert trained == [run] and run.image_count == 5
    assert len(list(Path(run.dataset_folder).glob("*.png"))) == 5


def test_batch_fails_only_the_character_run_when_captions_are_missing(tmp_path):
    data = _set(tmp_path / "set", {"Jay": 5}, captioned=False)
    (run,), _ = CR.plan(_main(tmp_path / "set", tmp_path / "out"), data,
                        CR.image_names(str(tmp_path / "set")))
    runner = BatchRunner()
    runner.start([run])
    assert run.status == FAILED


# ---- Home "Style + Characters" ------------------------------------------------

def _window(monkeypatch, folder, out):
    from PySide6.QtWidgets import QMessageBox
    from ui.main_window import MainWindow
    w = MainWindow()
    queued, started, warned = [], [], []
    monkeypatch.setattr(w._dataset_tab, "get_folder_path", lambda: str(folder))
    monkeypatch.setattr(w._train_tab, "get_lora_name", lambda: "Fam")
    monkeypatch.setattr(w._train_tab, "build_run_definition",
                        lambda: (_main(folder, out), ""))
    monkeypatch.setattr(w._batch_tab, "add_run", queued.append)
    monkeypatch.setattr(w._batch_tab, "runs", lambda: [])
    monkeypatch.setattr(w._batch_tab, "start_queue",
                        lambda ask_about_captions=True: started.append(ask_about_captions))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    return w, queued, started, warned


def test_home_buttons_emit_train_only_and_caption_variants():
    from ui.home_tab import HomeTab
    t = HomeTab()
    got = []
    t.cast_run_requested.connect(got.append)
    texts = {b.text(): b for b in t.findChildren(type(t._add_batch_btn))}
    texts["🚀  Train Style + Characters"].click()
    texts["⚒  Caption + Train\nStyle + Characters"].click()
    assert got == [False, True]
    assert "⚒  Caption + Train" in texts and "⚒  Forge It" not in texts


def test_train_style_and_characters_queues_main_then_each_character(tmp_path, monkeypatch):
    _set(tmp_path / "set", {"Jay": 6, "Gloria": 7, "Manny": 1})
    w, queued, started, warned = _window(monkeypatch, tmp_path / "set", tmp_path / "out")
    monkeypatch.setattr(w._home_tab, "character_steps", lambda: 750)
    w._on_home_cast_run(False)
    assert warned == []
    assert [r.lora_name for r in queued] == ["Fam", "Fam_Jay", "Fam_Gloria"]
    assert queued[0].caption_policy == "keep"
    assert all(r.target_steps == 750 for r in queued[1:])
    assert started == [False]


def test_train_only_refuses_an_uncaptioned_set(tmp_path, monkeypatch):
    _set(tmp_path / "set", {"Jay": 6}, captioned=False)
    w, queued, started, warned = _window(monkeypatch, tmp_path / "set", tmp_path / "out")
    w._on_home_cast_run(False)
    assert queued == [] and started == [] and "no caption" in warned[0]


def test_caption_variant_accepts_an_uncaptioned_set_and_asks_about_captions(tmp_path, monkeypatch):
    _set(tmp_path / "set", {"Jay": 6}, captioned=False)
    w, queued, started, warned = _window(monkeypatch, tmp_path / "set", tmp_path / "out")
    w._on_home_cast_run(True)
    assert [r.lora_name for r in queued] == ["Fam", "Fam_Jay"]
    assert queued[0].caption_policy == "ask" and started == [True]


# ---- a character's own set beside the main one (Lora Builder export) ----------------

def _own_set(main_folder: Path, token: str, n: int, captioned=False):
    folder = main_folder.with_name(main_folder.name + "_characters") / token
    _set(folder, {token: n}, captioned=captioned)
    return folder


def test_plan_prefers_the_characters_own_set(tmp_path):
    data = _set(tmp_path / "set", {"Jay": 6, "Gloria": 2})
    own = _own_set(tmp_path / "set", "Gloria", 40)
    runs, skipped = CR.plan(_main(tmp_path / "set", tmp_path / "out"), data,
                            CR.image_names(str(tmp_path / "set")))
    by = {r.character_token: r for r in runs}
    assert skipped == []
    assert by["Gloria"].dataset_folder == str(own) and by["Gloria"].image_count == 40
    assert by["Gloria"].character_source == ""                # trains (and captions) in place
    assert by["Jay"].character_source == str(tmp_path / "set")  # no own set: main set's solo shots


def test_batch_captions_a_characters_own_set_like_any_run(tmp_path, monkeypatch):
    data = _set(tmp_path / "set", {"Jay": 1})
    own = _own_set(tmp_path / "set", "Jay", 6)
    (run,), _ = CR.plan(_main(tmp_path / "set", tmp_path / "out", caption_policy="keep"), data,
                        CR.image_names(str(tmp_path / "set")))
    runner = BatchRunner()
    jobs = []
    monkeypatch.setattr(runner._captioner, "start", lambda job: jobs.append(job) or True)
    runner.start([run])
    assert jobs[0].dataset_folder == str(own) and jobs[0].trigger == "Jay"


def test_train_only_refuses_an_uncaptioned_own_set(tmp_path, monkeypatch):
    _set(tmp_path / "set", {"Jay": 1})
    _own_set(tmp_path / "set", "Jay", 6, captioned=False)
    w, queued, started, warned = _window(monkeypatch, tmp_path / "set", tmp_path / "out")
    w._on_home_cast_run(False)
    assert queued == [] and "Jay" in warned[0]


def test_plan_auto_steps_per_character(tmp_path):
    data = _set(tmp_path / "set", {"Jay": 20, "Gloria": 60})
    runs, _ = CR.plan(_main(tmp_path / "set", tmp_path / "out", model="moss"), data,
                      CR.image_names(str(tmp_path / "set")), steps=None)
    assert {r.character_token: r.target_steps for r in runs} == {"Jay": 2000, "Gloria": 2400}


def test_plan_fixed_steps_unchanged(tmp_path):
    data = _set(tmp_path / "set", {"Jay": 20, "Gloria": 60})
    runs, _ = CR.plan(_main(tmp_path / "set", tmp_path / "out"), data,
                      CR.image_names(str(tmp_path / "set")), steps=500)
    assert [r.target_steps for r in runs] == [500, 500]


def test_cast_queue_skips_main_for_moss(tmp_path):
    data = _set(tmp_path / "set", {"Jay": 6})
    main = _main(tmp_path / "set", tmp_path / "out", model="moss")
    runs, _ = CR.plan(main, data, CR.image_names(str(tmp_path / "set")), steps=None)
    assert CR.queue_for(main, runs, True) == (runs, "")
    anima = _main(tmp_path / "set", tmp_path / "out")
    aruns, _ = CR.plan(anima, data, CR.image_names(str(tmp_path / "set")))
    assert CR.queue_for(anima, aruns, False) == ([anima, *aruns], "")


def test_refuses_uncaptioned_main_source_for_moss(tmp_path):
    data = _set(tmp_path / "set", {"Jay": 6}, captioned=False)
    main = _main(tmp_path / "set", tmp_path / "out", model="moss")
    runs, _ = CR.plan(main, data, CR.image_names(str(tmp_path / "set")), steps=None)
    assert runs[0].character_source
    queue, msg = CR.queue_for(main, runs, False)
    assert queue == [] and msg == CR.MOSS_UNCAPTIONED_MSG
    assert msg.startswith("MOSS trains only per-character LoRAs")
