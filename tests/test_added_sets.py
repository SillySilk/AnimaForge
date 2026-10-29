import json
from pathlib import Path

import pytest
import toml
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from core import added_sets
from core.batch import RunDefinition
from core.config_generator import generate_configs
from core.engines.onetrainer import concepts
from core.engines.onetrainer.adapter import to_onetrainer_params


def _pics(folder: Path, n: int, captions=False, prefix="p"):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"{prefix}{i}.png").write_bytes(b"x")
        if captions:
            (folder / f"{prefix}{i}.txt").write_text("a caption", encoding="utf-8")
    return folder


class _Settings:
    def __init__(self, w):
        self.w = w

    def get(self, key):
        assert key == added_sets.SETTING
        return self.w


def test_add_pictures_copies_them_and_their_captions_into_a_named_folder(tmp_path):
    ds = _pics(tmp_path / "set", 2)
    src = _pics(tmp_path / "dina", 3, captions=True, prefix="d")
    (src / "notes.doc").write_bytes(b"x")
    sub, copied = added_sets.add_pictures(str(ds), "Dina", [str(p) for p in src.iterdir()])
    assert (sub, copied) == ("Dina", 3)
    names = sorted(p.name for p in (ds / "Dina").iterdir())
    assert names == ["d0.png", "d0.txt", "d1.png", "d1.txt", "d2.png", "d2.txt"]
    assert len(list(src.iterdir())) == 7                 # originals left in place
    assert added_sets.uncaptioned(str(ds / "Dina")) == 0


def test_uncaptioned_counts_pictures_without_a_caption(tmp_path):
    folder = _pics(tmp_path / "x", 3)
    (folder / "p0.txt").write_text("cap", encoding="utf-8")
    (folder / "p1.txt").write_text("  ", encoding="utf-8")
    assert added_sets.uncaptioned(str(folder)) == 2


def test_an_empty_name_is_refused(tmp_path):
    ds = _pics(tmp_path / "set", 1)
    with pytest.raises(ValueError):
        added_sets.add_pictures(str(ds), "  ", [])


def test_weighted_count_counts_added_pictures_at_their_weight(tmp_path):
    ds = _pics(tmp_path / "set", 380)
    _pics(ds / "Dina", 100)
    assert added_sets.weighted_count(380, str(ds), ["Dina", "Missing"], 3.0) == 680
    assert added_sets.folders(str(ds), ["Missing"]) == []
    assert added_sets.emphasis(_Settings(3.0)) == 3.0
    assert added_sets.emphasis(_Settings(0.2)) == 1.0     # never weighted down


def test_kohya_dataset_gets_one_subset_per_added_folder(tmp_path):
    ds = _pics(tmp_path / "set", 4)
    dina = _pics(ds / "Dina", 2)
    _main, dataset_cfg = generate_configs(
        output_dir=str(tmp_path / "run"), lora_name="S", dit_path="d", qwen3_path="q",
        vae_path="v", dataset_folder=str(ds), epochs=10, repeats=2,
        added_subsets=[(str(dina), 6)])
    subsets = toml.load(dataset_cfg)["datasets"][0]["subsets"]
    assert [(Path(s["image_dir"]).name, s["num_repeats"]) for s in subsets] == [
        ("set", 2), ("Dina", 6)]
    assert added_sets.kohya_repeats(2, 3.0) == 6


def test_onetrainer_gets_one_concept_per_added_folder(tmp_path):
    out = concepts.build_concepts(str(tmp_path / "set"), [(str(tmp_path / "set" / "Dina"), 3.0)])
    assert [(c["name"], c["balancing"]) for c in out] == [("set", 1.0), ("Dina", 3.0)]


def test_adapter_hands_added_folders_and_the_weighted_plan_to_onetrainer(tmp_path, monkeypatch):
    monkeypatch.setattr(added_sets, "emphasis", lambda settings=None: 3.0)
    ds = _pics(tmp_path / "set", 20)
    _pics(ds / "Dina", 10)
    rd = RunDefinition("S", str(ds), 20, model="krea2", added_folders=["Dina"])
    p = to_onetrainer_params(rd)
    assert p.added_concepts == [(str(ds / "Dina"), 3.0)]
    assert (p.image_count, p.plan_image_count) == (20, 50)
    assert to_onetrainer_params(RunDefinition("S", str(ds), 20, model="krea2")).plan_image_count == 0


def test_added_folders_survive_a_round_trip_through_run_json():
    rd = RunDefinition("S", "C:/set", 3, added_folders=["Dina"])
    assert RunDefinition.from_dict(rd.to_dict()).added_folders == ["Dina"]
    assert RunDefinition.from_dict({"lora_name": "S", "dataset_folder": "", "image_count": 0}
                                   ).added_folders == []


# ---- the Train tab flow ----

class FakeRunner(QObject):
    log_line = Signal(str)
    finished = Signal(bool)
    last = None

    def __init__(self, parent=None):
        super().__init__(parent)
        FakeRunner.last = self
        self.job = None

    def start(self, job):
        self.job = job
        return True


@pytest.fixture
def krea_tab(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
    from test_past_runs import _onetrainer_run
    from test_train_tab_onetrainer import _krea_tab
    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(added_sets, "emphasis", lambda settings=None: 3.0)
    t, s = _krea_tab(tmp_path, monkeypatch)
    ds = _pics(tmp_path / "simpsons", 20, captions=True)
    _onetrainer_run(tmp_path / "out", "Simpsons", ds)
    src = _pics(tmp_path / "incoming", 10, prefix="d")
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        lambda *a, **k: ([str(p) for p in src.iterdir()], ""))
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Dina", True))
    monkeypatch.setattr(QInputDialog, "getInt", lambda *a, **k: (3000, True))
    monkeypatch.setattr("core.caption_runner.CaptionRunner", FakeRunner)
    t.refresh_past_runs()
    return t, s, ds


def test_add_captions_the_new_pictures_then_trains_further_with_them(krea_tab):
    t, s, ds = krea_tab
    t._add_to_run(t._past_runs[0])
    job = FakeRunner.last.job
    assert job.dataset_folder == str(ds / "Dina") and job.trigger == "Dina"
    assert job.prefix.startswith("cat_x")                 # the set's own trigger leads
    assert s.params is None and t.is_training()           # waits for the captions
    FakeRunner.last.finished.emit(True)
    assert s.params.continue_epoch == 57
    assert s.params.added_concepts == [(str(ds / "Dina"), 3.0)]
    assert s.params.plan_image_count == 20 + 30
    assert s.params.target_steps == 1368 + 3000


def test_failed_captioning_trains_nothing_and_keeps_the_pictures(krea_tab):
    t, s, ds = krea_tab
    t._add_to_run(t._past_runs[0])
    FakeRunner.last.finished.emit(False)
    assert s.params is None and not t.is_training()
    assert len(list((ds / "Dina").glob("*.png"))) == 10


def test_captioned_pictures_skip_straight_to_training(krea_tab, tmp_path):
    t, s, ds = krea_tab
    for p in (tmp_path / "incoming").glob("*.png"):
        p.with_suffix(".txt").write_text("Dina, a woman", encoding="utf-8")
    FakeRunner.last = None
    t._add_to_run(t._past_runs[0])
    assert FakeRunner.last is None
    assert s.params.added_concepts == [(str(ds / "Dina"), 3.0)]
