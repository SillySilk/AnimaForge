from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from core import added_sets, consolidate
from core.batch import RunDefinition
from core.engines.onetrainer.adapter import to_onetrainer_params


def _img(folder: Path, stem: str, caption: str = "", caches=False):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.png").write_bytes(b"x")
    if caption:
        (folder / f"{stem}.txt").write_text(caption, encoding="utf-8")
        (folder / f"{stem}.nl").write_text(caption, encoding="utf-8")
    if caches:
        (folder / f"{stem}_1280x1280_moss.npz").write_bytes(b"latent")
        (folder / f"{stem}_moss_te.npz").write_bytes(b"text")


@pytest.fixture
def cast(tmp_path):
    ds = tmp_path / "Simpsons"
    _img(ds, "Marge_001_Character", "Simpsons, Marge, Marge stands in a kitchen.", caches=True)
    _img(ds, "Marge_002_Character", "Simpsons, Marge, Marge Simpson waves.", caches=True)
    _img(ds, "Marge Simpson_001_Character", "Simpsons, Marge Simpson, by the couch.",
         caches=True)
    _img(ds, "Marge-Burns_001_Character", "Simpsons, Marge, Burns, marge glares at Burns.",
         caches=True)
    _img(ds, "Bart_001_Character", "Simpsons, Bart, Bart skates past Margery.", caches=True)
    return ds


# ---- Consolidate ----

def test_consolidate_renames_files_and_reserializes_the_merged_group(cast):
    done = consolidate.consolidate(str(cast), ["Marge", "Marge Simpson"], "Marge Simpson")
    names = sorted(p.name for p in cast.glob("*.png"))
    assert names == ["Bart_001_Character.png", "Marge Simpson-Burns_001_Character.png",
                     "Marge Simpson_001_Character.png", "Marge Simpson_002_Character.png",
                     "Marge Simpson_003_Character.png"]
    assert done["renamed"] == 3                 # the untouched Marge Simpson file kept its name
    for png in cast.glob("Marge Simpson_*.png"):
        assert png.with_suffix(".txt").is_file() and png.with_suffix(".nl").is_file()


def test_consolidate_rewrites_whole_names_only_without_doubling(cast):
    consolidate.consolidate(str(cast), ["Marge", "Marge Simpson"], "Marge Simpson")
    texts = [p.read_text(encoding="utf-8") for p in cast.glob("*.txt")]
    joined = "\n".join(texts)
    assert "Marge Simpson Simpson" not in joined
    assert "Margery" in joined                   # not a whole-word match
    assert "Simpsons, Marge Simpson, Burns, Marge Simpson glares at Burns." in texts
    assert not any(" Marge," in t or t.endswith("Marge.") for t in texts)


def test_consolidate_keeps_image_caches_and_drops_stale_text_caches(cast):
    consolidate.consolidate(str(cast), ["Marge"], "Marge Simpson")
    stems = {p.stem for p in cast.glob("*.png")}
    for npz in cast.glob("*_1280x1280_moss.npz"):
        assert npz.name.replace("_1280x1280_moss.npz", "") in stems   # moved with its picture
    te = {p.name for p in cast.glob("*_te.npz")}
    # Only captions that changed lose their cached text; one already reading
    # "Marge Simpson" keeps it.
    assert te == {"Bart_001_Character_moss_te.npz", "Marge Simpson_001_Character_moss_te.npz"}


def test_consolidate_refuses_names_that_break_file_names(cast):
    with pytest.raises(ValueError):
        consolidate.consolidate(str(cast), ["Marge"], "Marge_S")
    with pytest.raises(ValueError):
        consolidate.consolidate(str(cast), [], "Marge")


def test_characters_tab_consolidate_button_follows_the_checkboxes(cast, monkeypatch):
    from ui.characters_tab import CharactersTab
    t = CharactersTab()
    t._folder_path = str(cast)
    t._render()
    assert set(t._picked) == {"Marge", "Marge Simpson", "Bart"}   # solo bundles only
    assert not t._consolidate_btn.isEnabled()
    t._picked["Marge"].setChecked(True)
    assert t._consolidate_btn.isEnabled()
    t.set_busy_check(lambda: "training is in progress")
    from PySide6.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    t._consolidate()
    assert warned and "training is in progress" in warned[0]
    assert (cast / "Marge_001_Character.png").is_file()      # nothing renamed


# ---- Boost ----

def test_subjects_counts_group_shots_for_everyone_in_them(cast):
    assert added_sets.subjects(str(cast)) == [
        ("Marge", 3), ("Bart", 1), ("Burns", 1), ("Marge Simpson", 1)]


def test_boost_folders_are_weighted_so_the_total_is_the_emphasis(cast):
    folders = added_sets.training_folders(str(cast), [], ["Marge"], 3.0)
    assert folders == [(str(cast / "_boost_Marge"), 3, 2.0)]
    assert added_sets.weighted_count(5, str(cast), [], 3.0, ["Marge"]) == 11
    assert added_sets.training_folders(str(cast), [], ["Marge"], 1.0) == []


def test_prepare_boosts_rebuilds_copies_and_drops_unboosted_ones(cast):
    added_sets.prepare_boosts(str(cast), ["Marge", "Bart"])
    marge = cast / "_boost_Marge"
    assert sorted(p.name for p in marge.glob("*.png")) == [
        "Marge-Burns_001_Character.png", "Marge_001_Character.png", "Marge_002_Character.png"]
    assert (marge / "Marge_001_Character.txt").is_file()
    assert (marge / "Marge_001_Character_1280x1280_moss.npz").is_file()
    added_sets.prepare_boosts(str(cast), ["Bart"])
    assert not marge.exists() and (cast / "_boost_Bart").is_dir()
    assert len(list(cast.glob("*.png"))) == 5               # the set itself untouched


def test_onetrainer_gets_a_concept_per_boost(cast, monkeypatch):
    monkeypatch.setattr(added_sets, "emphasis", lambda settings=None: 3.0)
    p = to_onetrainer_params(RunDefinition("S", str(cast), 5, model="krea2",
                                           boosted=["Bart"]))
    assert p.added_concepts == [(str(cast / "_boost_Bart"), 2.0)]
    assert p.plan_image_count == 5 + 2
    assert (cast / "_boost_Bart" / "Bart_001_Character.png").is_file()


def test_train_tab_launch_builds_boost_copies_and_kohya_subsets(cast, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from core import gpu_check, sets
    from ui.train_tab import TrainTab
    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(added_sets, "emphasis", lambda settings=None: 3.0)
    monkeypatch.setattr(gpu_check, "free_vram_mb", lambda: None)
    monkeypatch.setattr(sets, "_project_root", lambda: tmp_path)
    t = TrainTab()
    t._output_dir = str(tmp_path / "out")
    t.set_gpu_busy_check(lambda: "")
    monkeypatch.setattr(t, "_validate_for_training", lambda: (True, ""))
    monkeypatch.setattr(t, "_validate_for_config", lambda: (True, ""))
    monkeypatch.setattr(t, "_check_undersized_images", lambda interactive=True: True)
    monkeypatch.setattr(t, "_check_empty_captions", lambda interactive=True: True)
    configs = []
    (tmp_path / "c.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr("ui.train_tab.generate_configs",
                        lambda **kw: configs.append(kw) or (str(tmp_path / "c.toml"),
                                                            str(tmp_path / "d.toml")))
    monkeypatch.setattr(t._trainer, "start", lambda *a, **k: None)
    t.apply_run_definition(RunDefinition("S", str(cast), 5, target_steps=1000,
                                         output_dir=str(tmp_path / "out"), boosted=["Marge"]))
    t._start_training(confirm=False)
    subsets = configs[-1]["added_subsets"]
    repeats = configs[-1]["repeats"]
    assert subsets == [(str(cast / "_boost_Marge"), repeats * 2)]
    assert (cast / "_boost_Marge" / "Marge_002_Character.png").is_file()
