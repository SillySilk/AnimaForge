import json

from PySide6.QtCore import QSettings

from core import kreaforge_import as ki
from core.batch import RunDefinition
from core.settings import AppSettings


def _krea(tmp_path):
    root = tmp_path / "Krea Forge"
    data = tmp_path / "data"
    for name, imgs in (("Barry", 3), ("Klaus", 2)):
        ds = data / name
        ds.mkdir(parents=True)
        for i in range(imgs):
            (ds / f"{name}_{i}.png").write_bytes(b"x")
        run = root / "output" / name
        run.mkdir(parents=True)
        (run / "config.json").write_text("{}", encoding="utf-8")
        (run / "concepts.json").write_text(json.dumps([{"name": name, "path": str(ds)}]),
                                           encoding="utf-8")
        (run / f"{name}.trained-words.txt").write_text(
            f"Trained words — {name}\n\n    {name.lower()}x, 1boy\n\n"
            f"    {name.lower()}x  —  the set's trigger word — in 3 of 3 captions\n",
            encoding="utf-8")
    snap = root / "sets" / "Barry" / "captions-processed"
    snap.mkdir(parents=True)
    (snap / "Barry_0.txt").write_text("barryx, a man", encoding="utf-8")
    (root / "output" / "batch_status.json").write_text("{}", encoding="utf-8")
    return root


def _stores(tmp_path):
    src = QSettings(str(tmp_path / "krea.ini"), QSettings.IniFormat)
    dst = AppSettings(QSettings(str(tmp_path / "anima.ini"), QSettings.IniFormat))
    return src, dst


def test_settings_fill_only_empty_keys(tmp_path):
    src, dst = _stores(tmp_path)
    src.setValue("onetrainer_root", "D:/OT")
    src.setValue("forge_lora_dir", "G:/LORAS")
    src.setValue("ui_font_mode", "custom")
    src.setValue("ui_font_family", "Pirata One")
    src.setValue("debug_profiles", True)
    src.setValue("dit_path", "C:/stale/anima.safetensors")
    src.setValue("make_room_before_run", True)
    dst.set("forge_lora_dir", "E:/mine")
    copied = ki.import_settings(src, dst)
    assert sorted(copied) == ["debug_profiles", "onetrainer_root", "ui_font_family", "ui_font_mode"]
    assert dst.get("forge_lora_dir") == "E:/mine"
    assert dst.get("onetrainer_root") == "D:/OT" and dst.get("debug_profiles") is True
    assert dst.get("dit_path") == ""


def test_runs_become_krea2_sets_with_their_dataset_and_trigger(tmp_path):
    root = _krea(tmp_path)
    sets_root = tmp_path / "anima"
    created = ki.import_runs(root, sets_root)
    assert sorted(created) == ["Barry", "Klaus"]
    rd = RunDefinition.from_dict(json.loads((sets_root / "sets" / "Barry.json").read_text(encoding="utf-8")))
    assert rd.model == "krea2" and rd.lora_name == "Barry"
    assert rd.trigger_word == "barryx" and rd.image_count == 3
    assert rd.dataset_folder.endswith("Barry") and rd.output_dir == str(root / "output")
    assert (sets_root / "sets" / "Barry" / "captions-processed" / "Barry_0.txt").is_file()


def test_a_name_taken_by_an_anima_set_is_imported_beside_it(tmp_path):
    from core import sets
    root = _krea(tmp_path)
    sets_root = tmp_path / "anima"
    mine = sets.save_set("Barry", RunDefinition(lora_name="Barry", dataset_folder="X", image_count=1),
                         root=sets_root)
    before = mine.read_bytes()
    created = ki.import_runs(root, sets_root)
    assert "Barry - Krea 2" in created
    assert mine.read_bytes() == before


def test_import_is_idempotent_and_never_overwrites(tmp_path):
    root = _krea(tmp_path)
    sets_root = tmp_path / "anima"
    src, dst = _stores(tmp_path)
    first = ki.run_once(dst, root, sets_root, src)
    assert first
    snapshot = {p.name: p.read_bytes() for p in (sets_root / "sets").glob("*.json")}
    assert ki.run_once(dst, root, sets_root, src) == []
    assert {p.name: p.read_bytes() for p in (sets_root / "sets").glob("*.json")} == snapshot


def test_no_krea_forge_means_nothing_happens(tmp_path):
    src, dst = _stores(tmp_path)
    assert ki.run_once(dst, tmp_path / "nowhere", tmp_path / "anima", src) == []
    assert not (tmp_path / "anima").exists()
    assert not dst.get("kreaforge_imported")
