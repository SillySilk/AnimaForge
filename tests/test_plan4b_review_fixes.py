import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from core import caption_policy as cp, compare
from core.batch import RunDefinition
from core.caption_runner import CaptionJob, plan_stages


def _folder(tmp_path, nl=True, tags=True, txt=True):
    ds = tmp_path / "ds"
    ds.mkdir(exist_ok=True)
    for i in range(3):
        (ds / f"a{i}.png").write_bytes(b"x")
        if nl:
            (ds / f"a{i}.nl").write_text("A man stands.", encoding="utf-8")
        if tags and i:
            (ds / f"a{i}.tags").write_text("1boy", encoding="utf-8")
        if txt:
            (ds / f"a{i}.txt").write_text("hand written", encoding="utf-8")
    return ds


def _rd(ds, **kw):
    base = dict(lora_name="Mr X", dataset_folder=str(ds), image_count=3,
                caption_style="prose+tags", trigger_word="mrx")
    base.update(kw)
    return RunDefinition(**base)


# C1/C2/I5: comparison runs fill only the missing sidecars, then build their own style.
def test_comparison_runs_fill_missing_sidecars_and_rebuild_every_caption(tmp_path):
    ds = _folder(tmp_path)                      # a0 lacks .tags; every image has .nl and .txt
    a, b, _ = compare.plan(_rd(ds), "tags", ["p"])
    assert a.caption_policy == b.caption_policy == cp.FILL
    job = a.to_caption_job("C:/sd", "", a.caption_policy)
    stages = dict(plan_stages(job, cp.scan(str(ds))))
    assert stages["tag"] == [str(ds / "a0.png")]          # only the missing .tags
    assert "describe" not in stages                        # every .nl kept, never re-described
    assert len(stages["combine"]) == 3                     # every .txt rebuilt in its own style


def test_pressing_start_never_changes_a_comparison_runs_policy(tmp_path, monkeypatch):
    from core.settings import AppSettings
    from ui.batch_tab import BatchTab
    ds = _folder(tmp_path)
    a, b, _ = compare.plan(_rd(ds), "tags", ["p"])
    tab = BatchTab()
    monkeypatch.setattr(tab, "_persist", lambda: None)
    for default in (cp.KEEP, cp.OVERWRITE):
        monkeypatch.setattr(AppSettings, "get", lambda self, k, d=default: d if k == "caption_existing_policy" else None)
        assert tab._resolve_conflicts([a, b]) is True
        assert a.caption_policy == b.caption_policy == cp.FILL


# I1: a second comparison never reuses the first one's folders.
def test_each_comparison_gets_fresh_run_names(tmp_path, monkeypatch):
    import time
    ds = _folder(tmp_path)
    a1, _b1, _ = compare.plan(_rd(ds), "tags", ["p"])
    monkeypatch.setattr(time, "strftime", lambda fmt, *a: "20991231-235959")
    a2, _b2, _ = compare.plan(_rd(ds), "tags", ["p"])
    assert a1.lora_name != a2.lora_name


# I2: Krea previews pair by the render's own prompt index, not by sort order.
def test_krea_previews_pair_by_the_folder_index(tmp_path):
    from core import paths
    from core.engines.onetrainer import render
    entry = {"id": "g", "model": "krea2", "prompts": [f"p{i}" for i in range(12)],
             "runs": [{"name": "A", "style": "prose", "output_dir": ""},
                      {"name": "B", "style": "tags", "output_dir": ""}]}
    for run in entry["runs"]:
        dest = Path(render.destination(paths.run_output_dir(str(tmp_path), run["name"])))
        for i in range(12):
            if i == 3:
                continue                       # this prompt's render failed
            (dest / f"{i} - final").mkdir(parents=True)
            (dest / f"{i} - final" / f"final-{i}.jpg").write_bytes(b"x")
    rows = compare.final_previews(entry, str(tmp_path), {})
    assert rows[3]["a"] is None
    assert rows[10]["a"].endswith("final-10.jpg") and rows[2]["b"].endswith("final-2.jpg")


# I2b/I4: the Kohya side shows the shipped epoch and says what it shows.
def test_kohya_side_shows_the_shipped_epoch(tmp_path):
    from core import paths
    entry = {"id": "g", "model": "anima", "prompts": ["p0"],
             "runs": [{"name": "A", "style": "prose", "output_dir": ""},
                      {"name": "B", "style": "tags", "output_dir": ""}]}
    for run in entry["runs"]:
        s = Path(paths.run_output_dir(str(tmp_path), run["name"])) / "sample"
        s.mkdir(parents=True)
        for e in (4, 10):
            (s / f"{run['name']}_e{e:06d}_00_1.png").write_bytes(b"x")
    rows = compare.final_previews(entry, str(tmp_path), {"A": "done", "B": "failed"},
                                  shipped={"A": 4})
    assert rows[0]["a"].endswith("_e000004_00_1.png") and rows[0]["epoch"]["a"] == 4
    assert rows[0]["epoch"]["b"] == 10


def test_view_headers_say_each_sides_status():
    from PySide6.QtWidgets import QLabel
    from ui.compare_view import CompareView
    entry = {"set": "S", "runs": [{"name": "A", "style": "prose"}, {"name": "B", "style": "tags"}]}
    rows = [{"prompt": "p", "a": None, "b": None, "status": {"a": "done", "b": "failed"},
             "epoch": {"a": 4, "b": 6}}]
    view = CompareView(entry, rows)
    texts = [l.text() for l in view.findChildren(QLabel)]
    assert any("Tags" in t and "failed" in t and "epoch 6" in t for t in texts)


# I3: comparison runs always render previews.
def test_comparison_previews_render_even_with_previews_off(tmp_path):
    from PySide6.QtCore import QSettings
    from core.settings import AppSettings
    a = AppSettings(QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    a.set("sample_enable", False)
    assert a.prepare_sample_args(str(tmp_path), "L", "", prompts=["x"]) == {}
    assert a.prepare_sample_args(str(tmp_path), "L", "", prompts=["x"], force=True)


def test_an_uppercase_seed_is_respected(tmp_path):
    from PySide6.QtCore import QSettings
    from core.settings import AppSettings
    a = AppSettings(QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    a.set("sample_enable", True)
    args = a.prepare_sample_args(str(tmp_path), "L", "", prompts=["x --D 7"])
    assert Path(args["sample_prompts"]).read_text(encoding="utf-8").lower().count("--d") == 1
