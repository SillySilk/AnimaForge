import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

from core import sets
from core.batch import DONE, FAILED, QUEUED, RUNNING, RunDefinition, load_queue, save_queue
from remote import control, folders
from remote import server as rs


def _img(path: Path, color="red", size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    out = tmp_path / "out"
    paths = control.Paths(root, root / "batch_queue.json", out / "batch_status.json",
                          root / "logs" / "remote_batch.log", root)
    anima = RunDefinition(lora_name="Old", dataset_folder="x", image_count=3, model="anima",
                          subject_type="character", output_dir=str(out), trigger_word="old",
                          sample_prompts=["a"], network_weights="w.safetensors",
                          style_anchor="anchor", compare_group="g", status=DONE)
    krea = RunDefinition(lora_name="OldK", dataset_folder="x", image_count=3, model="krea2",
                         output_dir=str(out), target_steps=1234)
    sets.save_set("AnimaSet", anima, root)
    sets.save_set("KreaSet", krea, root)
    data = tmp_path / "data"
    for i in range(5):
        _img(data / f"{i}.png")
    return paths, data, out


# ---- templates and the queue ------------------------------------------------------

def test_a_new_run_keeps_the_sets_settings_and_drops_its_run_specific_bits(env):
    paths, data, out = env
    ok, msg = control.add_run(paths, str(data), "Fresh", "fr3sh", "AnimaSet")
    assert ok, msg
    (run,) = load_queue(paths.queue)
    assert (run.lora_name, run.trigger_word, run.image_count) == ("Fresh", "fr3sh", 5)
    assert run.dataset_folder == str(data) and run.output_dir == str(out)
    assert run.sample_prompts == [] and run.network_weights == "" and run.style_anchor == ""
    assert run.compare_group == "" and run.status == QUEUED
    from core.step_calculator import suggest_target_steps
    assert run.target_steps == suggest_target_steps("character", 5)


def test_a_krea_run_sizes_itself(env):
    paths, data, _out = env
    assert control.add_run(paths, str(data), "K", "", "KreaSet")[0]
    assert load_queue(paths.queue)[0].target_steps == 0


@pytest.mark.parametrize("folder,name,template,needle", [
    ("data", "", "AnimaSet", "name"),
    ("missing", "N", "AnimaSet", "folder"),
    ("empty", "N", "AnimaSet", "no images"),
    ("data", "N", "Nope", "saved set"),
])
def test_bad_new_runs_say_why(env, folder, name, template, needle):
    paths, data, _out = env
    (data.parent / "empty").mkdir()
    ok, msg = control.add_run(paths, str(data.parent / folder), name, "", template)
    assert not ok and needle in msg.lower()
    assert load_queue(paths.queue) == []


def test_names_are_never_reused(env):
    paths, data, out = env
    assert control.add_run(paths, str(data), "Dup", "", "AnimaSet")[0]
    ok, msg = control.add_run(paths, str(data), "dup", "", "AnimaSet")
    assert not ok and "already in the queue" in msg
    (out / "Taken").mkdir(parents=True)
    ok, msg = control.add_run(paths, str(data), "Taken", "", "AnimaSet")
    assert not ok and "output folder" in msg


def test_remove_run(env):
    paths, data, _out = env
    control.add_run(paths, str(data), "A", "", "AnimaSet")
    control.add_run(paths, str(data), "B", "", "AnimaSet")
    assert control.remove_run(paths, 0)[0]
    assert [r.lora_name for r in load_queue(paths.queue)] == ["B"]
    assert not control.remove_run(paths, 5)[0]


def test_templates_list_their_model(env):
    paths, _data, _out = env
    t = {d["name"]: d for d in control.list_templates(paths)}
    assert t["AnimaSet"]["model_name"] == "Anima" and t["KreaSet"]["model_name"] == "Krea 2"


# ---- status --------------------------------------------------------------------------

def _queue(paths, *specs):
    runs = [RunDefinition(lora_name=n, dataset_folder="d", image_count=4, status=s,
                          output_dir=str(paths.status.parent)) for n, s in specs]
    save_queue(str(paths.queue), runs)
    return runs


def test_live_status_lays_over_the_queue_while_running(env):
    paths, _data, _out = env
    _queue(paths, ("A", DONE), ("B", QUEUED))
    paths.status.parent.mkdir(parents=True, exist_ok=True)
    paths.status.write_text(json.dumps([
        {"lora_name": "A", "state": "done"},
        {"lora_name": "B", "state": "running", "phase": "training",
         "current_step": 40, "target_steps": 400}]), encoding="utf-8")
    rows = control.run_rows(paths, running=True)
    assert rows[1]["state"] == RUNNING and rows[1]["current_step"] == 40
    assert rows[1]["phase"] == "training"


def test_a_run_left_running_by_a_dead_runner_shows_as_queued(env):
    paths, _data, _out = env
    _queue(paths, ("A", RUNNING))
    assert control.run_rows(paths, running=False)[0]["state"] == QUEUED


def test_failed_runs_carry_their_error(env):
    paths, _data, _out = env
    _queue(paths, ("A", FAILED))
    paths.status.parent.mkdir(parents=True, exist_ok=True)
    paths.status.write_text(json.dumps([{"lora_name": "A", "state": "failed",
                                         "error_tail": ["CUDA out of memory"]}]), encoding="utf-8")
    assert control.run_rows(paths, False)[0]["error_tail"] == ["CUDA out of memory"]


def test_kohya_loss_comes_from_that_runs_part_of_the_log(env):
    paths, _data, _out = env
    a, b = _queue(paths, ("A", DONE), ("B", RUNNING))
    paths.log.parent.mkdir(parents=True)
    paths.log.write_text(
        "[Batch] (1/2) starting 'A'\n"
        "steps:  10%|#  | 10/100 [00:10<01:30,  1.00it/s, avr_loss=0.9]\n"
        "[Batch] (2/2) starting 'B'\n"
        "steps:  10%|#  | 10/100 [00:10<01:30,  1.00it/s, avr_loss=0.2]\r"
        "steps:  20%|## | 20/100 [00:20<01:20,  1.00it/s, avr_loss=0.1]\n", encoding="utf-8")
    assert control.loss_points(paths, b) == [[10, 0.2], [20, 0.1]]
    assert control.loss_points(paths, a) == [[10, 0.9]]


def test_onetrainer_loss_comes_from_its_own_log(env):
    paths, _data, out = env
    run = RunDefinition(lora_name="K", dataset_folder="d", image_count=4, model="krea2",
                        output_dir=str(out))
    log = out / "K" / "trainer.log"
    log.parent.mkdir(parents=True)
    log.write_text("step: 1%|#| 1/100 [00:01<00:10, loss=0.5, smooth loss=0.5]\n"
                   "step: 2%|#| 2/100 [00:02<00:10, loss=0.4, smooth loss=0.4]\n", encoding="utf-8")
    pts = control.loss_points(paths, run)
    assert [p[1] for p in pts] == [0.5, 0.4]
    assert "2/100" in control.log_tail(paths, run)[-1]


def test_long_curves_are_thinned():
    pts = [[i, 1.0 / (i + 1)] for i in range(1000)]
    thin = control._thin(pts)
    assert len(thin) == control.LOSS_POINTS and thin[0] == pts[0] and thin[-1] == pts[-1]


def test_run_files_lists_previews_and_loras_but_not_engine_state(env):
    paths, _data, out = env
    run = RunDefinition(lora_name="R", dataset_folder="d", image_count=4, output_dir=str(out))
    rd = out / "R"
    _img(rd / "sample" / "R_e000001_00_x.png")
    _img(rd / "R-state" / "junk.png")
    _img(rd / "workspace" / "cache" / "junk.png")
    (rd / "R.safetensors").write_bytes(b"lora")
    files = control.run_files(paths, run)
    assert files["previews"] == ["sample/R_e000001_00_x.png"]
    assert [f["name"] for f in files["loras"]] == ["R.safetensors"]
    assert control.run_file(paths, run, "R.safetensors") == (rd / "R.safetensors").resolve()
    assert control.run_file(paths, run, "../R/../../secret") is None


# ---- the runner process ------------------------------------------------------------

_FAKE_MAIN = """
import sys, time
print("[Batch] (1/1) starting 'A'", flush=True)
print("steps:  10%|#  | 10/100 [00:10<01:30,  1.00it/s, avr_loss=0.3]", flush=True)
time.sleep(60)
"""


def test_start_finds_and_stops_the_runner(env):
    paths, _data, _out = env
    (paths.root / "main.py").write_text(_FAKE_MAIN, encoding="utf-8")
    _queue(paths, ("A", RUNNING))
    proc = control.start_runner(paths, control.runner_command(paths, sys.executable))
    try:
        deadline = time.time() + 15
        while time.time() < deadline and not control.find_app_processes(paths.root)[1]:
            time.sleep(0.2)
        gui, runners = control.find_app_processes(paths.root)
        assert runners and not gui
        while time.time() < deadline and "avr_loss" not in paths.log.read_text(encoding="utf-8"):
            time.sleep(0.2)
        assert "starting 'A'" in paths.log.read_text(encoding="utf-8")
        assert control.stop_runner(paths) >= 1
        assert proc.wait(timeout=10) is not None
        assert not control.find_app_processes(paths.root)[1]
        assert load_queue(paths.queue)[0].status == QUEUED
    finally:
        if proc.poll() is None:
            proc.kill()


def test_the_desktop_app_is_told_apart_from_the_runner(env):
    paths, _data, _out = env
    (paths.root / "main.py").write_text("import time; time.sleep(60)", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(paths.root / "main.py")])
    try:
        deadline = time.time() + 15
        while time.time() < deadline and not control.find_app_processes(paths.root)[0]:
            time.sleep(0.2)
        gui, runners = control.find_app_processes(paths.root)
        assert gui and not runners
    finally:
        proc.kill()


# ---- folders -------------------------------------------------------------------------

def test_browse_lists_folders_and_a_few_images(tmp_path):
    for i in range(20):
        _img(tmp_path / "set" / f"{i:02d}.png")
    (tmp_path / "set" / "sub").mkdir()
    (tmp_path / "set" / "notes.txt").write_text("x")
    d = folders.browse(str(tmp_path / "set"))
    assert d["image_count"] == 20 and len(d["images"]) == folders.SAMPLE_IMAGES
    assert [f["name"] for f in d["folders"]] == ["sub"]
    assert d["parent"] == str(tmp_path)
    assert folders.browse("relative/path") is None
    assert folders.browse("")["folders"]                 # the drives


# ---- HTTP ------------------------------------------------------------------------------

class _Fake:
    def __init__(self):
        self.gui, self.runners, self.gpu, self.started, self.stopped = [], [], [], 0, 0

    def find(self, _root):
        return self.gui, self.runners

    def start(self, _paths):
        self.started += 1

    def stop(self, _paths):
        self.stopped += 1
        return 1


@pytest.fixture
def site(env, tmp_path):
    paths, data, out = env
    fake = _Fake()
    dash = rs.Dashboard(paths, tmp_path / "thumbs", gpu_apps=lambda: fake.gpu,
                        start=fake.start, stop=fake.stop, find=fake.find)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), rs.make_handler(dash, {"pc.example.ts.net"}))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, fake, paths, data, out
    httpd.shutdown()


def _get(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {})) as r:
        return r.status, dict(r.headers), r.read()


def _post(url, body=None, headers=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), headers=h,
                                 method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except ValueError:
            return e.code, {}


def test_page_and_state_load(site):
    base, *_ = site
    assert b"Lora Forge Remote" in _get(base + "/")[2]
    st = json.loads(_get(base + "/api/state")[2])
    assert st["running"] is False and st["runs"] == []


def test_unknown_hosts_and_foreign_origins_are_refused(site):
    base, *_ = site
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/api/state", {"Host": "evil.example"})
    assert e.value.code == 403
    assert _get(base + "/api/state", {"Host": "pc.example.ts.net"})[0] == 200
    assert _post(base + "/api/start", {}, {"Origin": "https://evil.example"})[0] == 403
    req = urllib.request.Request(base + "/api/start", data=b"x", method="POST",
                                 headers={"Content-Type": "text/plain"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 415


def test_queue_then_start_with_a_gpu_warning(site):
    base, fake, paths, data, _out = site
    code, r = _post(base + "/api/queue", {"folder": str(data), "name": "Web", "template": "AnimaSet"})
    assert code == 200 and r["ok"], r
    fake.gpu = ["Forge Neo"]
    code, r = _post(base + "/api/start")
    assert code == 409 and r["message"] == "gpu:Forge Neo" and fake.started == 0
    code, r = _post(base + "/api/start", {"force": True})
    assert r["ok"] and fake.started == 1


def test_nothing_changes_while_the_app_is_open_or_running(site):
    base, fake, _paths, data, _out = site
    fake.gui = [object()]
    code, r = _post(base + "/api/queue", {"folder": str(data), "name": "W", "template": "AnimaSet"})
    assert code == 409 and "open on the PC" in r["message"]
    fake.gui, fake.runners = [], [object()]
    code, r = _post(base + "/api/start")
    assert code == 409 and "running" in r["message"] and fake.started == 0
    assert _post(base + "/api/stop")[1]["ok"] and fake.stopped == 1


def test_start_with_nothing_queued_says_so(site):
    base, *_ = site
    code, r = _post(base + "/api/start")
    assert code == 409 and "Nothing" in r["message"]


def test_run_detail_files_and_thumbs(site):
    base, _fake, paths, data, out = site
    _post(base + "/api/queue", {"folder": str(data), "name": "D", "template": "AnimaSet"})
    _img(out / "D" / "sample" / "D_e000001_00_x.png", size=(800, 600))
    (out / "D" / "D.safetensors").write_bytes(b"0123456789")
    d = json.loads(_get(base + "/api/run?index=0")[2])
    assert d["files"]["previews"] == ["sample/D_e000001_00_x.png"]
    _s, h, body = _get(base + "/run-thumb?index=0&name=sample/D_e000001_00_x.png")
    assert h["Content-Type"] == "image/webp"
    status, h, part = _get(base + "/run-file?index=0&name=D.safetensors&download=1",
                           {"Range": "bytes=4-"})
    assert status == 206 and part == b"456789" and h["Content-Disposition"].startswith("attachment")
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/run-file?index=0&name=..%2F..%2Fapp%2Fbatch_queue.json")
    assert e.value.code == 404


def test_folder_browsing_and_any_image_thumbnail(site):
    base, _fake, _paths, data, _out = site
    d = json.loads(_get(base + "/api/folders?path=" + urllib.request.quote(str(data)))[2])
    assert d["image_count"] == 5
    assert _get(base + "/thumb?path=" + urllib.request.quote(d["images"][0]))[0] == 200
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/thumb?path=" + urllib.request.quote(str(data / "nope.txt")))
    assert e.value.code == 404
