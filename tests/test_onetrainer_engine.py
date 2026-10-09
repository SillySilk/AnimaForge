from pathlib import Path
from core import sysmem
from core.engines.onetrainer import engine

def _fake_install(tmp_path):
    (tmp_path / "venv" / "Scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "venv" / "Scripts" / "python.exe").write_bytes(b"")
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "train.py").write_text("", encoding="utf-8")
    marker = tmp_path / engine.COMPRESSION_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("", encoding="utf-8")
    return tmp_path

def test_validate_refuses_an_install_that_predates_compressed_weights(tmp_path):
    """The preset trains compressed int8 weights; an OneTrainer from before that
    data type existed rejects the config minutes into a run nobody is watching."""
    root = _fake_install(tmp_path)
    (root / engine.COMPRESSION_MARKER).unlink()
    ok, why = engine.validate(str(root))
    assert not ok and "update" in why.lower()

def test_locate_finds_a_complete_install(tmp_path):
    root = _fake_install(tmp_path)
    inst = engine.locate(str(root))
    assert inst is not None
    assert inst.python == str(root / "venv" / "Scripts" / "python.exe")
    assert inst.train_script == str(root / "scripts" / "train.py")

def test_locate_rejects_missing_pieces(tmp_path):
    assert engine.locate(str(tmp_path)) is None          # empty dir
    assert engine.locate("") is None
    assert engine.locate(r"C:\definitely\not\here") is None

def test_validate_names_the_missing_piece(tmp_path):
    ok, why = engine.validate(str(tmp_path))
    assert not ok and "venv" in why
    (tmp_path / "venv" / "Scripts").mkdir(parents=True)
    (tmp_path / "venv" / "Scripts" / "python.exe").write_bytes(b"")
    ok, why = engine.validate(str(tmp_path))
    assert not ok and "train.py" in why
    _fake_install(tmp_path)
    ok, why = engine.validate(str(tmp_path))
    assert ok and why == ""

def test_setup_problems_names_what_a_pipeline_must_fix_before_it_starts(tmp_path):
    """The Forge It chain captions for an hour before it reaches preflight, so the
    gates that are already knowable have to be asked first. Captions cannot be one of
    them — they do not exist yet — but the install and the output do."""
    dataset = tmp_path / "set"
    dataset.mkdir()
    problems = engine.setup_problems("", str(dataset), 5, "", "")
    text = " ".join(problems).lower()
    assert "onetrainer" in text and "output" in text
    # ...and never the caption check, which can only fail before captioning runs.
    assert "captions" not in text


def test_an_unset_trigger_is_not_a_refusal(tmp_path):
    """It used to be, and that put a dialog in front of the one path the app exists
    to leave running unattended. The UI fills it in from the LoRA name instead."""
    root = _fake_install(tmp_path / "ot")
    dataset = tmp_path / "set"; dataset.mkdir()
    out = tmp_path / "out"; out.mkdir()
    assert engine.setup_problems(str(root), str(dataset), 5, "", str(out)) == []


def test_setup_problems_is_quiet_on_a_configured_machine(tmp_path):
    root = _fake_install(tmp_path / "onetrainer")
    dataset = tmp_path / "set"
    dataset.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    # An uncaptioned set with a trigger set: nothing here should stop the pipeline,
    # even though preflight() would refuse it for having no captions yet.
    assert engine.setup_problems(str(root), str(dataset), 5, "ariatrig", str(out)) == []
    assert any("captions" in p for p in
               engine.preflight(str(root), str(dataset), 5, "ariatrig", str(out)))


def test_preflight_collects_every_problem(tmp_path):
    problems = engine.preflight("", str(tmp_path / "nope"), 0, "", "")
    text = " ".join(problems).lower()
    assert len(problems) >= 3
    assert "onetrainer" in text and "dataset" in text and "output" in text

def test_preflight_zero_images_is_fatal_even_with_valid_folder(tmp_path):
    root = _fake_install(tmp_path / "ot")
    data = tmp_path / "data"; data.mkdir()
    out = tmp_path / "out"; out.mkdir()
    problems = engine.preflight(str(root), str(data), 0, "trig", str(out))
    assert any("no images" in p.lower() for p in problems)

def _roomy_machine(monkeypatch):
    """A machine with room on both axes preflight measures: disk and commit.

    Both have to be stubbed or these tests read the real box they run on, and
    "is there 55 GB of commit free right now" is not a property of the code.
    """
    import shutil, collections
    Usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(1e12, 0, 1e12))
    monkeypatch.setattr(sysmem, "available_commit_gb", lambda: 200.0)
    monkeypatch.setattr(sysmem, "top_committer", lambda: None)

def _captions(folder, *texts):
    for i, t in enumerate(texts):
        (folder / f"img{i}.txt").write_text(t, encoding="utf-8")

def test_preflight_clean_run_is_empty(tmp_path, monkeypatch):
    root = _fake_install(tmp_path / "ot")
    data = tmp_path / "data"; data.mkdir()
    out = tmp_path / "out"; out.mkdir()
    _captions(data, "trig, a woman in a kitchen", "trig, a woman outdoors")
    _roomy_machine(monkeypatch)
    assert engine.preflight(str(root), str(data), 12, "trig", str(out)) == []


# ---------------------------------------------------------------------------
# The trigger has to be IN the captions. A set captioned elsewhere (or before the
# trigger was set) trains prose the trigger never appears in, so the finished LoRA
# never answers to it -- and every sample renders a token training never saw. That
# cost a full run once, so it is raised before the hour is spent rather than 90
# minutes after. Said, though, not asked: it used to be a refusal, and a refusal
# here is a dialog nobody is in the room to answer. It lives in advisories() now.
# ---------------------------------------------------------------------------

def test_a_trigger_no_caption_carries_is_said_without_stopping_the_run(tmp_path, monkeypatch):
    root = _fake_install(tmp_path / "ot")
    data = tmp_path / "data"; data.mkdir()
    out = tmp_path / "out"; out.mkdir()
    _captions(data, "Beverly sits on a kitchen counter", "Billy stands by the window")
    _roomy_machine(monkeypatch)
    assert engine.preflight(str(root), str(data), 12, "BB", str(out)) == []
    said = engine.advisories(str(data), "BB")
    assert any("BB" in a and "trigger" in a.lower() for a in said), said

def test_advisory_trigger_match_is_word_bounded(tmp_path, monkeypatch):
    # "BBQ" contains the letters but is not the trigger token.
    root = _fake_install(tmp_path / "ot")
    data = tmp_path / "data"; data.mkdir()
    out = tmp_path / "out"; out.mkdir()
    _captions(data, "a BBQ in the garden")
    _roomy_machine(monkeypatch)
    assert engine.advisories(str(data), "BB")

def test_the_trigger_in_any_single_caption_is_enough(tmp_path, monkeypatch):
    root = _fake_install(tmp_path / "ot")
    data = tmp_path / "data"; data.mkdir()
    out = tmp_path / "out"; out.mkdir()
    _captions(data, "Beverly sits on a counter", "bb, Billy by the window")   # case-insensitive
    _roomy_machine(monkeypatch)
    assert engine.preflight(str(root), str(data), 12, "BB", str(out)) == []
    assert engine.advisories(str(data), "BB") == []

def test_a_dataset_with_no_captions_at_all_is_still_refused(tmp_path, monkeypatch):
    """Not an advisory. Every image would train against an empty prompt, so the run
    teaches the model nothing — an hour thrown away, which is worth a refusal even
    on the path that must not stop for anything else."""
    root = _fake_install(tmp_path / "ot")
    data = tmp_path / "data"; data.mkdir()
    out = tmp_path / "out"; out.mkdir()
    _roomy_machine(monkeypatch)
    problems = engine.preflight(str(root), str(data), 12, "BB", str(out))
    assert any("caption" in p.lower() for p in problems), problems
    # ...and it is not double-reported as something the run may carry on through.
    assert engine.advisories(str(data), "BB") == []

def test_the_trigger_check_survives_a_regex_flavoured_trigger(tmp_path, monkeypatch):
    root = _fake_install(tmp_path / "ot")
    data = tmp_path / "data"; data.mkdir()
    out = tmp_path / "out"; out.mkdir()
    _captions(data, "a.b, a woman in a kitchen")
    _roomy_machine(monkeypatch)
    assert engine.advisories(str(data), "a.b") == []
    _captions(data, "axb, a woman in a kitchen")   # would match if '.' stayed a wildcard
    assert engine.advisories(str(data), "a.b")


# ---------------------------------------------------------------------------
# Commit charge -- RAM plus pagefile -- is the resource a 12B run actually
# exhausts, and it is invisible to every VRAM check the app makes. Run out of it
# and PyTorch's allocation returns null, c10.dll faults, and Windows kills
# python.exe with an access violation: no traceback, no OOM marker, trainer.log
# stops mid-progress-bar. Said before the run, never a refusal: refusing was
# tried, and a run left overnight came back untrained behind a dialog while most
# runs under the line would simply have loaded slower.

def _tight_machine(monkeypatch, free_gb, top=None):
    import shutil, collections
    Usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(1e12, 0, 1e12))
    monkeypatch.setattr(sysmem, "available_commit_gb", lambda: free_gb)
    monkeypatch.setattr(sysmem, "top_committer", lambda: top)


def _clean_run(tmp_path):
    root = _fake_install(tmp_path / "ot")
    data = tmp_path / "data"; data.mkdir()
    out = tmp_path / "out"; out.mkdir()
    _captions(data, "trig, a woman in a kitchen")
    return str(root), str(data), str(out)


def test_a_run_the_system_may_not_be_able_to_commit_is_warned_not_refused(tmp_path, monkeypatch):
    root, data, out = _clean_run(tmp_path)
    _tight_machine(monkeypatch, 21.0)
    assert engine.preflight(root, data, 12, "trig", out) == []
    said = engine.advisories(data, "trig")
    assert len(said) == 1
    text = said[0]
    assert "21" in text and str(engine.MIN_FREE_COMMIT_GB) in text
    assert "memory" in text.lower()
    assert "starting anyway" in text.lower()


def test_the_memory_warning_names_the_process_the_user_could_close(tmp_path, monkeypatch):
    root, data, out = _clean_run(tmp_path)
    _tight_machine(monkeypatch, 21.0,
                   top=(r"D:\Tools\WebUI\venv\Scripts\Python.exe launch.py", 52.2))
    # The note is composed by sysmem, so the real one has to run over the stub.
    monkeypatch.setattr(sysmem, "biggest_consumer_note",
                        sysmem.biggest_consumer_note)
    said = engine.advisories(data, "trig")
    assert said and "WebUI" in said[0]
    assert "52 GB" in said[0]


def test_the_memory_warning_ignores_a_small_process_and_points_at_the_pagefile(tmp_path, monkeypatch):
    """Nothing worth closing means the machine itself is too small -- say that
    instead of telling the user to close a 300 MB process."""
    root, data, out = _clean_run(tmp_path)
    _tight_machine(monkeypatch, 12.0, top=("notepad.exe", 0.3))
    said = engine.advisories(data, "trig")
    assert said and "pagefile" in said[0].lower()
    assert "notepad" not in said[0]


def test_a_probe_that_cannot_answer_says_nothing(tmp_path, monkeypatch):
    """None means the OS declined to report, not that memory is short."""
    root, data, out = _clean_run(tmp_path)
    _tight_machine(monkeypatch, None)
    assert engine.preflight(root, data, 12, "trig", out) == []
    assert engine.advisories(data, "trig") == []


def test_room_to_spare_says_nothing(tmp_path, monkeypatch):
    root, data, out = _clean_run(tmp_path)
    _tight_machine(monkeypatch, float(engine.MIN_FREE_COMMIT_GB))
    assert engine.preflight(root, data, 12, "trig", out) == []
    assert engine.advisories(data, "trig") == []
