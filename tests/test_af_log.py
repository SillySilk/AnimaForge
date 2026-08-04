import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import af_log


def test_append_creates_file_and_writes_line(tmp_path):
    p = tmp_path / "caption.log"
    af_log.append("hello", path=p)
    assert "hello" in p.read_text(encoding="utf-8")


def test_append_creates_missing_parent_dir(tmp_path):
    p = tmp_path / "logs" / "caption.log"
    af_log.append("hello", path=p)
    assert p.is_file()


def test_append_stamps_each_line(tmp_path):
    p = tmp_path / "caption.log"
    af_log.append("hello", path=p)
    line = p.read_text(encoding="utf-8").strip()
    # "2026-08-02 14:03:11  hello"
    assert line.endswith("hello")
    assert line[:4].isdigit()


def test_append_is_utf8_safe(tmp_path):
    p = tmp_path / "caption.log"
    af_log.append("caption — 日本語 ✓", path=p)
    assert "日本語" in p.read_text(encoding="utf-8")


def test_append_accumulates_lines(tmp_path):
    p = tmp_path / "caption.log"
    for i in range(3):
        af_log.append(f"line {i}", path=p)
    assert len(p.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_rotation_moves_current_to_previous(tmp_path):
    p = tmp_path / "caption.log"
    af_log.append("x" * 200, path=p, max_bytes=100)
    af_log.append("fresh", path=p, max_bytes=100)
    assert p.with_suffix(".log.1").is_file()
    assert p.read_text(encoding="utf-8").strip().endswith("fresh")


def test_rotation_replaces_an_existing_previous(tmp_path):
    p = tmp_path / "caption.log"
    p.with_suffix(".log.1").write_text("old", encoding="utf-8")
    af_log.append("y" * 200, path=p, max_bytes=100)
    af_log.append("fresh", path=p, max_bytes=100)
    assert "old" not in p.with_suffix(".log.1").read_text(encoding="utf-8")


def test_append_never_raises_on_a_bad_path(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    af_log.append("hello", path=blocker / "nested" / "caption.log")  # must not raise


# --- RunLog: the training log on disk ---------------------------------------
#
# The training log was a QTextEdit and nothing else, so a run that died at exit
# code 1 left no artifact to send to a bug report. Three rounds of support on one
# report produced no log at all. RunLog writes one file per run, flushed per line
# so a hard crash still leaves the tail behind.


def test_runlog_writes_lines_to_disk(tmp_path):
    p = tmp_path / "training.log"
    log = af_log.RunLog(path=p)
    log.open("run started")
    log.write("steps: 1/100")
    log.close("done")
    text = p.read_text(encoding="utf-8")
    assert "run started" in text and "steps: 1/100" in text and "done" in text


def test_runlog_flushes_without_close(tmp_path):
    # The crash case: the process dies before close() runs.
    p = tmp_path / "training.log"
    log = af_log.RunLog(path=p)
    log.open("run started")
    log.write("last thing before the crash")
    assert "last thing before the crash" in p.read_text(encoding="utf-8")


def test_runlog_truncates_each_run(tmp_path):
    # One run per file: the previous run moves aside so 'the log' is never an
    # ambiguous pile of runs.
    p = tmp_path / "training.log"
    first = af_log.RunLog(path=p)
    first.open("run one")
    first.close("")
    second = af_log.RunLog(path=p)
    second.open("run two")
    second.close("")
    assert "run one" not in p.read_text(encoding="utf-8")
    assert "run one" in p.with_suffix(".log.1").read_text(encoding="utf-8")


def test_runlog_is_utf8_safe(tmp_path):
    p = tmp_path / "training.log"
    log = af_log.RunLog(path=p)
    log.open("start")
    log.write("学習開始 ✓")
    log.close("")
    assert "学習開始" in p.read_text(encoding="utf-8")


def test_runlog_never_raises_on_an_unusable_path(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    log = af_log.RunLog(path=blocker / "nested" / "training.log")
    log.open("start")     # must not raise
    log.write("line")     # must not raise
    log.close("end")      # must not raise


def test_runlog_write_before_open_is_harmless(tmp_path):
    af_log.RunLog(path=tmp_path / "training.log").write("stray")  # must not raise


def test_train_log_path_is_beside_the_caption_log():
    assert af_log.train_log_path().parent == af_log.log_path().parent
    assert af_log.train_log_path().name == "training.log"
