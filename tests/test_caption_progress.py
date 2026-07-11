import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.caption_progress import parse_progress, ProgressTick


def test_joycaption_line():
    t = parse_progress("[JoyCaption] (3/12) aria_03.png: a woman in a garden…")
    assert t == ProgressTick(phase="Describe", done=3, total=12, filename="aria_03.png")


def test_joycaption_line_filename_with_space():
    t = parse_progress("[JoyCaption] (5/12) my photo.png: a woman in a garden…")
    assert t == ProgressTick(phase="Describe", done=5, total=12, filename="my photo.png")


def test_tagger_done_line_is_none():
    assert parse_progress("[Tagger] Tagging done.") is None


def test_phase_line_without_count_is_none():
    assert parse_progress("[JoyCaption] Captioning 12 image(s)…") is None


def test_skip_and_garbage_and_empty_are_none():
    assert parse_progress("[JoyCaption] SKIP aria.png: no caption produced") is None
    assert parse_progress("random output line") is None
    assert parse_progress("") is None
