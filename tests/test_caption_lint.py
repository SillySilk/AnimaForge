import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.caption_lint import lint_caption, format_warning


def _families(lint):
    return {family for family, _values in lint.conflicts}


def test_conflicting_hair_colors_flagged():
    lint = lint_caption("mychar, 1girl, black hair, smiling, brown hair")
    assert ("hair color", ["black", "brown"]) in lint.conflicts


def test_single_hair_color_is_clean():
    lint = lint_caption("mychar, 1girl, brown hair, smiling")
    assert lint.conflicts == []


def test_conflicting_eye_colors_flagged():
    lint = lint_caption("1girl, blue eyes, portrait, green eyes")
    assert ("eye color", ["blue", "green"]) in lint.conflicts


def test_conflict_found_across_prose_and_tags():
    # The combined .txt mixes NL prose and booru tags — a conflict between the
    # sentence and the tag tail is exactly the customer's reported case.
    lint = lint_caption("Her long black hair shines in the sun. 1girl, brown hair")
    assert ("hair color", ["black", "brown"]) in lint.conflicts


def test_multicolored_hair_suppresses_hair_conflict():
    lint = lint_caption("1girl, multicolored hair, pink hair, blue hair")
    assert "hair color" not in _families(lint)


def test_heterochromia_suppresses_eye_conflict():
    lint = lint_caption("1girl, heterochromia, blue eyes, green eyes")
    assert "eye color" not in _families(lint)


def test_hair_length_conflict_flagged():
    lint = lint_caption("1girl, short hair, long hair")
    assert ("hair length", ["long", "short"]) in lint.conflicts


def test_multi_person_flag():
    assert lint_caption("2girls, black hair, blonde hair").multi_person is True
    assert lint_caption("1girl, black hair").multi_person is False


def test_empty_caption_is_clean():
    lint = lint_caption("")
    assert lint.conflicts == [] and lint.multi_person is False


def test_format_warning_mentions_values_and_multi_person():
    warn = format_warning(lint_caption("2girls, black hair, blonde hair"))
    assert "black" in warn and "blonde" in warn
    assert "hair color" in warn
    assert "multiple" in warn.lower()   # softened: may be intentional with 2+ people


def test_format_warning_empty_when_clean():
    assert format_warning(lint_caption("1girl, brown hair, blue eyes")) == ""
