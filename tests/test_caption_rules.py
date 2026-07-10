from core.caption_rules import apply_caption_rules, dump_rules, parse_rules
from core.dataset_manager import combine_all, combine_caption


def test_whole_word_only_man_does_not_eat_woman_or_human():
    rules = [("man", "woman")]
    assert apply_caption_rules("a man and a woman", rules) == "a woman and a woman"
    assert apply_caption_rules("a human being", rules) == "a human being"
    assert apply_caption_rules("romantic command", rules) == "romantic command"


def test_capitalization_is_carried_over():
    assert apply_caption_rules("Man standing", [("man", "woman")]) == "Woman standing"
    assert apply_caption_rules("MAN standing", [("man", "woman")]) == "WOMAN standing"


def test_empty_replacement_bans_a_tag_and_tidies_the_commas():
    assert apply_caption_rules("1girl, 1boy, solo", [("1boy", "")]) == "1girl, solo"
    assert apply_caption_rules("1boy, solo", [("1boy", "")]) == "solo"
    assert apply_caption_rules("1girl, 1boy", [("1boy", "")]) == "1girl"


def test_underscored_and_multiword_terms():
    assert apply_caption_rules("score_7, safe", [("score_7", "")]) == "safe"
    assert apply_caption_rules("long hair, solo", [("long hair", "short hair")]) == \
        "short hair, solo"


def test_rules_apply_in_order():
    assert apply_caption_rules("a man", [("man", "boy"), ("boy", "child")]) == "a child"


def test_no_rules_returns_text_unchanged():
    assert apply_caption_rules("1girl, solo", []) == "1girl, solo"


def test_parse_rules_never_raises_on_garbage():
    assert parse_rules("not json") == []
    assert parse_rules(None) == []
    assert parse_rules('{"not": "a list"}') == []
    assert parse_rules('[{"find": "  ", "replace": "x"}]') == []   # blank find is dropped
    assert parse_rules('[{"find": "1boy"}]') == [("1boy", "")]


def test_roundtrip():
    rules = [("man", "woman"), ("1boy", "")]
    assert parse_rules(dump_rules(rules)) == rules


def test_rules_never_touch_the_trigger_prefix_or_lead():
    """A rule banning 'man' must not delete a character named Man, nor the trigger."""
    out = combine_caption(nl="a man stands", tags="1boy, solo",
                          prefix="manbag, masterpiece", lead="Man",
                          rules=[("man", ""), ("1boy", "")])
    assert out.startswith("manbag, masterpiece, Man")   # prefix + lead intact
    assert "1boy" not in out


def test_combine_all_applies_rules_but_leaves_sidecars_untouched(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "a.nl").write_text("a man stands", encoding="utf-8")
    (tmp_path / "a.tags").write_text("1boy, solo", encoding="utf-8")

    written, errors = combine_all(str(tmp_path), rules=[("man", "woman"), ("1boy", "")])

    assert written == 1
    assert errors == 0
    txt = (tmp_path / "a.txt").read_text(encoding="utf-8")
    assert "1boy" not in txt
    assert "woman" in txt
    # sidecars are pristine — combine never mutates its own inputs
    assert (tmp_path / "a.nl").read_text(encoding="utf-8") == "a man stands"
    assert (tmp_path / "a.tags").read_text(encoding="utf-8") == "1boy, solo"
