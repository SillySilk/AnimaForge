import pytest

from core import config_overrides as co


def test_flatten_makes_dotted_paths_and_keeps_lists_whole():
    cfg = {"a": 1, "b": {"c": True, "d": {"e": "x"}}, "f": [1, 2]}
    assert co.flatten(cfg) == {"a": 1, "b.c": True, "b.d.e": "x", "f": [1, 2]}


def test_apply_sets_nested_values_on_a_copy():
    cfg = {"t": {"lr": 1.0}, "k": 1}
    out = co.apply(cfg, {"t.lr": 0.5, "t.new": "y", "fresh.deep": 3})
    assert out == {"t": {"lr": 0.5, "new": "y"}, "k": 1, "fresh": {"deep": 3}}
    assert cfg == {"t": {"lr": 1.0}, "k": 1}


def test_apply_with_nothing_is_an_equal_copy():
    cfg = {"t": {"lr": 1.0}}
    out = co.apply(cfg, {})
    assert out == cfg and out is not cfg


def test_apply_refuses_a_path_through_a_value():
    with pytest.raises(ValueError, match="k.x"):
        co.apply({"k": 1}, {"k.x": 2})


@pytest.mark.parametrize("text,like,want", [
    ("true", False, True), ("No", True, False), ("12", 3, 12), ("0.5", 1.0, 0.5),
    ("2", 1.0, 2.0), ("[1, 2]", [], [1, 2]), ('{"a": 1}', {}, {"a": 1}),
    ("hello", "x", "hello"), ("3", None, 3), ("abc", None, "abc"),
])
def test_parse_follows_the_existing_type(text, like, want):
    assert co.parse(text, like) == want


@pytest.mark.parametrize("text,like", [("true", 3), ("12x", 1.0), ("maybe", True), ("{", [])])
def test_bad_value_for_a_typed_setting_is_refused(text, like):
    with pytest.raises(ValueError):
        co.parse(text, like)


def test_format_value_round_trips():
    for v in (True, 3, 0.25, [1, "a"], {"k": 1}, "plain text"):
        assert co.parse(co.format_value(v), v) == v
