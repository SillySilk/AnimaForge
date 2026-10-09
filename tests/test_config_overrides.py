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


def test_lists_of_groups_are_addressed_by_index():
    cfg = {"datasets": [{"res": 1, "subsets": [{"image_dir": "A", "num_repeats": 10}]}], "tags": [1, 2]}
    flat = co.flatten(cfg)
    assert flat == {"datasets.0.res": 1, "datasets.0.subsets.0.image_dir": "A",
                    "datasets.0.subsets.0.num_repeats": 10, "tags": [1, 2]}
    out = co.apply(cfg, {"datasets.0.subsets.0.caption_dropout_rate": 0.1})
    assert out["datasets"][0]["subsets"][0] == {"image_dir": "A", "num_repeats": 10,
                                                "caption_dropout_rate": 0.1}


def test_a_list_index_out_of_range_is_refused():
    with pytest.raises(ValueError, match="datasets.3"):
        co.apply({"datasets": [{"a": 1}]}, {"datasets.3.a": 2})


@pytest.mark.parametrize("text,want", [("0.5", 0.5), ("8", 8), ("8.0", 8)])
def test_a_whole_number_setting_accepts_a_fraction(text, want):
    got = co.parse(text, 8)
    assert got == want and type(got) is type(want)


def test_locked_paths_match_patterns():
    pats = ("training_arguments.output_dir", "dataset_toml.datasets.*.subsets.*.image_dir", "model_arguments.*")
    assert co.is_locked("training_arguments.output_dir", pats)
    assert co.is_locked("dataset_toml.datasets.0.subsets.0.image_dir", pats)
    assert co.is_locked("model_arguments.vae", pats)
    assert not co.is_locked("training_arguments.noise_offset", pats)


@pytest.mark.parametrize("path", ["t", "t.lr.x", "a..b", "a.", ".a", ""])
def test_a_new_path_that_breaks_the_structure_is_refused(path):
    assert co.check_new_path(path, {"t.lr": 1.0, "k": 2}) != ""


def test_a_new_path_inside_a_group_is_fine():
    assert co.check_new_path("t.noise", {"t.lr": 1.0}) == ""
    assert co.check_new_path("fresh.deep", {"t.lr": 1.0}) == ""
