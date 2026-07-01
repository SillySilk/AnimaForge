import json
from pathlib import Path

import sys
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from core import characters as C


def test_character_roundtrip():
    c = C.Character(token="asuka", description="red hair, red plugsuit")
    assert C.Character.from_dict(c.to_dict()) == c


def test_dataset_characters_roundtrip():
    d = C.DatasetCharacters(
        roster=[C.Character("asuka", "red hair"), C.Character("rei", "blue hair")],
        style_anchor="@mystyle",
        assignments={"img_001.png": {"present": ["asuka"], "oneoffs": []}},
    )
    back = C.DatasetCharacters.from_dict(d.to_dict())
    assert back == d


def test_load_missing_returns_empty(tmp_path):
    data = C.load(str(tmp_path))
    assert data.roster == [] and data.style_anchor == "" and data.assignments == {}


def test_save_then_load(tmp_path):
    d = C.DatasetCharacters(roster=[C.Character("rei", "blue hair")], style_anchor="@s")
    C.save(str(tmp_path), d)
    on_disk = json.loads((tmp_path / C.FILENAME).read_text(encoding="utf-8"))
    assert on_disk["roster"][0]["token"] == "rei"
    assert C.load(str(tmp_path)) == d


def test_load_file_corrupt_returns_empty(tmp_path):
    p = tmp_path / C.FILENAME
    p.write_text("{not json", encoding="utf-8")
    assert C.load_file(str(p)) == C.DatasetCharacters()


# ----------------------------------------------------------------------
# Cast resolution
# ----------------------------------------------------------------------

def _sample():
    return C.DatasetCharacters(
        roster=[C.Character("asuka", "red hair"), C.Character("rei", "blue hair")],
        style_anchor="@mystyle",
        assignments={
            "assigned.png": {"present": ["asuka"],
                             "oneoffs": [{"token": "gendo", "description": "older man"}]},
            "empty.png": {"present": [], "oneoffs": []},
        },
    )


def test_present_unassigned_returns_full_roster():
    data = _sample()
    toks = [c.token for c in C.present_for_image(data, "never_seen.png")]
    assert toks == ["asuka", "rei"]


def test_present_assigned_resolves_present_and_oneoffs():
    data = _sample()
    toks = [c.token for c in C.present_for_image(data, "assigned.png")]
    assert toks == ["asuka", "gendo"]


def test_present_empty_assignment_returns_nothing():
    assert C.present_for_image(_sample(), "empty.png") == []


def test_explicit_tokens_unassigned_is_empty():
    assert C.explicit_tokens_for_image(_sample(), "never_seen.png") == []


def test_explicit_tokens_assigned():
    assert C.explicit_tokens_for_image(_sample(), "assigned.png") == ["asuka", "gendo"]


def test_explicit_tokens_skips_unknown_present_token():
    data = C.DatasetCharacters(
        roster=[C.Character("asuka", "red hair")],
        assignments={"x.png": {"present": ["asuka", "ghost"], "oneoffs": []}},
    )
    assert C.explicit_tokens_for_image(data, "x.png") == ["asuka"]


def test_build_character_block_full():
    data = _sample()
    block = C.build_character_block(C.present_for_image(data, "assigned.png"), data.style_anchor)
    assert "<characters>" in block
    assert "asuka: red hair" in block
    assert "gendo: older man" in block
    assert "<style_anchor>@mystyle</style_anchor>" in block


def test_build_character_block_omits_empty():
    assert C.build_character_block([], "") == ""
    assert C.build_character_block([], "@only") == "<style_anchor>@only</style_anchor>"


def test_build_character_block_token_without_description():
    block = C.build_character_block([C.Character("solo", "")], "")
    assert "<characters>\nsolo\n</characters>" == block


# ----------------------------------------------------------------------
# Deterministic enforcer
# ----------------------------------------------------------------------

def test_enforce_adds_missing_token_and_anchor():
    out = C.enforce_anchors_in_tags("1girl, solo", ["asuka"], "@mystyle")
    assert out == "1girl, solo, asuka, @mystyle"


def test_enforce_is_case_insensitive_and_idempotent():
    once = C.enforce_anchors_in_tags("1girl, Asuka", ["asuka"], "")
    assert once == "1girl, Asuka"                      # already present (case-insensitive)
    twice = C.enforce_anchors_in_tags(once, ["asuka"], "")
    assert twice == once                               # idempotent


def test_enforce_preserves_order_and_ignores_blank_anchor():
    assert C.enforce_anchors_in_tags("b, a", ["c"], "") == "b, a, c"
    assert C.enforce_anchors_in_tags("a", [], "   ") == "a"


def test_enforce_handles_empty_tags():
    assert C.enforce_anchors_in_tags("", ["asuka"], "@s") == "asuka, @s"


# ---- filename-based detection ----

def test_parse_character_name_basic():
    assert C.parse_character_name("sarah_01") == "sarah"
    assert C.parse_character_name("sarah 1") == "sarah"
    assert C.parse_character_name("sarah-2") == "sarah"
    assert C.parse_character_name("red_riding_3") == "red riding"


def test_parse_character_name_rejects_non_character():
    assert C.parse_character_name("landscape") is None     # no trailing number
    assert C.parse_character_name("001") is None            # no name
    assert C.parse_character_name("") is None


def test_characters_from_filenames_groups_and_orders():
    names = ["sarah_01.png", "sarah_02.png", "john_1.jpg", "scenery.png"]
    tokens, assignments = C.characters_from_filenames(names)
    assert tokens == ["sarah", "john"]                      # ordered, unique
    assert assignments["sarah_02.png"] == {"present": ["sarah"], "oneoffs": []}
    assert "scenery.png" not in assignments                 # non-matching skipped


def test_merge_detected_is_idempotent_and_keeps_descriptions():
    data = C.DatasetCharacters(roster=[C.Character("sarah", "red hair")])
    tokens, assignments = C.characters_from_filenames(["sarah_01.png", "john_01.png"])
    C.merge_detected(data, tokens, assignments)
    C.merge_detected(data, tokens, assignments)             # twice -> no dupes
    toks = [c.token for c in data.roster]
    assert toks == ["sarah", "john"]
    assert data.roster[0].description == "red hair"         # existing desc preserved
    assert data.assignments["john_01.png"]["present"] == ["john"]


# ---- natural-language correction ops ----

def test_apply_ops_rename_updates_roster_and_assignments():
    data = C.DatasetCharacters(
        roster=[C.Character("jane", "red hair")],
        assignments={"img5.png": {"present": ["jane"], "oneoffs": []}},
    )
    C.apply_ops(data, [{"op": "rename", "from": "jane", "to": "sarah"}])
    assert data.roster[0].token == "sarah"
    assert data.assignments["img5.png"]["present"] == ["sarah"]


def test_apply_ops_add_remove_describe_set_present():
    data = C.DatasetCharacters(roster=[C.Character("a")])
    C.apply_ops(data, [
        {"op": "add", "token": "b", "description": "blue"},
        {"op": "set_description", "token": "a", "description": "tall"},
        {"op": "set_present", "image": "x.png", "present": ["a", "b"]},
        {"op": "remove", "token": "a"},
    ])
    toks = {c.token: c.description for c in data.roster}
    assert toks == {"b": "blue"}                            # a removed, b added
    assert data.assignments["x.png"]["present"] == ["b"]    # 'a' purged from present


def test_apply_ops_ignores_unknown_and_malformed():
    data = C.DatasetCharacters(roster=[C.Character("a")])
    C.apply_ops(data, [{"op": "frobnicate"}, "not a dict", {"op": "rename", "from": "a"}])
    assert [c.token for c in data.roster] == ["a"]          # unchanged


# ---- Character Doctor find/replace ----

def test_replace_token_scoped_to_selected_images():
    data = C.DatasetCharacters(
        roster=[C.Character("jane")],
        assignments={
            "a.png": {"present": ["jane"], "oneoffs": []},
            "b.png": {"present": ["jane"], "oneoffs": []},
            "c.png": {"present": ["jane"], "oneoffs": []},
        },
    )
    changed = C.replace_token(data, "jane", "sarah", image_names=["a.png", "b.png"])
    assert changed == 2
    assert data.assignments["a.png"]["present"] == ["sarah"]
    assert data.assignments["b.png"]["present"] == ["sarah"]
    assert data.assignments["c.png"]["present"] == ["jane"]   # out of scope, untouched
    assert "sarah" in {c.token for c in data.roster}          # auto-added to roster


def test_replace_token_all_images_and_dedup():
    data = C.DatasetCharacters(
        roster=[C.Character("jane"), C.Character("sarah")],
        assignments={"a.png": {"present": ["jane", "sarah"], "oneoffs": []}},
    )
    changed = C.replace_token(data, "jane", "sarah")          # None scope = all
    assert changed == 1
    assert data.assignments["a.png"]["present"] == ["sarah"]  # de-duplicated


def test_replace_token_noops_on_empty_or_same():
    data = C.DatasetCharacters(assignments={"a.png": {"present": ["x"], "oneoffs": []}})
    assert C.replace_token(data, "", "y") == 0
    assert C.replace_token(data, "x", "x") == 0
    assert data.assignments["a.png"]["present"] == ["x"]


# ---- Task 1: hint field + membership/example helpers ----

def test_character_hint_roundtrips():
    from core.characters import Character
    c = Character("sarah", "red hair", "redhead")
    assert c.to_dict() == {"token": "sarah", "description": "red hair", "hint": "redhead",
                           "role": "subject"}
    back = Character.from_dict({"token": "sarah", "description": "red hair", "hint": "redhead"})
    assert back.hint == "redhead"
    # back-compat: missing hint -> ""
    assert Character.from_dict({"token": "x"}).hint == ""


def test_example_images_for_token():
    from core import characters as ch
    data = ch.DatasetCharacters(
        roster=[ch.Character("a"), ch.Character("b")],
        assignments={
            "1.png": {"present": ["a"], "oneoffs": []},
            "2.png": {"present": ["a", "b"], "oneoffs": []},
            "3.png": {"present": ["b"], "oneoffs": []},
        },
    )
    names = ["1.png", "2.png", "3.png"]
    assert ch.example_images_for_token(data, names, "a") == ["1.png", "2.png"]
    assert ch.example_images_for_token(data, names, "a", limit=1) == ["1.png"]
    assert ch.example_images_for_token(data, names, "z") == []


def test_add_and_remove_token_to_images():
    from core import characters as ch
    data = ch.DatasetCharacters(roster=[ch.Character("a")],
                                assignments={"1.png": {"present": ["a"], "oneoffs": []}})
    # add a brand-new token to two images (one already has 'a'); target auto-added to roster
    n = ch.add_token_to_images(data, ["1.png", "2.png"], "b")
    assert n == 2
    assert "b" in {c.token for c in data.roster}
    assert data.assignments["1.png"]["present"] == ["a", "b"]
    assert data.assignments["2.png"]["present"] == ["b"]
    # adding again is idempotent (no dupes, returns 0 changed)
    assert ch.add_token_to_images(data, ["1.png"], "b") == 0
    # remove
    removed = ch.remove_token_from_images(data, ["1.png"], "b")
    assert removed == 1
    assert data.assignments["1.png"]["present"] == ["a"]


def test_split_off_new_character():
    from core import characters as ch
    data = ch.DatasetCharacters(
        roster=[ch.Character("blonde_knight")],
        assignments={
            "1.png": {"present": ["blonde_knight"], "oneoffs": []},
            "2.png": {"present": ["blonde_knight"], "oneoffs": []},
        },
    )
    # the AI merged two people under 'blonde_knight'; split 2.png into 'mara'
    changed = ch.split_off_new_character(data, ["2.png"], "mara", old_token="blonde_knight")
    assert changed == 1
    assert "mara" in {c.token for c in data.roster}
    assert data.assignments["2.png"]["present"] == ["mara"]
    assert data.assignments["1.png"]["present"] == ["blonde_knight"]


def _data_with_assignments():
    return C.DatasetCharacters(
        roster=[C.Character("amber"), C.Character("bess")],
        assignments={
            "a1.png": {"present": ["amber"], "oneoffs": []},
            "a2.png": {"present": ["amber"], "oneoffs": []},
            "both.png": {"present": ["amber", "bess"], "oneoffs": []},
            "scene.png": {"present": [], "oneoffs": []},
            "unseen.png": None,  # never reviewed -> not scenery
        },
    )


def test_together_combinations_groups_multi_present():
    data = _data_with_assignments()
    names = ["a1.png", "a2.png", "both.png", "scene.png"]
    combos = C.together_combinations(data, names)
    assert len(combos) == 1
    assert combos[0]["tokens"] == ["amber", "bess"]
    assert combos[0]["count"] == 1
    assert combos[0]["examples"] == ["both.png"]


def test_scenery_images_only_explicit_empty():
    data = _data_with_assignments()
    names = ["a1.png", "both.png", "scene.png", "unseen.png"]
    # unseen.png has a None assignment in the dict; drop it like a real scan would
    data.assignments.pop("unseen.png", None)
    assert C.scenery_images(data, names) == ["scene.png"]


def test_scenery_excludes_unidentified():
    data = C.DatasetCharacters(assignments={
        "scene.png": {"present": [], "oneoffs": []},
        "person.png": {"present": [], "oneoffs": [], "unidentified": True},
    })
    assert C.scenery_images(data, ["scene.png", "person.png"]) == ["scene.png"]


def test_stable_color_is_deterministic_hex():
    c1 = C.stable_color_for("amber")
    assert c1 == C.stable_color_for("amber")
    assert c1.startswith("#") and len(c1) == 7
    assert C.stable_color_for("amber") != C.stable_color_for("bess")


def test_role_defaults_subject_and_roundtrips():
    c = C.Character(token="amber")
    assert c.role == "subject"
    back = C.Character.from_dict({"token": "bess", "role": "label"})
    assert back.role == "label"
    # unknown role falls back to subject
    assert C.Character.from_dict({"token": "x", "role": "bogus"}).role == "subject"


def test_set_role():
    data = C.DatasetCharacters(roster=[C.Character("amber"), C.Character("bess")])
    assert C.set_role(data, "bess", "label") is True
    assert next(c for c in data.roster if c.token == "bess").role == "label"
    assert C.set_role(data, "nobody", "label") is False


def test_unrecognized_images_only_flagged_empty():
    data = C.DatasetCharacters(assignments={
        "scene.png": {"present": [], "oneoffs": []},
        "person.png": {"present": [], "oneoffs": [], "unidentified": True},
        "amber.png": {"present": ["amber"], "oneoffs": []},
    })
    assert C.unrecognized_images(data, ["scene.png", "person.png", "amber.png"]) == ["person.png"]


def test_naming_an_unrecognized_image_clears_the_sentinel():
    data = C.DatasetCharacters(assignments={
        "person.png": {"present": [], "oneoffs": [], "unidentified": True},
    })
    C.add_token_to_images(data, ["person.png"], "amber")
    assert data.assignments["person.png"]["present"] == ["amber"]
    assert data.assignments["person.png"]["unidentified"] is False
    assert C.unrecognized_images(data, ["person.png"]) == []
