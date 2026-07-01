import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import naming as N


# ---- parse_name ----

def test_parse_single_subject_with_space():
    assert N.parse_name("Yogi bear_001_Character") == {
        "subjects": ["Yogi bear"], "serial": "001", "category": "Character"}


def test_parse_multi_subject_hyphen():
    out = N.parse_name("Homer-Marge-Lisa_001_Character")
    assert out["subjects"] == ["Homer", "Marge", "Lisa"]
    assert out["serial"] == "001" and out["category"] == "Character"


def test_parse_category_case_insensitive_canonical():
    assert N.parse_name("Tim Burton_002_style")["category"] == "Style"
    assert N.parse_name("Picnic basket_012_OBJECT")["category"] == "Object"


def test_parse_invalid():
    assert N.parse_name("Homer_Character") is None            # only 2 fields
    assert N.parse_name("Homer_01_Character") is None          # serial < 3 digits
    assert N.parse_name("Homer_001_Dog") is None               # unknown category
    assert N.parse_name("_001_Character") is None              # empty name
    assert N.parse_name("Homer_001_Character_x") is None       # 4 fields
    assert N.parse_name("") is None


# ---- project_category / validate ----

def test_project_category_dominant():
    stems = ["a_001_Character", "b_002_Character", "c_001_Style"]
    assert N.project_category(stems) == "Character"
    assert N.project_category(["junk", "more junk"]) is None


def test_validate_folder_flags_offcategory_and_malformed():
    names = ["Bart_001_Character.png", "Homer-Marge_002_Character.jpg",
             "scene.png", "Tim_001_Style.png"]
    res = N.validate_folder(names)
    assert res["category"] == "Character"
    assert set(res["valid"]) == {"Bart_001_Character.png", "Homer-Marge_002_Character.jpg"}
    bad = {d["name"]: d["reason"] for d in res["invalid"]}
    assert "scene.png" in bad and "Tim_001_Style.png" in bad
    assert "category" in bad["Tim_001_Style.png"].lower()


def test_assignments_from_names_uses_subjects():
    names = ["Yogi bear_001_Character.png", "Homer-Marge_002_Character.png", "scene.png"]
    tokens, assignments = N.assignments_from_names(names)
    assert tokens == ["Yogi bear", "Homer", "Marge"]
    assert assignments["Homer-Marge_002_Character.png"]["present"] == ["Homer", "Marge"]
    assert "scene.png" not in assignments


# ---- bundles_from_names ----

def test_bundles_groups_solo_and_combined():
    names = ["Homer_001_Character.png", "Homer_002_Character.png",
             "Homer-Marge_003_Character.png", "Lisa_001_Character.png", "scene.png"]
    b = N.bundles_from_names(names)
    assert b["category"] == "Character"
    solo = {g["name"]: g["images"] for g in b["solo"]}
    assert solo["Homer"] == ["Homer_001_Character.png", "Homer_002_Character.png"]
    assert solo["Lisa"] == ["Lisa_001_Character.png"]
    assert "Marge" not in solo                      # only ever appears multi-subject
    assert len(b["combined"]) == 1
    combo = b["combined"][0]
    assert combo["name"] == "Homer + Marge"
    assert combo["subjects"] == ["Homer", "Marge"]
    assert combo["images"] == ["Homer-Marge_003_Character.png"]
    assert b["needs_naming"] == ["scene.png"]


def test_bundles_wrong_category_is_needs_naming():
    names = ["Bart_001_Character.png", "Tim_001_Style.png"]
    b = N.bundles_from_names(names)
    assert b["category"] == "Character"
    assert [g["name"] for g in b["solo"]] == ["Bart"]
    assert b["needs_naming"] == ["Tim_001_Style.png"]


def test_bundles_empty():
    b = N.bundles_from_names([])
    assert b == {"category": None, "solo": [], "combined": [], "needs_naming": []}


# ---- rename_image / write_characters ----

def test_rename_image_moves_image_and_sidecars(tmp_path):
    (tmp_path / "bart.png").write_bytes(b"x")
    (tmp_path / "bart.txt").write_text("caption", encoding="utf-8")
    (tmp_path / "bart.tags").write_text("tags", encoding="utf-8")
    out = N.rename_image(str(tmp_path), "bart.png", "Bart_001_Character.png")
    assert out == "Bart_001_Character.png"
    assert (tmp_path / "Bart_001_Character.png").is_file()
    assert (tmp_path / "Bart_001_Character.txt").read_text(encoding="utf-8") == "caption"
    assert (tmp_path / "Bart_001_Character.tags").is_file()
    assert not (tmp_path / "bart.png").exists()


def test_rename_image_target_exists_raises(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"y")
    raised = False
    try:
        N.rename_image(str(tmp_path), "a.png", "b.png")
    except FileExistsError:
        raised = True
    assert raised and (tmp_path / "a.png").is_file()


def test_write_characters_from_names(tmp_path):
    from core import characters as ch
    names = ["Homer-Marge_001_Character.png", "Lisa_002_Character.png", "scene.png"]
    N.write_characters_from_names(str(tmp_path), names)
    data = ch.load(str(tmp_path))
    assert [c.token for c in data.roster] == ["Homer", "Marge", "Lisa"]
    assert data.assignments["Homer-Marge_001_Character.png"]["present"] == ["Homer", "Marge"]
    assert "scene.png" not in data.assignments


# ---- auto_format ----

def test_auto_format_assigns_serials_and_canonical_category(tmp_path):
    (tmp_path / "Homer-Marge_001_Character.png").write_bytes(b"x")  # already correct
    (tmp_path / "Homer-Marge.png").write_bytes(b"x")                # missing serial+category
    (tmp_path / "Bart_002_character.jpg").write_bytes(b"x")         # wrong-case category
    names = sorted(p.name for p in tmp_path.iterdir())
    N.auto_format(str(tmp_path), names)
    after = sorted(p.name for p in tmp_path.iterdir())
    # Homer-Marge group -> 001 and 002 Character; Bart -> 001 Character (canonical case)
    assert "Homer-Marge_001_Character.png" in after
    assert "Homer-Marge_002_Character.png" in after
    assert "Bart_001_Character.jpg" in after
    # no stray temp files, no lowercase category remains
    assert not any(f.startswith("__af_") for f in after)
    assert not any("_character." in f for f in after)


def test_auto_format_moves_sidecars(tmp_path):
    (tmp_path / "ref_001_Character.png").write_bytes(b"x")  # gives the project a category
    (tmp_path / "Bart.png").write_bytes(b"x")
    (tmp_path / "Bart.txt").write_text("cap", encoding="utf-8")
    images = [p.name for p in tmp_path.iterdir() if p.suffix != ".txt"]
    N.auto_format(str(tmp_path), images)
    assert (tmp_path / "Bart_001_Character.png").is_file()
    assert (tmp_path / "Bart_001_Character.txt").read_text(encoding="utf-8") == "cap"


def test_auto_format_no_category_does_nothing(tmp_path):
    (tmp_path / "junk.png").write_bytes(b"x")
    applied = N.auto_format(str(tmp_path), ["junk.png"])
    assert applied == [] and (tmp_path / "junk.png").is_file()
