"""The describe script's cast loading, which must fail before the model loads."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("joycaption_run_under_test",
                                                  ROOT / "scripts" / "joycaption_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cast_file(folder, data):
    from core import characters
    Path(characters.path_for(str(folder))).write_text(json.dumps(data), encoding="utf-8")


def test_a_good_cast_loads(tmp_path):
    _cast_file(tmp_path, {"roster": [{"token": "Mira"}],
                          "assignments": {"a.png": {"present": ["Mira"]}}})
    cast, problem = _script().load_cast(tmp_path)
    assert problem == "" and [c.token for c in cast.roster] == ["Mira"]


def test_no_cast_file_is_an_empty_cast(tmp_path):
    cast, problem = _script().load_cast(tmp_path)
    assert problem == "" and cast.roster == []


def test_a_cast_file_of_the_wrong_shape_is_named(tmp_path):
    _cast_file(tmp_path, [1, 2])
    cast, problem = _script().load_cast(tmp_path)
    assert cast is None and "cast file" in problem


def test_a_bad_assignment_is_caught_before_any_image(tmp_path):
    _cast_file(tmp_path, {"roster": [], "assignments": {"a.png": ["x"]}})
    cast, problem = _script().load_cast(tmp_path)
    assert cast is None and "cast file" in problem
