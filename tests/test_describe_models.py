import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.describe_models import (
    CUSTOM_INDEX, DESCRIBE_MODELS, entry_for_repo, legacy_has_snapshot,
    resolve_cache_dir, resolve_model_id,
)

JOY = "fancyfeast/llama-joycaption-beta-one-hf-llava"


def _make_snapshot(root: Path, repo_id: str, filename: str = "config.json") -> Path:
    """Build the Hugging Face hub cache layout for a repo under `root`."""
    d = root / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "abc123"
    d.mkdir(parents=True)
    (d / filename).write_text("{}", encoding="utf-8")
    return d


def test_default_model_is_first_entry():
    assert DESCRIBE_MODELS[0][1] == JOY


def test_resolve_model_id_maps_index():
    assert resolve_model_id(0, "") == DESCRIBE_MODELS[0][1]


def test_resolve_model_id_uses_custom_at_custom_index():
    assert resolve_model_id(CUSTOM_INDEX, "some/repo") == "some/repo"


def test_resolve_model_id_falls_back_when_custom_is_invalid():
    assert resolve_model_id(CUSTOM_INDEX, "no-slash") == DESCRIBE_MODELS[0][1]
    assert resolve_model_id(CUSTOM_INDEX, "  ") == DESCRIBE_MODELS[0][1]
    assert resolve_model_id(CUSTOM_INDEX, "has space/repo") == DESCRIBE_MODELS[0][1]


def test_resolve_model_id_falls_back_on_out_of_range_index():
    assert resolve_model_id(99, "") == DESCRIBE_MODELS[0][1]


def test_legacy_has_snapshot_true_when_repo_present(tmp_path):
    _make_snapshot(tmp_path, JOY)
    assert legacy_has_snapshot(tmp_path, JOY) is True


def test_legacy_has_snapshot_false_for_a_different_repo(tmp_path):
    _make_snapshot(tmp_path, "someone/other-model")
    assert legacy_has_snapshot(tmp_path, JOY) is False


def test_legacy_has_snapshot_false_when_snapshot_dir_is_empty(tmp_path):
    (tmp_path / f"models--{JOY.replace('/', '--')}" / "snapshots" / "abc").mkdir(parents=True)
    assert legacy_has_snapshot(tmp_path, JOY) is False


def test_legacy_has_snapshot_false_when_dir_missing(tmp_path):
    assert legacy_has_snapshot(tmp_path / "nope", JOY) is False


def test_resolve_cache_dir_prefers_explicit(tmp_path):
    _make_snapshot(tmp_path, JOY)
    assert resolve_cache_dir("D:/models", tmp_path, JOY) == "D:/models"


def test_resolve_cache_dir_reuses_legacy_when_model_present(tmp_path):
    _make_snapshot(tmp_path, JOY)
    assert resolve_cache_dir("", tmp_path, JOY) == str(tmp_path)


def test_resolve_cache_dir_returns_none_when_legacy_empty(tmp_path):
    assert resolve_cache_dir("", tmp_path, JOY) is None


def test_resolve_cache_dir_returns_none_when_legacy_holds_other_model(tmp_path):
    _make_snapshot(tmp_path, "someone/other-model")
    assert resolve_cache_dir("", tmp_path, JOY) is None


def test_entry_for_repo_finds_curated_and_misses_custom():
    assert entry_for_repo(JOY)[1] == JOY
    assert entry_for_repo("nobody/unknown") is None
