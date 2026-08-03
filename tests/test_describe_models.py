import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.describe_models import (
    CUSTOM_INDEX, DESCRIBE_MODELS, entry_for_repo, has_model_snapshot,
    resolve_cache_dir, resolve_model_id,
)

JOY = "fancyfeast/llama-joycaption-beta-one-hf-llava"


def _rev_dir(root: Path, repo_id: str) -> Path:
    return root / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "abc123"


def _make_snapshot(root: Path, repo_id: str) -> Path:
    """A complete Hugging Face hub snapshot: config plus real weights."""
    d = _rev_dir(root, repo_id)
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * 64)
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
    assert has_model_snapshot(tmp_path, JOY) is True


def test_legacy_has_snapshot_false_for_a_different_repo(tmp_path):
    _make_snapshot(tmp_path, "someone/other-model")
    assert has_model_snapshot(tmp_path, JOY) is False


def test_has_model_snapshot_false_when_snapshot_dir_is_empty(tmp_path):
    (tmp_path / f"models--{JOY.replace('/', '--')}" / "snapshots" / "abc").mkdir(parents=True)
    assert has_model_snapshot(tmp_path, JOY) is False


def test_has_model_snapshot_false_when_dir_missing(tmp_path):
    assert has_model_snapshot(tmp_path / "nope", JOY) is False


def test_has_model_snapshot_false_for_an_aborted_download(tmp_path):
    """The real failure mode observed on disk: refs/main points at a valid
    revision whose snapshot holds only a stray 195-byte processor_config.json.
    Reporting that as downloaded sends the run straight into a load failure."""
    d = _rev_dir(tmp_path, JOY)
    d.mkdir(parents=True)
    (d / "processor_config.json").write_text('{"x": 1}', encoding="utf-8")
    assert has_model_snapshot(tmp_path, JOY) is False


def test_has_model_snapshot_false_when_weights_are_zero_bytes(tmp_path):
    d = _rev_dir(tmp_path, JOY)
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"")
    assert has_model_snapshot(tmp_path, JOY) is False


def test_has_model_snapshot_false_with_weights_but_no_config(tmp_path):
    d = _rev_dir(tmp_path, JOY)
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"\x00" * 64)
    assert has_model_snapshot(tmp_path, JOY) is False


def test_has_model_snapshot_accepts_pytorch_bin_weights(tmp_path):
    d = _rev_dir(tmp_path, JOY)
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "pytorch_model.bin").write_bytes(b"\x00" * 64)
    assert has_model_snapshot(tmp_path, JOY) is True


def test_resolve_cache_dir_ignores_an_aborted_legacy_download(tmp_path):
    """An incomplete legacy folder must fall through to the HF default rather
    than pinning the run to a directory that cannot load."""
    d = _rev_dir(tmp_path, JOY)
    d.mkdir(parents=True)
    (d / "processor_config.json").write_text('{"x": 1}', encoding="utf-8")
    assert resolve_cache_dir("", tmp_path, JOY) is None


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
