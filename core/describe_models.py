"""Describe-model registry and cache resolution.

Pure module — Qt and settings are imported lazily inside `read_describe_defaults`,
so the rules below are unit-testable without a display.

Why this exists: the old code passed an explicit `cache_dir` to every
`from_pretrained` call, which overrides HF_HOME / HF_HUB_CACHE / the default
~/.cache/huggingface/hub. A user who already held the model in the standard hub
cache (including via a directory junction) was invisible to AnimaForge, which
then tried to re-download ~17GB. `resolve_cache_dir` returning None is the fix:
no cache_dir argument at all means Hugging Face looks where it normally does.
"""
from pathlib import Path

# Sentinel index for the "Custom (Hugging Face repo ID)" combo entry.
CUSTOM_INDEX = -1

# (label, repo_id, min_vram_gb, quantizable)
# min_vram_gb is the floor for the model's PREFERRED load mode (4-bit where
# quantizable, bf16 otherwise). It drives a warning, never a hard block.
DESCRIBE_MODELS = [
    ("JoyCaption Beta One 8B — best quality (8GB+ VRAM)",
     "fancyfeast/llama-joycaption-beta-one-hf-llava", 8, True),
    ("Qwen2.5-VL 3B — faster, lighter (6GB+ VRAM)",
     "Qwen/Qwen2.5-VL-3B-Instruct", 6, True),
    ("SmolVLM 2B — smallest, quickest (4GB+ VRAM)",
     "HuggingFaceTB/SmolVLM-Instruct", 4, True),
]

DEFAULT_MODEL_ID = DESCRIBE_MODELS[0][1]


def valid_repo_id(repo_id: str) -> bool:
    """A Hugging Face repo ID is 'org/name' with no whitespace."""
    rid = (repo_id or "").strip()
    return "/" in rid and not any(c.isspace() for c in rid)


def entry_for_repo(repo_id: str):
    """The DESCRIBE_MODELS row for this repo, or None when it's a custom ID."""
    for entry in DESCRIBE_MODELS:
        if entry[1] == repo_id:
            return entry
    return None


def resolve_model_id(index: int, custom_id: str) -> str:
    """Repo ID for a saved (index, custom) pair.

    Falls back to the first curated entry when the index is out of range or the
    custom ID is malformed — a bad saved value must never wedge captioning.
    """
    if index == CUSTOM_INDEX:
        return custom_id.strip() if valid_repo_id(custom_id) else DEFAULT_MODEL_ID
    if 0 <= index < len(DESCRIBE_MODELS):
        return DESCRIBE_MODELS[index][1]
    return DEFAULT_MODEL_ID


def legacy_has_snapshot(cache_dir, repo_id: str) -> bool:
    """True when `cache_dir` already holds a non-empty snapshot for `repo_id`.

    Repo-specific on purpose: a folder holding a DIFFERENT model must not be
    reused, or switching models would silently load the wrong weights.
    """
    snapshots = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return False
    try:
        return any(any(rev.iterdir()) for rev in snapshots.iterdir() if rev.is_dir())
    except OSError:
        return False


def resolve_cache_dir(explicit: str, legacy_dir, repo_id: str):
    """Which cache_dir to pass to from_pretrained, or None to let HF decide.

    1. An explicit Setup override wins.
    2. An existing AnimaForge/joycaption_model holding THIS model is reused, so
       upgrading users don't re-download what they already have.
    3. Otherwise None — HF_HOME / HF_HUB_CACHE / ~/.cache/huggingface/hub apply.
    """
    explicit = (explicit or "").strip()
    if explicit:
        return explicit
    if legacy_has_snapshot(legacy_dir, repo_id):
        return str(legacy_dir)
    return None


def legacy_cache_dir() -> Path:
    """The pre-2.9 app-local model folder."""
    from core.paths import app_root
    return app_root() / "joycaption_model"


def hf_default_cache_dir() -> Path:
    """Where Hugging Face itself would cache, honouring HF_HUB_CACHE / HF_HOME."""
    import os
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return Path(hub)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_is_present(repo_id: str, explicit: str = "") -> bool:
    """True when this model is already downloaded wherever it would be loaded from."""
    explicit = (explicit or "").strip()
    if explicit:
        return legacy_has_snapshot(Path(explicit), repo_id)
    return (legacy_has_snapshot(legacy_cache_dir(), repo_id)
            or legacy_has_snapshot(hf_default_cache_dir(), repo_id))


def read_describe_defaults(settings=None):
    """(repo_id, cache_dir_or_None, label) from saved settings."""
    from core.settings import AppSettings
    s = settings or AppSettings()
    repo_id = resolve_model_id(
        s.get("describe_model_index"), s.get("describe_model_custom_id"))
    cache_dir = resolve_cache_dir(
        s.get("describe_cache_dir"), legacy_cache_dir(), repo_id)
    entry = entry_for_repo(repo_id)
    label = entry[0] if entry else repo_id
    return repo_id, cache_dir, label
