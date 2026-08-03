import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.trainer import training_env_overrides


def test_windows_disables_libuv_so_accelerate_can_rendezvous():
    """torch>=2.4 asks TCPStore for the libuv backend by default. Windows wheels built
    without libuv abort the accelerate rendezvous ("use_libuv was requested but PyTorch
    was build without libuv support") and the run dies with exit code 1 before step 1."""
    env = training_env_overrides(existing={}, windows=True)
    assert env["USE_LIBUV"] == "0"


def test_an_explicit_user_value_is_never_clobbered():
    """A user whose build DOES have libuv can opt back in from their own environment."""
    env = training_env_overrides(existing={"USE_LIBUV": "1"}, windows=True)
    assert "USE_LIBUV" not in env


def test_non_windows_keeps_the_modern_libuv_store():
    env = training_env_overrides(existing={}, windows=False)
    assert "USE_LIBUV" not in env


def test_utf8_and_allocator_settings_are_still_applied():
    env = training_env_overrides(existing={}, windows=True)
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONWARNINGS"] == "ignore::UserWarning"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_defaults_read_the_real_environment_without_raising():
    env = training_env_overrides()
    assert env["PYTHONUTF8"] == "1"
