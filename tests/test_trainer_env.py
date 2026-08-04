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


# --- launch-time diagnostic -------------------------------------------------
# Training has no on-disk log, so when a run dies at exit code 1 the only evidence
# is the Train tab's log pane. It must say which USE_LIBUV the subprocess actually
# got, and where that value came from — "we set it" and "your environment set it,
# so we left it alone" are the two cases that look identical from the outside.

from core.trainer import libuv_status


def test_status_says_animaforge_set_it():
    s = libuv_status(existing={}, windows=True)
    assert "USE_LIBUV=0" in s and "AnimaForge" in s


def test_status_calls_out_an_inherited_value_that_defeats_the_fix():
    s = libuv_status(existing={"USE_LIBUV": "1"}, windows=True)
    assert "USE_LIBUV=1" in s
    assert "your environment" in s


def test_status_reports_an_inherited_zero_as_inherited():
    s = libuv_status(existing={"USE_LIBUV": "0"}, windows=True)
    assert "USE_LIBUV=0" in s and "your environment" in s


def test_status_on_non_windows_reports_the_torch_default():
    s = libuv_status(existing={}, windows=False)
    assert "USE_LIBUV=1" in s


from core.trainer import failure_hint


def test_the_libuv_error_gets_a_plain_english_hint():
    line = ("RuntimeError: use_libuv was requested but PyTorch was build without "
            "libuv support")
    hint = failure_hint(line)
    assert hint and "USE_LIBUV" in hint
    assert "Rebuilding PyTorch is not necessary" in hint


def test_ordinary_lines_get_no_hint():
    assert failure_hint("steps:  33%|###  | 100/3000 [01:32<24:17]") is None
    assert failure_hint("") is None


# --- Setup "force" override --------------------------------------------
# The Setup tab's "Ignore my environment" checkbox (force_use_libuv_off setting)
# lets a user override an inherited USE_LIBUV without editing Windows by hand.

def test_force_overrides_an_inherited_value():
    env = training_env_overrides(existing={"USE_LIBUV": "1"}, windows=True, force=True)
    assert env["USE_LIBUV"] == "0"


def test_force_is_a_no_op_off_windows():
    env = training_env_overrides(existing={"USE_LIBUV": "1"}, windows=False, force=True)
    assert "USE_LIBUV" not in env


def test_force_defaults_to_off_and_changes_nothing():
    env = training_env_overrides(existing={"USE_LIBUV": "1"}, windows=True)
    assert "USE_LIBUV" not in env  # same as before this change


def test_status_reports_a_forced_override():
    s = libuv_status(existing={"USE_LIBUV": "1"}, windows=True, force=True)
    assert "USE_LIBUV=0" in s and "forced" in s


def test_status_with_force_but_nothing_to_override_reads_normally():
    s = libuv_status(existing={}, windows=True, force=True)
    assert "USE_LIBUV=0" in s and "AnimaForge" in s


def test_status_on_non_windows_still_reports_an_inherited_value():
    s = libuv_status(existing={"USE_LIBUV": "1"}, windows=False)
    assert "USE_LIBUV=1" in s and "your environment" in s


# --- Windows registry helpers -------------------------------------------
# The persisted (registry) USE_LIBUV is distinct from os.environ: a running
# AnimaForge process only sees the value it inherited at launch. `reg` is
# injectable so these tests never touch the real Windows registry.

from core.trainer import (
    clear_windows_user_env_libuv,
    windows_system_env_libuv,
    windows_user_env_libuv,
)


class _FakeKey:
    def __init__(self, store):
        self._store = store

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeWinReg:
    """Stands in for the winreg module's shape: HKEY_* constants, OpenKey/QueryValueEx/
    DeleteValue. Two independent stores model the User vs. System registry hives."""
    HKEY_CURRENT_USER = "HKCU"
    HKEY_LOCAL_MACHINE = "HKLM"
    KEY_SET_VALUE = 0x2

    def __init__(self, user=None, system=None):
        self._user = dict(user or {})
        self._system = dict(system or {})

    def OpenKey(self, hive, subkey, *args, **kwargs):
        store = self._user if hive == self.HKEY_CURRENT_USER else self._system
        return _FakeKey(store)

    def QueryValueEx(self, key, name):
        if name not in key._store:
            raise FileNotFoundError(name)
        return key._store[name], 1

    def DeleteValue(self, key, name):
        if name not in key._store:
            raise FileNotFoundError(name)
        del key._store[name]


def test_reads_a_user_scope_value():
    reg = _FakeWinReg(user={"USE_LIBUV": "1"})
    assert windows_user_env_libuv(reg=reg) == "1"


def test_returns_none_when_nothing_is_set():
    assert windows_user_env_libuv(reg=_FakeWinReg()) is None
    assert windows_system_env_libuv(reg=_FakeWinReg()) is None


def test_reads_a_system_scope_value_separately_from_user_scope():
    reg = _FakeWinReg(system={"USE_LIBUV": "1"})
    assert windows_system_env_libuv(reg=reg) == "1"
    assert windows_user_env_libuv(reg=reg) is None


def test_clear_deletes_the_user_scope_value_only():
    reg = _FakeWinReg(user={"USE_LIBUV": "1"}, system={"USE_LIBUV": "1"})
    clear_windows_user_env_libuv(reg=reg, broadcast=False)
    assert windows_user_env_libuv(reg=reg) is None
    assert windows_system_env_libuv(reg=reg) == "1"   # system scope is never touched


def test_clear_is_safe_to_call_when_nothing_is_set():
    reg = _FakeWinReg()
    clear_windows_user_env_libuv(reg=reg, broadcast=False)   # must not raise
    assert windows_user_env_libuv(reg=reg) is None
