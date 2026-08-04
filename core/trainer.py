import os
import re
import sys
try:
    import winreg
except ImportError:  # pragma: no cover - winreg only exists on Windows
    winreg = None
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

from utils.proc import apply_no_window


def training_env_overrides(existing=None, windows=None, force=False) -> dict:
    """Environment the training subprocess needs on top of the inherited one.

    USE_LIBUV is the important one. Since torch 2.4 a c10d TCPStore asks for the libuv
    backend unless USE_LIBUV=0 (torch/distributed/rendezvous.py reads exactly that
    variable). Plenty of Windows PyTorch wheels are compiled without libuv, so building
    one raises "use_libuv was requested but PyTorch was build without libuv support" and
    the whole run exits with code 1 before a single training step — with nothing on
    screen but the exit code. The legacy store it falls back to is equivalent here: we
    launch --num_processes=1, so the store is a single-rank formality either way.

    This covers the code paths that honor the variable (init_process_group's rendezvous
    and the elastic dynamic rendezvous). It does NOT cover torch's *static* TCP
    rendezvous, which ignores USE_LIBUV entirely — see ACCELERATE_CONFIG_NAME below for
    the setting that keeps runs off that path.

    An explicit USE_LIBUV in the user's own environment always wins, so anyone with a
    libuv-enabled build can opt back in — unless `force=True` (the Setup → Training
    Environment "ignore my environment" toggle), which makes AnimaForge set USE_LIBUV=0
    unconditionally. That's the self-service alternative to deleting the variable from
    Windows by hand.
    """
    if windows is None:
        windows = sys.platform.startswith("win")
    if existing is None:
        existing = os.environ
    env = {
        # Force UTF-8 and suppress known harmless warnings from the subprocess
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONWARNINGS": "ignore::UserWarning",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    if windows and (force or not existing.get("USE_LIBUV")):
        env["USE_LIBUV"] = "0"
    return env


def libuv_status(existing=None, windows=None, force=False) -> str:
    """One line for the training log: the USE_LIBUV the subprocess really gets, and why.

    Worth logging because the two failure modes are indistinguishable from outside.
    We deliberately never clobber a USE_LIBUV already in the user's environment unless
    `force=True`, so a stray `USE_LIBUV=1` there silently defeats the fix and the run
    still dies at exit code 1 — looking exactly like the fix never shipped.
    """
    if windows is None:
        windows = sys.platform.startswith("win")
    if existing is None:
        existing = os.environ
    inherited = existing.get("USE_LIBUV")
    if inherited and windows and force:
        return (f"USE_LIBUV=0 (forced — your Setup override replaces your "
                 f"environment's USE_LIBUV={inherited})")
    if inherited:
        return (f"USE_LIBUV={inherited} (from your environment — AnimaForge left it "
                f"alone; unset it to let AnimaForge choose)")
    if windows:
        return "USE_LIBUV=0 (set by AnimaForge for this run)"
    return "USE_LIBUV=1 (torch default; not Windows)"


ACCELERATE_CONFIG_NAME = "accelerate_single_gpu.yaml"

# `accelerate launch` reads ~/.cache/huggingface/accelerate/default_config.yaml whenever
# that file exists, and a `distributed_type: MULTI_GPU` in it sets multi_gpu=True even
# though we pass --num_processes=1 (the "you need at least 2 processes" guard runs before
# the file is loaded, so nothing catches the contradiction). That routes the launch through
# multi_gpu_launcher -> torch elastic -> static_tcp_rendezvous, which constructs TCPStore()
# with no use_libuv kwarg and therefore ignores USE_LIBUV completely — on a Windows wheel
# built without libuv the run dies before step 1 no matter what we put in the environment.
#
# Anyone who has ever run `accelerate config` has that file, so we cannot leave the choice
# of launcher to chance. Passing our own config pins simple_launcher, which builds no
# TCPStore at all. AnimaForge is single-GPU by design (--num_processes=1), so this only
# makes explicit what the run was always asking for.
_ACCELERATE_SINGLE_GPU_CONFIG = """\
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
machine_rank: 0
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""


def write_accelerate_config(configs_dir) -> Path:
    """Write the single-process accelerate config for a run and return its path.

    Rewritten every launch rather than created once: it costs nothing, and it means a
    stale or hand-edited copy in an old run folder can never quietly reintroduce the
    multi-GPU launcher. Note `'NO'` must stay quoted — bare NO is YAML's boolean false.
    """
    configs_dir = Path(configs_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)
    path = configs_dir / ACCELERATE_CONFIG_NAME
    path.write_text(_ACCELERATE_SINGLE_GPU_CONFIG, encoding="utf-8")
    return path


def failure_hint(line: str):
    """Plain-English explanation for a known-cryptic launch failure, else None.

    torch's wording ("PyTorch was build without libuv support") is both misspelled and
    meaningless to a user whose only other evidence is exit code 1.

    The hint must not blame the user's USE_LIBUV: the error reaches them through torch's
    static TCP rendezvous, which never reads that variable. An earlier version of this
    text sent a user hunting through System Properties for a variable that was not set
    and would not have mattered.
    """
    low = (line or "").lower()
    if "libuv" in low and "without libuv" in low:
        return ("[AnimaForge] That means this PyTorch build has no libuv support, and "
                "the run reached torch's multi-process launcher, which needs it. "
                "AnimaForge pins accelerate to a single-process config to stay off that "
                "path, so seeing this means the pin did not take — please report it "
                "with this log. Rebuilding PyTorch is not necessary.")
    return None


_USER_ENV_SUBKEY = "Environment"
_SYSTEM_ENV_SUBKEY = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"


def _read_registry_env_value(hive_attr: str, subkey: str, reg=None):
    """USE_LIBUV from one registry hive, or None if unset/off Windows.

    `reg` defaults to the real `winreg` module; tests inject a fake so nothing here
    ever touches the actual Windows registry.
    """
    reg = reg or winreg
    if reg is None:
        return None
    hive = getattr(reg, hive_attr)
    try:
        with reg.OpenKey(hive, subkey) as key:
            value, _ = reg.QueryValueEx(key, "USE_LIBUV")
            return str(value)
    except (FileNotFoundError, OSError):
        return None


def windows_user_env_libuv(reg=None):
    """USE_LIBUV persisted in the Windows *User* environment, or None if unset.

    This is the registry value, not the current process's os.environ — a running
    AnimaForge inherited its environment at launch and won't see a later registry
    edit until it restarts.
    """
    return _read_registry_env_value("HKEY_CURRENT_USER", _USER_ENV_SUBKEY, reg)


def windows_system_env_libuv(reg=None):
    """USE_LIBUV persisted in the Windows *System* environment, or None if unset.

    Read-only: AnimaForge never clears this scope (it needs admin elevation and
    affects every app on the machine, not just training).
    """
    return _read_registry_env_value("HKEY_LOCAL_MACHINE", _SYSTEM_ENV_SUBKEY, reg)


def clear_windows_user_env_libuv(reg=None, broadcast=True) -> bool:
    """Delete USE_LIBUV from the Windows *User* environment. No-op if unset or off Windows.

    Only ever touches HKEY_CURRENT_USER — never System scope. The caller is
    responsible for confirming with the user first; this performs the deletion
    unconditionally. `broadcast=False` skips notifying other processes (tests only).
    Returns True if a value was actually deleted, False if there was nothing to
    delete or the delete failed (e.g. a permissions error) — the caller uses this
    to tell the user whether anything happened.
    """
    reg = reg or winreg
    if reg is None:
        return False
    try:
        with reg.OpenKey(reg.HKEY_CURRENT_USER, _USER_ENV_SUBKEY, 0, reg.KEY_SET_VALUE) as key:
            reg.DeleteValue(key, "USE_LIBUV")
    except (FileNotFoundError, OSError):
        return False
    if broadcast:
        _broadcast_environment_change()
    return True


def _broadcast_environment_change() -> None:
    """Tell other running processes the environment changed. Best-effort, never raises."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        result = ctypes.c_long()
        ctypes.windll.user32.SendMessageTimeoutW(
            0xFFFF, 0x1A, 0, "Environment", 0x0002, 5000, ctypes.byref(result))
    except Exception:
        pass


class TrainingProcess(QObject):
    progress_updated = Signal(int)   # current step (0-based int)
    log_line = Signal(str)           # raw log line string
    training_finished = Signal(bool) # True = success, False = error/killed
    training_started = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None
        self._total_steps = 3000
        self._current_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_total_steps(self, total_steps: int):
        self._total_steps = max(1, total_steps)

    def build_command(self, config_path: str, sdscripts_path: str) -> tuple:
        """
        Build the accelerate launch command for anima_train_network.py.

        Returns (program, args_list).
        Uses the unified .venv interpreter (which has accelerate + the ML stack);
        a legacy sd-scripts/venv is honored only for old two-venv installs.
        """
        from core.env import subprocess_python
        python = subprocess_python(sdscripts_path)

        train_script = str(Path(sdscripts_path) / "anima_train_network.py")
        # Ours, not the user's ~/.cache one. Must come before train_script or accelerate
        # would hand it to sd-scripts instead of reading it. Distinct from the
        # --config_file after train_script, which is the training config.
        accelerate_config = write_accelerate_config(Path(config_path).parent)
        args = [
            "-m",
            "accelerate.commands.launch",
            f"--config_file={accelerate_config}",
            "--num_cpu_threads_per_process=2",
            "--num_processes=1",
            "--num_machines=1",
            "--mixed_precision=bf16",
            "--dynamo_backend=no",
            train_script,
            f"--config_file={config_path}",
        ]
        return python, args

    def _env_overrides(self) -> dict:
        """This launch's subprocess env overrides, honoring the Setup 'force' toggle."""
        from core.settings import AppSettings
        return training_env_overrides(force=AppSettings().get("force_use_libuv_off"))

    def _libuv_log_status(self) -> str:
        from core.settings import AppSettings
        return libuv_status(force=AppSettings().get("force_use_libuv_off"))

    def start(self, config_path: str, sdscripts_path: str):
        """Start the training process."""
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            return

        program, args = self.build_command(config_path, sdscripts_path)

        self._process = QProcess(self)
        apply_no_window(self._process)  # no console window pop-up on Windows
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_ready_read)
        self._process.finished.connect(self._on_finished)

        from PySide6.QtCore import QProcessEnvironment
        env = QProcessEnvironment.systemEnvironment()
        for key, value in self._env_overrides().items():
            env.insert(key, value)
        self._process.setProcessEnvironment(env)

        self._current_step = 0
        self._process.start(program, args)
        self.training_started.emit()
        self.log_line.emit(f"[AnimaForge] Launching: {program} {' '.join(args)}")
        self.log_line.emit(f"[AnimaForge] {self._libuv_log_status()}")

    def stop(self):
        """Terminate the running training process."""
        if self._process is not None:
            if self._process.state() != QProcess.NotRunning:
                self._process.terminate()
                if not self._process.waitForFinished(5000):
                    self._process.kill()
            self._process = None

    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.state() != QProcess.NotRunning

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_ready_read(self):
        if self._process is None:
            return
        data = self._process.readAllStandardOutput()
        text = bytes(data).decode("utf-8", errors="replace")
        for line in text.splitlines():
            if line:
                if "impl_abstract" in line:
                    continue
                self.log_line.emit(line)
                hint = failure_hint(line)
                if hint:
                    self.log_line.emit(hint)
                step = self._parse_progress(line)
                if step is not None:
                    self._current_step = step
                    self.progress_updated.emit(step)

    def _on_finished(self, exit_code: int, exit_status):
        success = exit_code == 0
        # The exit-code line goes out BEFORE training_finished. The Train tab closes
        # its on-disk run log when that signal arrives, and on a failed run the exit
        # code is the single line a bug report most needs — emitting it afterwards
        # dropped it from the file.
        status = "completed successfully" if success else f"finished with exit code {exit_code}"
        self.log_line.emit(f"[AnimaForge] Training process {status}.")
        self.training_finished.emit(success)

    def _parse_progress(self, line: str):
        """
        Extract the current step count from Kohya output.

        Only the main training bar counts — sd-scripts names it ``desc="steps"``.
        Other tqdm bars in the same stream (latent caching, "Sampling" while a
        preview set renders, weight loading) would otherwise clobber the step,
        the total, and the dials mid-run.

        Patterns handled:
          steps:  33%|███       | 100/3000 [01:32<24:17,  1.10it/s, loss=0.0821]
          steps: 100/3000
        """
        if not line.lstrip().lower().startswith("steps"):
            return None

        # Pattern 1: tqdm bar  "current/total [..."
        m = re.search(r"(\d+)/(\d+)\s*\[", line)
        if m:
            current = int(m.group(1))
            total = int(m.group(2))
            if total > 0:
                self._total_steps = total
            return current

        # Pattern 2: plain "steps: current/total"
        m = re.search(r"steps:\s*(\d+)/(\d+)", line, re.IGNORECASE)
        if m:
            current = int(m.group(1))
            total = int(m.group(2))
            if total > 0:
                self._total_steps = total
            return current

        return None

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def current_step(self) -> int:
        return self._current_step
