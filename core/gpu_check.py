"""Best-effort GPU free-VRAM probe (Windows, NVIDIA) for the pre-launch guard.

Never raises and never blocks: if nvidia-smi is missing or unparseable, callers
get None and skip the warning rather than failing a launch.
"""
import re
import subprocess

from utils.proc import no_window_creationflags


def parse_free_mb(stdout: str):
    for line in (stdout or "").splitlines():
        m = re.search(r"\d+", line)
        if m:
            return int(m.group())
    return None


def free_vram_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=no_window_creationflags(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return parse_free_mb(out.stdout)


def parse_gpu_name(stdout: str):
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line:
            return line
    return None


def gpu_name():
    """Marketing name of the first GPU (e.g. 'NVIDIA GeForce RTX 5090'), or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            creationflags=no_window_creationflags(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return parse_gpu_name(out.stdout)


def is_rtx_50_series(name) -> bool:
    """Blackwell cards need cu128 torch wheels (cu121 has no sm_120 kernels).

    Consumer models are RTX 5050/5060/5070/5080/5090 (+Ti/D variants). The third
    digit is restricted to 5-9 so 'RTX 5000 Ada' / 'Quadro RTX 5000' (older pro
    cards) do NOT match; pro Blackwell parts carry 'Blackwell' in the name.
    """
    n = (name or "").lower()
    return "blackwell" in n or bool(re.search(r"rtx\s*50[5-9]0\b", n))


def resident_gpu_apps():
    found = []
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=10,
                             creationflags=no_window_creationflags())
        text = (out.stdout or "").lower()
        if "llama-server" in text:
            found.append("LM Studio (llama-server)")
        if "launch.py" in text or "forge" in text:
            found.append("Forge Neo")
    except (OSError, subprocess.SubprocessError):
        pass
    return found
