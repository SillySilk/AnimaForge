"""Create the unified `.venv` and install the full AnimaForge stack into it.

Run by a base Python (system 3.10/3.11 or the downloaded standalone) from install.bat:

    <base_python> scripts/bootstrap.py

Creates `.venv` at the repo root, then installs — in order so torch (cu121) wins over
the CPU torch that sd-scripts' `diffusers[torch]` would otherwise pull:
  1. pip upgrade
  2. torch>=2.5 + torchvision from the cu121 index
  3. sd-scripts/requirements.txt  (its trailing `-e .` installs the kohya lib editable)
  4. requirements.txt             (GUI: PySide6, Pillow, toml)
  5. onnx + onnxruntime-gpu       (WD14 tagger: onnx loads the model, ort runs it)
Then verifies the install. Windows-only. Pure helpers are unit-tested; see
docs/superpowers/specs/2026-06-24-venv-install-design.md.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
MIN_PY = (3, 10)
MAX_PY = (3, 11)  # inclusive: 3.10 and 3.11 are both supported by the torch/sd-scripts pins

# sd-scripts is fetched (not a submodule) at this exact upstream commit so that even
# "Download ZIP" users get the Anima-capable code. Bump to adopt a newer kohya.
SD_SCRIPTS_URL = "https://github.com/kohya-ss/sd-scripts.git"
SD_SCRIPTS_COMMIT = "1a3ec9ea745fe9883551dfca5c947ea3d6aa68c7"


# ---- pure helpers (unit-tested) -------------------------------------------

def parse_version(text: str):
    """Parse 'Python 3.10.14' → (3, 10, 14); return None if no match."""
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(g) for g in m.groups()) if m else None


def python_ok(version_str: str) -> bool:
    """True if a `python --version` string is a supported 3.10/3.11 interpreter."""
    v = parse_version(version_str)
    if not v:
        return False
    return MIN_PY <= (v[0], v[1]) <= MAX_PY


def pip_commands(venv_py: str, repo: str):
    """Ordered list of pip command argv lists to populate the venv. Pure."""
    repo_p = Path(repo)
    base = [venv_py, "-m", "pip", "install"]
    return [
        [venv_py, "-m", "pip", "install", "--upgrade", "pip"],
        base + ["torch>=2.5", "torchvision", "--index-url", TORCH_INDEX],
        base + ["-r", str(repo_p / "sd-scripts" / "requirements.txt")],
        base + ["-r", str(repo_p / "requirements.txt")],
        base + ["onnx", "onnxruntime-gpu"],
    ]


def step_cwd(cmd, repo):
    """Working directory a pip step must run in, or None for the default.

    The sd-scripts requirements file ends with `-e .` to install the kohya library
    editable. pip resolves `.` against the *current working directory*, not the
    requirements file's location — so that step MUST run from the sd-scripts dir
    (where setup.py lives), or `.` points at the repo root (no setup.py) and the
    install aborts before the GUI deps are installed. Pure."""
    repo_p = Path(repo)
    if str(repo_p / "sd-scripts" / "requirements.txt") in cmd:
        return str(repo_p / "sd-scripts")
    return None


def needs_sd_scripts(repo: str) -> bool:
    """True if sd-scripts hasn't been fetched yet (the Anima entrypoint is missing)."""
    return not (Path(repo) / "sd-scripts" / "anima_train_network.py").is_file()


def sd_scripts_clone_commands(repo: str, commit: str):
    """argv lists to clone kohya sd-scripts and pin it to `commit`. Pure."""
    dest = Path(repo) / "sd-scripts"
    return [
        ["git", "clone", SD_SCRIPTS_URL, str(dest)],
        ["git", "-C", str(dest), "checkout", commit],
    ]


SSL_MARKER = "CERTIFICATE_VERIFY_FAILED"


def is_ssl_cert_error(text: str) -> bool:
    """True if pip output shows a TLS cert verification failure (intercepting proxy)."""
    return SSL_MARKER in (text or "")


def with_truststore(cmd):
    """Add pip's `--use-feature=truststore` to a `pip install` command (validates
    against the OS trust store, where a corporate proxy's root CA lives). Pure."""
    if "install" in cmd and "--use-feature=truststore" not in cmd:
        i = cmd.index("install")
        return cmd[: i + 1] + ["--use-feature=truststore"] + cmd[i + 1:]
    return cmd


# ---- side-effecting steps -------------------------------------------------

def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe"


def _run(cmd, **kw):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, **kw)


def _run_tee(cmd, cwd=None):
    """Run cmd, streaming output live AND capturing it. Returns (returncode, text)."""
    print(f"\n>>> {' '.join(str(c) for c in cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, errors="replace", cwd=cwd)
    buf = []
    for line in proc.stdout:
        sys.stdout.write(line)
        buf.append(line)
    sys.stdout.flush()
    proc.wait()
    return proc.returncode, "".join(buf)


def create_venv(base_python: str, venv_dir: Path) -> Path:
    if not _venv_python(venv_dir).is_file():
        r = _run([base_python, "-m", "venv", str(venv_dir)])
        if r.returncode != 0:
            raise SystemExit(f"venv creation failed (exit {r.returncode}).")
    return _venv_python(venv_dir)


def ensure_sd_scripts():
    """Fetch sd-scripts at the pinned commit if absent. No-op once present (dev machines
    and re-runs). Submodule-free so GitHub 'Download ZIP' users get the code too."""
    if not needs_sd_scripts(str(REPO)):
        return
    for cmd in sd_scripts_clone_commands(str(REPO), SD_SCRIPTS_COMMIT):
        r = _run(cmd)
        if r.returncode != 0:
            raise SystemExit(
                f"\nFailed to fetch sd-scripts (exit {r.returncode}): {' '.join(cmd)}\n"
                "Check your internet connection and that `git` is installed, then re-run."
            )


def install_all(venv_py: Path):
    for cmd in pip_commands(str(venv_py), str(REPO)):
        cwd = step_cwd(cmd, str(REPO))
        rc, out = _run_tee(cmd, cwd=cwd)
        if rc == 0:
            continue
        # TLS-intercepting proxy? pip's certifi bundle rejects the proxy's root CA even
        # though the OS trusts it. Retry once validating against the OS trust store.
        if is_ssl_cert_error(out) and "install" in cmd:
            print("\n[bootstrap] TLS certificate verification failed — likely a corporate/"
                  "intercepting proxy. Retrying via the OS trust store (truststore)…")
            rc2, _ = _run_tee(with_truststore(cmd), cwd=cwd)
            if rc2 == 0:
                continue
            raise SystemExit(
                "\nStill failing after the OS-trust-store retry. If you are behind a TLS-"
                "intercepting proxy, point pip at your corporate root CA, e.g.:\n"
                "  .venv\\Scripts\\python -m pip config set global.cert <path-to-corp-ca.pem>\n"
                "then re-run install.bat."
            )
        raise SystemExit(
            f"\nInstall step failed (exit {rc}):\n  {' '.join(cmd)}\n"
            "Check your network connection and that you have an NVIDIA GPU + driver, "
            "then re-run install.bat."
        )


def verify(venv_py: Path):
    check = (
        "import PySide6, torch;"
        "print('PySide6', PySide6.__version__);"
        "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    )
    _run([str(venv_py), "-c", check])
    train_script = REPO / "sd-scripts" / "anima_train_network.py"
    if train_script.is_file():
        print(f"\nOK: found {train_script.relative_to(REPO)}")
    else:
        print("\nWARNING: sd-scripts/anima_train_network.py not found — the fetch step "
              "may have failed. Re-run install.bat (needs git + internet).")


def main():
    print("=" * 64)
    print("  AnimaForge — environment bootstrap (unified .venv)")
    print("=" * 64)
    venv_dir = REPO / ".venv"
    venv_py = create_venv(sys.executable, venv_dir)
    ensure_sd_scripts()
    install_all(venv_py)
    verify(venv_py)
    print("\n" + "=" * 64)
    print("  Done. Launch the app with:  launch.bat")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
