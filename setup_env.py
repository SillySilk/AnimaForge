"""
PonyExpress — Environment Setup Script

Run this script to install the Python dependencies required to run
PonyExpress (the GUI application itself, NOT the Kohya sd-scripts
training backend).

Usage:
    python setup_env.py
"""

import subprocess
import sys
import os
from pathlib import Path


REQUIREMENTS = [
    "PySide6>=6.6.0",
    "Pillow>=10.0.0",
    "toml>=0.10.2",
]

REQUIREMENTS_FILE = Path(__file__).parent / "requirements.txt"


def run(cmd: list, description: str) -> bool:
    print(f"\n>>> {description}")
    print(f"    Command: {' '.join(cmd)}\n")
    try:
        result = subprocess.run(cmd, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ERROR: Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"    ERROR: Executable not found: {cmd[0]}")
        return False


def check_python_version():
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print(
            f"ERROR: Python 3.10 or higher is required. "
            f"You are running Python {major}.{minor}."
        )
        sys.exit(1)
    print(f"Python version: {major}.{minor} — OK")


def main():
    print("=" * 60)
    print("  PonyExpress — GUI Dependency Installer")
    print("=" * 60)

    check_python_version()

    print(f"\nInstalling from: {REQUIREMENTS_FILE}")

    # Upgrade pip first
    run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        "Upgrading pip",
    )

    # Install from requirements.txt
    success = run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        "Installing PonyExpress GUI requirements",
    )

    if success:
        print("\n" + "=" * 60)
        print("  Installation complete!")
        print("  Run PonyExpress with:  python main.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("  Installation encountered errors.")
        print("  Try installing manually:")
        for req in REQUIREMENTS:
            print(f"    pip install \"{req}\"")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
