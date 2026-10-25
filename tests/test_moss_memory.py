import subprocess
import types

import pytest

from core import gpu_check
from core import moss_memory as mm

F24 = {"convrot_int8_text_encoder": True}
F16 = {"convrot_int8": True, "convrot_int8_text_encoder": True}
F12 = {**F16, "blocks_to_swap": 28, "block_swap_h2d_only": True}
BLOCK = "MOSS-Image needs a GPU with at least 12 GB of VRAM."
TRITON = "INT8 will run slower without triton-windows — re-run install.bat."
UNKNOWN = "Couldn't read the GPU's memory; using the safest MOSS memory setting (12 GB tier)."


@pytest.mark.parametrize("mb,tier,flags", [
    (24564, "24", F24), (23000, "24", F24),
    (22999, "16", F16), (16376, "16", F16), (15000, "16", F16),
    (14999, "12", F12), (12282, "12", F12), (11000, "12", F12),
])
def test_tier_boundaries(mb, tier, flags):
    assert mm.tier_for(mb) == (tier, flags)


@pytest.mark.parametrize("mb", [10999, 8188])
def test_below_12gb_blocks(mb):
    with pytest.raises(mm.MossVramError) as e:
        mm.tier_for(mb)
    assert str(e.value) == BLOCK


def test_unknown_vram_uses_12_tier():
    assert mm.tier_for(None) == ("12", F12)


def test_tier_flags_are_copies():
    mm.tier_for(None)[1]["blocks_to_swap"] = 1
    assert mm.tier_for(None)[1]["blocks_to_swap"] == 28


def test_ram_warning_12_only():
    w = mm.preflight_warnings("12", 16, True, True)
    assert len(w) == 1 and "found 16 GB" in w[0]
    assert w[0] == ("12 GB MOSS training swaps model blocks into system RAM; "
                    "32 GB of RAM is recommended (found 16 GB).")
    assert mm.preflight_warnings("24", 16, True, True) == []
    assert mm.preflight_warnings("16", 16, True, True) == []
    assert mm.preflight_warnings("12", 32, True, True) == []
    assert mm.preflight_warnings("12", None, True, True) == []


@pytest.mark.parametrize("tier", ["24", "16", "12"])
def test_triton_warning_every_tier(tier):
    assert TRITON in mm.preflight_warnings(tier, 64, False, True)
    assert TRITON not in mm.preflight_warnings(tier, 64, True, True)


def test_unknown_gpu_warning():
    w = mm.preflight_warnings("12", 64, True, False)
    assert w == [UNKNOWN]


def test_has_triton_never_raises(monkeypatch):
    def boom(name):
        raise RuntimeError("broken")
    monkeypatch.setattr(mm.importlib, "import_module", boom)
    assert mm.has_triton() is False
    monkeypatch.setattr(mm.importlib, "import_module", lambda n: object())
    assert mm.has_triton() is True


def _run(monkeypatch, result):
    calls = []

    def fake(cmd, **kw):
        calls.append(cmd)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(gpu_check.subprocess, "run", fake)
    return calls


def test_total_vram_mb_parses(monkeypatch):
    calls = _run(monkeypatch, types.SimpleNamespace(returncode=0, stdout="24564\n"))
    assert gpu_check.total_vram_mb() == 24564
    assert "--query-gpu=memory.total" in calls[0]


def test_total_vram_mb_never_raises(monkeypatch):
    _run(monkeypatch, FileNotFoundError())
    assert gpu_check.total_vram_mb() is None
    _run(monkeypatch, subprocess.TimeoutExpired("x", 1))
    assert gpu_check.total_vram_mb() is None
    _run(monkeypatch, types.SimpleNamespace(returncode=1, stdout=""))
    assert gpu_check.total_vram_mb() is None
    _run(monkeypatch, types.SimpleNamespace(returncode=0, stdout="N/A\n"))
    assert gpu_check.total_vram_mb() is None
