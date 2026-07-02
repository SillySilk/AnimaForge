import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import gpu_check


def test_parse_free_mb_reads_first_gpu():
    assert gpu_check.parse_free_mb("12467\n8000\n") == 12467


def test_parse_free_mb_handles_units_and_blanks():
    assert gpu_check.parse_free_mb(" 9000 MiB \n") == 9000
    assert gpu_check.parse_free_mb("") is None
    assert gpu_check.parse_free_mb("garbage") is None


def test_parse_gpu_name_first_nonempty_line():
    assert gpu_check.parse_gpu_name("NVIDIA GeForce RTX 5090\n") == "NVIDIA GeForce RTX 5090"
    assert gpu_check.parse_gpu_name("\n  NVIDIA GeForce RTX 3060 \nsecond") == "NVIDIA GeForce RTX 3060"
    assert gpu_check.parse_gpu_name("") is None


def test_is_rtx_50_series_matches_blackwell_consumer_cards():
    for name in ("NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 5080",
                 "NVIDIA GeForce RTX 5070 Ti", "NVIDIA GeForce RTX 5060",
                 "NVIDIA GeForce RTX 5050", "GeForce RTX 5090 D",
                 "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"):
        assert gpu_check.is_rtx_50_series(name), name


def test_is_rtx_50_series_rejects_older_and_pro_cards():
    for name in ("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3060",
                 "NVIDIA GeForce GTX 1080 Ti", "Quadro RTX 5000",
                 "NVIDIA RTX 5000 Ada Generation", "NVIDIA RTX 500 Ada Generation",
                 "", None):
        assert not gpu_check.is_rtx_50_series(name), name
