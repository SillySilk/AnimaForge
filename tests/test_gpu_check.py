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
