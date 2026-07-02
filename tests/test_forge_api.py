import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import forge_api


def test_build_test_payload_has_trigger_and_lora():
    p = forge_api.build_test_payload("mychar_v1", "mychar", "standing in a field")
    assert p["prompt"].startswith("mychar, standing in a field")
    assert "<lora:mychar_v1:1.0>" in p["prompt"]
    assert p["width"] == 1024 and p["steps"] == 24


def test_build_test_payload_no_double_trigger():
    p = forge_api.build_test_payload("lr", "mychar", "mychar, closeup")
    assert p["prompt"].count("mychar") == 1 or p["prompt"].startswith("mychar, closeup")
    assert "<lora:lr:1.0>" in p["prompt"]


def test_deliver_lora_copies_file(tmp_path):
    src = tmp_path / "out" / "mylora.safetensors"
    src.parent.mkdir()
    src.write_bytes(b"weights")
    dest_dir = tmp_path / "forge" / "Lora"
    out = forge_api.deliver_lora(str(src), str(dest_dir), api_url=None)
    assert Path(out).is_file()
    assert Path(out).read_bytes() == b"weights"
    assert Path(out).name == "mylora.safetensors"


def test_deliver_lora_renames_with_dest_name(tmp_path):
    src = tmp_path / "out" / "mylora.safetensors"
    src.parent.mkdir()
    src.write_bytes(b"weights")
    dest_dir = tmp_path / "comfy" / "loras"
    out = forge_api.deliver_lora(str(src), str(dest_dir), api_url=None,
                                 dest_name="mylora_mychar.safetensors")
    assert Path(out).name == "mylora_mychar.safetensors"
    assert Path(out).read_bytes() == b"weights"


def test_ping_false_on_bad_url():
    # Nothing listening on this port → graceful False, not an exception.
    assert forge_api.ping("http://127.0.0.1:9", timeout=1.0) is False
