"""Thin client for Forge / AUTOMATIC1111 's REST API (deliver a LoRA, test-render it).

Stdlib only (urllib + shutil). Forge must be started with `--api`. The QThread worker that
drives test-render lives in core/forge_worker.py; these are the pure pieces.
"""
import base64
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_NEGATIVE = "lowres, worst quality, low quality, jpeg artifacts, watermark, signature"


def _get(api_url: str, path: str, timeout: float = 5.0):
    req = urllib.request.Request(api_url.rstrip("/") + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(api_url: str, path: str, body: dict, timeout: float = 300.0):
    req = urllib.request.Request(
        api_url.rstrip("/") + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ping(api_url: str, timeout: float = 5.0) -> bool:
    """True if Forge's API is reachable (and --api is enabled)."""
    try:
        data = _get(api_url, "/sdapi/v1/sd-models", timeout=timeout)
        return isinstance(data, list)
    except (urllib.error.URLError, ValueError, OSError):
        return False


def deliver_lora(src_safetensors: str, dest_dir: str, api_url: str = None,
                 dest_name: str = None) -> str:
    """Copy the trained LoRA into Forge's models/Lora dir; refresh Forge if api_url given.

    `dest_name` renames the copy (used to suffix the trigger word so end users can
    read it off the filename). Returns the destination path. Raises on copy failure.
    """
    src = Path(src_safetensors)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / (dest_name or src.name)
    shutil.copy2(src, out)
    if api_url:
        try:
            _post(api_url, "/sdapi/v1/refresh-loras", {}, timeout=30.0)
        except (urllib.error.URLError, ValueError, OSError):
            pass  # file is delivered; refresh is best-effort
    return str(out)


def build_test_payload(lora_name: str, trigger: str, prompt: str, weight: float = 1.0,
                       steps: int = 24, cfg_scale: float = 5.0, width: int = 1024,
                       height: int = 1024, sampler_name: str = "Euler a",
                       negative_prompt: str = DEFAULT_NEGATIVE) -> dict:
    """A txt2img payload that leads with the trigger and references the LoRA."""
    trig = (trigger or "").strip()
    base = prompt.strip()
    if trig and not base.lower().startswith(trig.lower()):
        base = f"{trig}, {base}" if base else trig
    full = f"{base} <lora:{lora_name}:{weight}>".strip()
    return {
        "prompt": full,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "sampler_name": sampler_name,
        "batch_size": 1,
        "n_iter": 1,
    }


def txt2img(api_url: str, payload: dict, timeout: float = 300.0):
    """Run a txt2img call; return a list of decoded PNG bytes."""
    data = _post(api_url, "/sdapi/v1/txt2img", payload, timeout=timeout)
    return [base64.b64decode(b.split(",", 1)[-1]) for b in data.get("images", [])]
