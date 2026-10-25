"""MOSS-Image runs through the batch queue: model, VRAM tier, Low-VRAM isolation,
and refusal of missing files before anything launches."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication

import core.batch as batch_mod
from core import gpu_check, lowvram, moss_memory
from core.batch import FAILED, BatchRunner, RunDefinition

app = QApplication.instance() or QApplication([])


def _files(tmp_path):
    out = []
    for n in ("dit", "te", "vae"):
        p = tmp_path / f"{n}.safetensors"
        p.write_bytes(b"x")
        out.append(str(p))
    return out


def _run(tmp_path, name, model, **kw):
    folder = tmp_path / f"ds_{name}"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"x")
    (folder / "a.txt").write_text("a cat", encoding="utf-8")
    dit, te, vae = _files(tmp_path)
    base = dict(lora_name=name, dataset_folder=str(folder), sdscripts_path="C:/sd",
                enable_bucket=False, caption_policy="keep", model=model,
                dit_path=dit, qwen3_path=te, vae_path=vae,
                output_dir=str(tmp_path / "out"), image_count=4, target_steps=100)
    base.update(kw)
    return RunDefinition(**base)


def _runner(monkeypatch, vram):
    monkeypatch.setattr(gpu_check, "total_vram_mb", lambda: vram)
    monkeypatch.setattr(moss_memory, "has_triton", lambda: True)
    started, cfgs = [], []
    real = batch_mod.generate_configs

    def spy(**kw):
        out = real(**kw)
        cfgs.append(Path(out[0]).read_text(encoding="utf-8"))
        return out
    monkeypatch.setattr(batch_mod, "generate_configs", spy)
    r = BatchRunner()
    monkeypatch.setattr(r._trainer, "start",
                        lambda cfg, sd, model="anima": started.append(model))
    return r, started, cfgs


def test_batch_moss_passes_model_and_tier_flags(tmp_path, monkeypatch):
    r, started, cfgs = _runner(monkeypatch, 16376)
    r.start([_run(tmp_path, "m1", "moss")])
    assert started == ["moss"]
    assert "convrot_int8 = true" in cfgs[0]


def test_batch_moss_blocks_small_gpu(tmp_path, monkeypatch):
    r, started, cfgs = _runner(monkeypatch, 8188)
    run = _run(tmp_path, "m2", "moss")
    r.start([run])
    assert run.status == FAILED
    assert started == [] and cfgs == []


def test_batch_moss_block_message_says_12_gb(tmp_path, monkeypatch):
    r, _, _ = _runner(monkeypatch, 8188)
    logs = []
    r.log_line.connect(logs.append)
    r.start([_run(tmp_path, "m2b", "moss")])
    assert any("at least 12 GB" in line for line in logs)


def test_batch_moss_ignores_lowvram_but_anima_keeps_it(tmp_path, monkeypatch):
    lowvram.set_current(lowvram.recipe_for(12))
    try:
        r, started, cfgs = _runner(monkeypatch, 16376)
        r.start([_run(tmp_path, "m3", "moss")])
        assert started == ["moss"]
        assert "gradient_accumulation_steps" not in cfgs[0]
        assert "fp8_base" not in cfgs[0]
        r2, started2, cfgs2 = _runner(monkeypatch, 16376)
        r2.start([_run(tmp_path, "a3", "anima")])
        assert started2 == ["anima"]
        assert "gradient_accumulation_steps" in cfgs2[0]
    finally:
        lowvram.clear()


def test_batch_moss_missing_file_fails_named(tmp_path, monkeypatch):
    r, started, cfgs = _runner(monkeypatch, 16376)
    run = _run(tmp_path, "m4", "moss")
    gone = Path(run.dit_path)
    gone.unlink()
    logs = []
    r.log_line.connect(logs.append)
    r.start([run])
    assert run.status == FAILED
    assert started == [] and cfgs == []
    assert any(str(gone) in line for line in logs)


def test_batch_moss_unknown_gpu_warns_and_runs(tmp_path, monkeypatch):
    r, started, _ = _runner(monkeypatch, None)
    logs = []
    r.log_line.connect(logs.append)
    r.start([_run(tmp_path, "m5", "moss")])
    assert started == ["moss"]
    assert any("12 GB tier" in line for line in logs)


def test_batch_does_not_auto_deliver_a_moss_lora_to_forge(tmp_path, monkeypatch):
    import core.batch as batch_mod
    import core.forge_api as forge_api
    monkeypatch.setattr(batch_mod.AppSettings, "get", lambda self, k: {
        "forge_auto_deliver": True, "forge_lora_dir": str(tmp_path / "forge"),
        "forge_api_url": "http://127.0.0.1:1"}.get(k))
    delivered = []
    monkeypatch.setattr(forge_api, "deliver_lora",
                        lambda *a, **k: delivered.append(a) or "ok")
    shipped = tmp_path / "M.safetensors"
    shipped.write_bytes(b"lora")
    r = batch_mod.BatchRunner()
    logs = []
    r.log_line.connect(logs.append)
    r._maybe_deliver(batch_mod.RunDefinition("M", "C:/ds", 9, model="moss"), str(shipped))
    assert delivered == []
    assert any("not sent to Forge automatically" in line for line in logs)
    r._maybe_deliver(batch_mod.RunDefinition("K", "C:/ds", 9, model="krea2"), str(shipped))
    assert len(delivered) == 1
