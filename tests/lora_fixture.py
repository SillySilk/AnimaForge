"""Builds real .safetensors LoRA files for tests.

Small enough to construct inline, structurally faithful enough that
core.trainer.verify reads them exactly as it reads a 114 MB trained file:
8-byte little-endian header length, JSON header, then the tensor blob.
"""
import json
import struct

# The lora_B peak a working Krea 2 LoRA carries. Measured across the 10 known-good
# files in G:\LORAS\Characters (2026-08-17): 2.7e-02 to 3.0e-01.
HEALTHY_B = 4.0e-2

# What Bev_and_Billy actually shipped: lora_B still at its zero init.
DEAD_B = 1.7e-8


def bf16(value: float) -> bytes:
    """BF16 is the top 16 bits of the float32 pattern, so truncation is the cast."""
    return struct.pack("<f", value)[2:]


def write_lora(path, *, prefix="diffusion_model", b_value=HEALTHY_B,
               a_value=1.3e-2, blocks=2, with_b=True):
    """A structurally real safetensors LoRA. Defaults describe a healthy file."""
    rank, dim = 16, 8
    tensors, blob = {}, bytearray()

    def add(name, rows, cols, value):
        start = len(blob)
        blob.extend(bf16(value) * (rows * cols))
        tensors[name] = {"dtype": "BF16", "shape": [rows, cols],
                         "data_offsets": [start, len(blob)]}

    for i in range(blocks):
        stem = f"{prefix}.blocks.{i}.attn.wq"
        add(f"{stem}.lora_A.weight", rank, dim, a_value)
        if with_b:
            add(f"{stem}.lora_B.weight", dim, rank, b_value)

    header = json.dumps(tensors).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(blob)
    return path
