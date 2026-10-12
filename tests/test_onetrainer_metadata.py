import json
import struct
from pathlib import Path

from core.engines.onetrainer import metadata, verify


# ---------------------------------------------------------------------------
# A synthetic safetensors, built to the spec by hand. The app venv has no
# safetensors package (requirements.txt is PySide6/Pillow/toml) and neither does
# the code under test, so the format is written out here the same way it is read.
# ---------------------------------------------------------------------------

def _write_safetensors(path, tensors, meta=None):
    """`tensors` is {name: bytes}; every one is declared F32 and 1-D."""
    header, offset = {}, 0
    for name, blob in tensors.items():
        header[name] = {"dtype": "F32", "shape": [len(blob) // 4],
                        "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
    if meta is not None:
        header["__metadata__"] = meta
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for blob in tensors.values():
            f.write(blob)
    return path


def _tensor_bytes(path):
    """Everything after the header — the part a rewrite must not touch."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        f.seek(8 + n)
        return f.read()


def _lora_file(tmp_path, name="Aria.safetensors", meta=None):
    weights = struct.pack("<4f", 0.04, -0.03, 0.01, 0.02)
    return _write_safetensors(
        tmp_path / name,
        {"diffusion_model.blocks.0.attn.wq.lora_A.weight": weights,
         "diffusion_model.blocks.0.attn.wq.lora_B.weight": weights},
        meta)


def _captions(folder, *texts):
    for i, text in enumerate(texts):
        (folder / f"img{i:03d}.txt").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Reading the words out of the captions
# ---------------------------------------------------------------------------

def test_a_word_is_counted_once_per_caption_not_once_per_mention(tmp_path):
    _captions(tmp_path, "aria, aria, aria sits down", "aria, a kitchen")
    assert metadata.caption_tags(tmp_path)["aria"] == 2


def test_casing_does_not_split_a_word_in_two(tmp_path):
    """A trigger written 'Aria' in one caption and 'aria' in the next is one word —
    and it is reported in the casing it was first written in, because that is the
    casing the user will type into a prompt box."""
    _captions(tmp_path, "Aria, a kitchen", "aria, a garden")
    tags = metadata.caption_tags(tmp_path)
    assert list(tags.items())[0] == ("Aria", 2)


def test_the_trigger_leads_the_trained_words(tmp_path):
    _captions(tmp_path, "masterpiece, aria, a kitchen", "masterpiece, aria, a garden")
    words = metadata.trained_words(str(tmp_path), "aria", "masterpiece")
    assert words[0] == "aria"
    assert "masterpiece" in words


def test_the_trigger_leads_even_when_no_caption_carries_it(tmp_path):
    """The list is what to prompt with. A set captioned before the trigger was set
    is a problem the run already warned about (engine.advisories); dropping the
    trigger from the trained words would hide it instead."""
    _captions(tmp_path, "a kitchen", "a garden")
    assert metadata.trained_words(str(tmp_path), "aria")[0] == "aria"


def test_a_sentence_about_one_image_is_not_a_trained_word(tmp_path):
    """Captions here are prose, so most comma fragments are description. What makes
    a trained word is recurring across the set, not appearing in it."""
    _captions(tmp_path,
              "aria, standing beside a rusted blue tractor in the rain",
              "aria, sitting", "aria, walking", "aria, reading")
    words = metadata.trained_words(str(tmp_path), "aria")
    assert "aria" in words
    assert not any("tractor" in w for w in words)


def test_prose_never_reaches_the_frequency_table(tmp_path):
    """The whole table goes into the file header. Prose captions verbatim would
    make it enormous and tell the readers that parse it nothing."""
    long_prose = "a woman standing beside a rusted blue tractor in heavy rain"
    _captions(tmp_path, f"aria, {long_prose}", f"aria, {long_prose}")
    table = metadata.tag_frequency(str(tmp_path))[tmp_path.name]
    assert "aria" in table
    assert long_prose not in table


def test_the_frequency_table_is_keyed_by_the_dataset_folder(tmp_path):
    """kohya's shape: {concept name: {tag: count}} — readers walk the outer key."""
    folder = tmp_path / "bev_and_billy"
    folder.mkdir()
    _captions(folder, "bb, a kitchen", "bb, a garden")
    assert list(metadata.tag_frequency(str(folder))) == ["bev_and_billy"]


# ---------------------------------------------------------------------------
# Writing it into the file
# ---------------------------------------------------------------------------

def test_metadata_is_written_and_reads_back(tmp_path):
    path = _lora_file(tmp_path)
    metadata.write_metadata(str(path), {"ss_output_name": "Aria",
                                        "modelspec.trigger_phrase": "aria"})
    back = metadata.read_metadata(str(path))
    assert back["ss_output_name"] == "Aria"
    assert back["modelspec.trigger_phrase"] == "aria"


def test_not_one_weight_moves_when_the_header_grows(tmp_path):
    """data_offsets are relative to the end of the header, so a bigger header is
    free — but only if the buffer is copied through untouched. This is an hour of
    GPU time; a rewrite that shifted it by a byte would destroy the LoRA silently."""
    path = _lora_file(tmp_path)
    before = _tensor_bytes(path)
    metadata.write_metadata(str(path), {"ss_tag_frequency": "x" * 4000})
    assert _tensor_bytes(path) == before


def test_the_file_still_verifies_as_a_loadable_lora_afterwards(tmp_path):
    """verify.py parses the header independently of this module. If a rewrite broke
    the offsets or the namespace, it is the check that would notice."""
    path = _lora_file(tmp_path)
    assert verify.verify_output(str(path)) is None
    metadata.write_metadata(str(path), {"ss_output_name": "Aria"})
    assert verify.verify_output(str(path)) is None


def test_the_header_stays_eight_byte_aligned(tmp_path):
    """safetensors pads its header with spaces to a multiple of 8. A reader that
    assumes the alignment and finds the buffer one byte early reads garbage."""
    path = _lora_file(tmp_path)
    metadata.write_metadata(str(path), {"ss_output_name": "Aria"})
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
    assert n % 8 == 0


def test_existing_metadata_survives_the_write(tmp_path):
    path = _lora_file(tmp_path, meta={"kept": "yes", "ss_output_name": "old"})
    metadata.write_metadata(str(path), {"ss_output_name": "new"})
    back = metadata.read_metadata(str(path))
    assert back["kept"] == "yes" and back["ss_output_name"] == "new"


def test_every_value_written_is_a_string(tmp_path):
    """The safetensors contract is Dict[str, str]. A reader that hits an int gives
    up on the whole header, so an int here costs every key, not one."""
    path = _lora_file(tmp_path)
    metadata.write_metadata(str(path), {"ss_num_train_images": 47})
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    assert all(isinstance(v, str) for v in header["__metadata__"].values())


def test_a_file_that_is_not_a_safetensors_is_refused_not_mangled(tmp_path):
    path = tmp_path / "notalora.safetensors"
    path.write_bytes(b"not a safetensors file at all")
    try:
        metadata.write_metadata(str(path), {"ss_output_name": "Aria"})
    except metadata.MetadataError:
        pass
    else:
        raise AssertionError("a garbage file was accepted")
    assert path.read_bytes() == b"not a safetensors file at all"
    assert not list(tmp_path.glob("*.tmp")), "a temp file was left behind"


# ---------------------------------------------------------------------------
# Writing it beside the file
# ---------------------------------------------------------------------------

def test_the_sidecar_carries_a_line_that_can_be_pasted_as_is(tmp_path):
    _captions(tmp_path, "aria, masterpiece, a kitchen", "aria, masterpiece, a garden")
    words = metadata.trained_words(str(tmp_path), "aria", "masterpiece")
    text = metadata.sidecar_text("Aria", words, str(tmp_path), "aria")
    assert "aria, masterpiece" in text
    assert "2 of 2 captions" in text


def test_stamp_writes_both_copies(tmp_path):
    _captions(tmp_path, "aria, a kitchen", "aria, a garden")
    path = _lora_file(tmp_path)
    written = metadata.stamp(str(path), "Aria", str(tmp_path), "aria",
                             plan={"epochs": 20, "total_steps": 400},
                             image_count=2)
    assert Path(written["sidecar"]).is_file()
    assert written["sidecar"].endswith("Aria.trained-words.txt")
    back = metadata.read_metadata(str(path))
    assert back["ss_output_name"] == "Aria"
    assert back["ss_max_train_steps"] == "400"
    assert "aria" in back["ss_trained_words"]
    assert json.loads(back["ss_tag_frequency"])[tmp_path.name]["aria"] == 2


def test_the_app_labels_its_own_keys_as_animaforge(tmp_path):
    (tmp_path / "a.txt").write_text("toshi, a man")
    meta = metadata.build_metadata("Toshi", str(tmp_path), trigger_word="toshi")
    assert "animaforge_version" in meta and meta["animaforge_trigger"] == "toshi"
    assert not any(k.startswith("krea_forge_") for k in meta)


def test_metadata_reports_the_rank_the_run_trains_at(tmp_path):
    from core.engines.onetrainer.metadata import build_metadata
    from core.engines.onetrainer.preset import load_template
    t = load_template()
    meta = build_metadata("x", str(tmp_path))
    assert meta["ss_network_dim"] == str(t["lora_rank"]) == "32"
    assert meta["ss_network_alpha"] == "32.0"
