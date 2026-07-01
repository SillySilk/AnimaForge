import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts"))

import llm_refine_run as L


def test_user_text_refine_has_all_blocks_and_instruction():
    t = L.build_user_text("1girl, solo", "a girl on a bench", "lighting", "character")
    assert "<tags>1girl, solo</tags>" in t
    assert "<draft>a girl on a bench</draft>" in t
    assert "<focus>lighting</focus>" in t
    assert "<lora_type>character</lora_type>" in t
    assert t.strip().endswith("Fuse and verify; output the two lines now.")


def test_user_text_fresh_omits_empty_blocks():
    t = L.build_user_text("1girl", "", "", "")
    assert "<tags>1girl</tags>" in t
    assert "<draft>" not in t and "<focus>" not in t and "<lora_type>" not in t
    assert t.strip().endswith("Caption this image in the two-line format now.")


def test_system_prompt_switch():
    assert L.system_prompt_for(True) == L.SYSTEM_REFINE
    assert L.system_prompt_for(False) == L.SYSTEM_FRESH
    assert "draft" in L.SYSTEM_REFINE.lower()


def test_build_messages_structure():
    m = L.build_messages("SYS", "USER", "QUJD")
    assert m[0] == {"role": "system", "content": "SYS"}
    content = m[1]["content"]
    assert content[0] == {"type": "text", "text": "USER"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"


def test_clean_caption_strips_preamble_and_quotes():
    assert L.clean_caption('Sure, here is the caption: "a red cat sitting."') == "a red cat sitting."
    assert L.clean_caption('  "1girl, solo, outdoors"  ') == "1girl, solo, outdoors"


def test_clean_caption_collapses_newlines_and_bullets():
    assert L.clean_caption("- a dog in\na field") == "a dog in a field"


def test_clean_caption_leaves_plain_text():
    assert L.clean_caption("a girl standing in a field") == "a girl standing in a field"


def test_is_refusal():
    assert L.is_refusal("I'm sorry, I can't help with that")
    assert L.is_refusal("As an AI, I cannot")
    assert not L.is_refusal("a girl standing in a field")


def test_parse_fused_output_splits_prose_and_tags():
    raw = "A girl kneels on a bed, looking at the viewer.\nTAGS: 1girl, kneeling, on bed, looking at viewer"
    prose, tags = L.parse_fused_output(raw)
    assert prose == "A girl kneels on a bed, looking at the viewer."
    assert tags == "1girl, kneeling, on bed, looking at viewer"


def test_parse_fused_output_no_tags_marker():
    prose, tags = L.parse_fused_output("Just prose, no tag line here.")
    assert prose == "Just prose, no tag line here."
    assert tags == ""  # don't clobber existing .tags when the model omits the marker


def test_parse_fused_output_strips_preamble_and_normalizes_tags():
    raw = 'Sure, here is the caption: A red-haired woman stands outdoors.\nTAGS: 1girl,  solo ,\noutdoors'
    prose, tags = L.parse_fused_output(raw)
    assert prose == "A red-haired woman stands outdoors."
    assert tags == "1girl, solo, outdoors"


def test_clean_tags_normalizes():
    assert L.clean_tags("TAGS: a,  b , , c\n") == "a, b, c"


def test_user_text_includes_character_block():
    block = "<characters>\nasuka: red hair\n</characters>\n<style_anchor>@s</style_anchor>"
    t = L.build_user_text("1girl", "a girl", "", "character", block)
    assert "<characters>\nasuka: red hair\n</characters>" in t
    assert "<style_anchor>@s</style_anchor>" in t
    # block sits before the closing instruction line
    assert t.strip().endswith("Fuse and verify; output the two lines now.")


def test_user_text_without_block_is_unchanged():
    t = L.build_user_text("1girl", "", "", "")
    assert "<characters>" not in t and "<style_anchor>" not in t


def test_system_prompts_mention_token_and_style_anchor():
    for sp in (L.SYSTEM_REFINE, L.SYSTEM_FRESH):
        assert "token" in sp.lower()
        assert "style_anchor" in sp.lower()
