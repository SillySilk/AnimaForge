"""Generate natural-language captions for a folder of images using JoyCaption.

Writes one sidecar caption file per image (default extension '.nl'), leaving the
booru-tag '.tags' files and the merged '.txt' files untouched. Intended to be
launched as a subprocess by core/joycaption.py inside the sd-scripts venv.

Model: fancyfeast/llama-joycaption-beta-one-hf-llava (anime/booru-aware, uncensored).
The model (~17GB) downloads to a local cache on first run.
"""
import argparse
import sys
from pathlib import Path

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"
DEFAULT_PROMPT = "Write a descriptive caption for this image in a formal tone."


def log(msg: str):
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder")
    parser.add_argument("--ext", default=".nl", help="sidecar extension to write")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--cache_dir", default=None, help="model cache directory")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_time", type=float, default=90.0,
                        help="max seconds to spend generating one caption before moving on")
    parser.add_argument("--no_4bit", action="store_true",
                        help="disable 4-bit quantized loading (use full bf16)")
    args = parser.parse_args()

    folder = Path(args.image_folder)
    if not folder.is_dir():
        log(f"[JoyCaption] ERROR: folder not found: {folder}")
        return 1

    images = sorted(
        [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED],
        key=lambda f: f.name.lower(),
    )
    if not images:
        log("[JoyCaption] No images found.")
        return 1

    # Decide which images need captioning before loading the heavy model.
    todo = []
    for img in images:
        sidecar = img.with_suffix(args.ext)
        if sidecar.is_file() and sidecar.read_text(encoding="utf-8").strip() and not args.overwrite:
            continue
        todo.append(img)

    if not todo:
        log("[JoyCaption] All images already have captions (use overwrite to redo).")
        return 0

    log(f"[JoyCaption] Loading model {MODEL_ID} (first run downloads ~17GB)…")
    try:
        import torch
        from PIL import Image
        from transformers import LlavaForConditionalGeneration, AutoProcessor
    except Exception as e:
        log(f"[JoyCaption] ERROR importing dependencies: {e}")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = args.cache_dir or str(Path(__file__).resolve().parents[1] / "joycaption_model")

    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        loaded_4bit = False
        # JoyCaption Beta One is ~8B params (~16GB in bf16) — too large to run
        # comfortably on a 16GB GPU, where it spills into shared memory and crawls.
        # Load it 4-bit (NF4) so it fits in ~6GB and runs fast.
        if device == "cuda" and not args.no_4bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    # Keep the vision tower + projector in bf16 — quantizing them causes
                    # "self and mat2 must have the same dtype (BFloat16 and Byte)" because
                    # Llava's vision path doesn't dequantize 4-bit weights.
                    llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                )
                model = LlavaForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    quantization_config=bnb,
                    torch_dtype=torch.bfloat16,
                    device_map={"": 0},
                    cache_dir=cache_dir,
                )
                loaded_4bit = True
                log("[JoyCaption] Loaded in 4-bit NF4 (fits 16GB VRAM).")
            except Exception as e:
                log(f"[JoyCaption] 4-bit load failed ({e}); falling back to full bf16.")
        if not loaded_4bit:
            model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16, cache_dir=cache_dir,
            ).to(device)
        model.eval()
    except Exception as e:
        log(f"[JoyCaption] ERROR loading model: {e}")
        return 1

    log(f"[JoyCaption] Model ready on {device} ({'4-bit' if loaded_4bit else 'bf16'}). "
        f"Captioning {len(todo)} image(s)…")

    convo = [
        {"role": "system", "content": "You are a helpful image captioner."},
        {"role": "user", "content": args.prompt},
    ]
    convo_string = processor.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )

    done = 0
    for img in todo:
        try:
            image = Image.open(img).convert("RGB")
            inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to(device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            with torch.no_grad():
                generate_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    max_time=args.max_time,  # hard cap per image so one never stalls the batch
                )[0]
            generate_ids = generate_ids[inputs["input_ids"].shape[1]:]
            caption = processor.tokenizer.decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            if not caption:
                log(f"[JoyCaption] SKIP {img.name}: no caption produced (hit {args.max_time:.0f}s cap?).")
                continue
            img.with_suffix(args.ext).write_text(caption, encoding="utf-8")
            done += 1
            log(f"[JoyCaption] ({done}/{len(todo)}) {img.name}: {caption[:60]}…")
        except Exception as e:
            log(f"[JoyCaption] ERROR on {img.name}: {e}")

    log(f"[JoyCaption] Done. Wrote {done} caption(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
