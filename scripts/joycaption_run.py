"""Generate natural-language captions for a folder of images using JoyCaption.

Writes one sidecar caption file per image (default extension '.nl'), leaving the
booru-tag '.tags' files and the merged '.txt' files untouched. Intended to be
launched as a subprocess by core/joycaption.py inside the sd-scripts venv.

Default model: fancyfeast/llama-joycaption-beta-one-hf-llava (anime/booru-aware,
uncensored); --model_id takes any Hugging Face vision-language repo that loads
through AutoModelForImageTextToText. Setup owns the choice; see core/describe_models.py.

Pass --cache_dir ONLY to override where models live. Omitting it lets Hugging Face
use HF_HUB_CACHE / HF_HOME / ~/.cache/huggingface/hub, so a model already on disk
(including via a directory junction) is found instead of re-downloaded.
"""
import argparse
import subprocess
import sys
import traceback
from pathlib import Path

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
DEFAULT_MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"
DEFAULT_PROMPT = "Write a descriptive caption for this image in a formal tone."

# bf16 needs roughly 2 bytes per parameter plus activations; 4-bit needs about a
# quarter of that. Used only to decide whether an unquantized fallback is viable.
_BF16_MIN_FREE_MB = 15000


def log(msg: str):
    print(msg, flush=True)


def _transformers_version() -> str:
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return "unknown"


def _free_vram_mb():
    """Free VRAM in MB, or None when nvidia-smi is absent or unparseable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        for line in out.stdout.splitlines():
            digits = "".join(c for c in line if c.isdigit())
            if digits:
                return int(digits)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder")
    parser.add_argument("--ext", default=".nl", help="sidecar extension to write")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--name", default="",
                        help="trigger word to call the subject by when no character is assigned")
    parser.add_argument("--cache_dir", default=None, help="model cache directory")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID,
                        help="Hugging Face repo ID of the describe model")
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

    model_id = args.model_id
    log(f"[JoyCaption] Loading model {model_id}…")
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception:
        log("[JoyCaption] ERROR importing dependencies:")
        log(traceback.format_exc())
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # cache_dir is only passed when explicitly given. Passing one unconditionally
    # overrides HF_HUB_CACHE / HF_HOME / ~/.cache/huggingface/hub, which made an
    # already-downloaded model (or a junction to one) invisible and triggered a
    # redundant ~17GB download. No flag == Hugging Face looks where it normally does.
    cache_kw = {"cache_dir": args.cache_dir} if args.cache_dir else {}

    free_mb = _free_vram_mb()
    log(f"[JoyCaption] device={device} "
        f"vram_free={'unknown' if free_mb is None else str(free_mb) + 'MB'} "
        f"torch={torch.__version__} transformers={_transformers_version()}")
    log(f"[JoyCaption] cache_dir={args.cache_dir or 'auto (Hugging Face default)'}")

    try:
        processor = AutoProcessor.from_pretrained(model_id, **cache_kw)
        model = None
        loaded_4bit = False
        # The big describe models are ~8B params (~17GB in bf16) — too large to run
        # comfortably on a 16GB GPU, where they spill into shared memory and crawl.
        # Load 4-bit (NF4) so they fit in ~6GB and run fast.
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
                model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    quantization_config=bnb,
                    torch_dtype=torch.bfloat16,
                    device_map={"": 0},
                    **cache_kw,
                )
                loaded_4bit = True
                log("[JoyCaption] Loaded in 4-bit NF4.")
            except Exception:
                log("[JoyCaption] 4-bit load failed:")
                log(traceback.format_exc())
                # Falling through to bf16 silently was the old behaviour, and on a
                # card that can't hold ~17GB it doesn't fail — it spills to shared
                # memory and crawls, which looks identical to a hang. Fail fast.
                if free_mb is not None and free_mb < _BF16_MIN_FREE_MB:
                    log(f"[JoyCaption] ERROR: only {free_mb}MB VRAM free; the "
                        f"unquantized fallback needs ~{_BF16_MIN_FREE_MB}MB and would "
                        f"run unusably slowly. Fix the 4-bit load (is bitsandbytes "
                        f"installed?) or pick a smaller model in Setup.")
                    return 1
        if model is None:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, **cache_kw,
            ).to(device)
        model.eval()
    except Exception:
        log(f"[JoyCaption] ERROR loading model {model_id}:")
        log(traceback.format_exc())
        log("[JoyCaption] Full detail is in logs/caption.log.")
        return 1

    log(f"[JoyCaption] Model ready on {device} ({'4-bit' if loaded_4bit else 'bf16'}). "
        f"Captioning {len(todo)} image(s)…")

    # Named people go into the prompt so the prose uses their names (core/characters.py).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core import characters

    cast = characters.load(str(folder))

    def convo_for(img):
        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": characters.describe_prompt_for_image(
                args.prompt, cast, img.name, trigger=args.name)},
        ]
        return processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

    done = 0
    for img in todo:
        try:
            convo_string = convo_for(img)
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
