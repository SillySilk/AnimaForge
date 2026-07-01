"""Generate sample/preview prompts from a dataset's prevalent caption keywords.

Flow: mine the most common comma-separated tokens across the dataset's `.tags`/`.txt`
sidecars, hand them to the local LM Studio model, and ask for a few concrete, diverse
sample prompts that represent the series. The trigger word is NOT included in the
prompts themselves — config_generator/prepare_sample_args prepends it at write time.

Pure helpers (collect_keywords / build_messages / parse_prompts) are unit-tested;
PromptGenWorker is the thin Qt thread that performs the HTTP call.
"""
import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

from PySide6.QtCore import QThread, Signal

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
CAPTION_EXTS = (".tags", ".txt", ".nl")

# Generic boilerplate that says nothing about *this* dataset — excluded from keywords.
_STOPWORDS = {
    "masterpiece", "best quality", "high quality", "highres", "absurdres",
    "ultra detailed", "detailed", "score_9", "score_8", "score_7",
    "score_8_up", "score_7_up", "very aesthetic", "newest", "general",
}


def _tokenize(caption: str):
    """Split a caption into normalized comma-separated tokens."""
    for raw in caption.split(","):
        tok = re.sub(r"\s+", " ", raw).strip().lower()
        if tok:
            yield tok


def collect_keywords(captions, top_n: int = 20, trigger: str = ""):
    """Return the top_n most frequent caption tokens across `captions`.

    `captions` is an iterable of caption strings. The trigger word and generic
    quality boilerplate are excluded so the result describes the actual subject matter.
    """
    trig = (trigger or "").strip().lower()
    counts = Counter()
    for cap in captions:
        # de-dup within a single caption so one image can't inflate a token's count
        for tok in set(_tokenize(cap)):
            if tok == trig or tok in _STOPWORDS:
                continue
            counts[tok] += 1
    return [tok for tok, _ in counts.most_common(top_n)]


def read_dataset_captions(folder: str):
    """Read the best available caption per image (.tags > .txt > .nl). Returns a list."""
    folder_p = Path(folder)
    if not folder_p.is_dir():
        return []
    captions = []
    seen_stems = set()
    for img in sorted(folder_p.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS or img.stem in seen_stems:
            continue
        seen_stems.add(img.stem)
        for ext in CAPTION_EXTS:
            side = img.with_suffix(ext)
            if side.is_file():
                try:
                    text = side.read_text(encoding="utf-8").strip()
                except OSError:
                    text = ""
                if text:
                    captions.append(text)
                    break
    return captions


def grab_caption_blocks(folder, n: int, rng=None):
    """Return up to `n` random, verbatim caption blocks from a captioned dataset.

    Each block is the full merged `.txt` caption the trainer reads (natural language
    + tags together). Empty/whitespace-only captions are skipped. When fewer than `n`
    captions exist, all of them are returned (shuffled). `n <= 0`, a missing folder, or
    no captions yields []. Pass a seeded `random.Random` for deterministic selection.
    """
    from core.dataset_manager import scan_folder

    if n <= 0:
        return []
    items = scan_folder(folder) if folder else []
    blocks = [cap for cap in (d.get("caption", "").strip() for d in items) if cap]
    if not blocks:
        return []
    rng = rng or random
    if len(blocks) <= n:
        out = list(blocks)
        rng.shuffle(out)
        return out
    return rng.sample(blocks, n)


def build_messages(keywords, trigger: str = "", lora_type: str = "", n: int = 3, characters=None):
    """Build OpenAI-style chat messages asking for n sample prompts. Pure.

    When `characters` (roster tokens) are given, the model is asked to feature them by name so
    previews show the actual named characters; otherwise it falls back to the generic guidance.
    """
    kw = ", ".join(keywords) if keywords else "(no keywords found)"
    type_hint = f" The LoRA subject type is: {lora_type}." if lora_type else ""
    chars = [c.strip() for c in (characters or []) if c and c.strip()]
    if chars:
        char_line = (
            f"Feature these characters by their exact name token where it fits, varying which one "
            f"appears across the prompts: {', '.join(chars)}. Use the tokens verbatim. "
        )
        name_rule = "Do NOT include any trigger word — that is added automatically."
    else:
        char_line = ""
        name_rule = "Do NOT include any trigger word or character name — that is added automatically."
    system = (
        "You write concise, concrete image-generation prompts for testing a LoRA during "
        "training. Each prompt is a single line of comma-separated tags/phrases describing "
        "one coherent scene. No numbering, no markdown, no commentary — just the prompts, "
        "one per line."
    )
    user = (
        f"These are the most common tags across the training dataset:\n{kw}\n\n"
        f"Write exactly {n} diverse sample prompts that represent recurring elements of this "
        f"dataset (vary pose, framing, and setting between them).{type_hint} {char_line}"
        f"{name_rule} "
        f"Output exactly {n} lines, one prompt per line, nothing else."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_prompts(text: str, n: int = 3):
    """Parse the model's reply into a clean list of up to n prompt lines. Pure."""
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        # strip leading list markers: "1.", "1)", "-", "*", "•"
        s = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", s)
        s = s.strip().strip('"').strip("'").strip()
        if s:
            out.append(s)
        if len(out) >= n:
            break
    return out


def _chat(url: str, model: str, messages, max_tokens: int = 400, timeout: int = 120):
    """POST to {url}/chat/completions and return the assistant content string."""
    endpoint = url.rstrip("/") + "/chat/completions"
    payload = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if model.strip():
        payload["model"] = model.strip()
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode("utf-8"))
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    # Some reasoning models leave content empty and fill reasoning_content; prefer content.
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()


class PromptGenWorker(QThread):
    """Mine dataset keywords and ask LM Studio for n sample prompts, off the UI thread."""

    finished_ok = Signal(list)   # list[str] prompts
    failed = Signal(str)         # error message
    log_line = Signal(str)

    def __init__(self, folder, url, model, trigger="", lora_type="", n=3, characters=None, parent=None):
        super().__init__(parent)
        self._folder = folder
        self._url = url
        self._model = model
        self._trigger = trigger
        self._lora_type = lora_type
        self._n = n
        self._characters = characters or []

    def run(self):
        try:
            captions = read_dataset_captions(self._folder)
            if not captions:
                self.failed.emit("No captions found in the dataset — tag/caption it first.")
                return
            keywords = collect_keywords(captions, top_n=20, trigger=self._trigger)
            self.log_line.emit(f"[Prompts] Top keywords: {', '.join(keywords[:12])}")
            messages = build_messages(keywords, self._trigger, self._lora_type, self._n,
                                      self._characters)
            content = _chat(self._url, self._model, messages)
            prompts = parse_prompts(content, self._n)
            if not prompts:
                self.failed.emit("Model returned no usable prompts (try raising max tokens).")
                return
            self.finished_ok.emit(prompts)
        except Exception as e:  # noqa: BLE001 — surface any HTTP/parse error to the UI
            self.failed.emit(str(e))
