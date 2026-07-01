<div align="center">

<img src="https://raw.githubusercontent.com/SillySilk/AnimaForge/civitai-assets/hero.png" alt="AnimaForge" width="100%">

# AnimaForge

### A free, local, one-click LoRA trainer — hand-tuned for **Anima**

[![License: MIT](https://img.shields.io/badge/License-MIT-d4af37.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Windows-blue)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76b900)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-3776ab)

**Point it at a folder of images, name your character, hit Train.**
No subscriptions, no queues, no uploading your dataset to someone else's server.

[**📖 Read the launch article on Civitai**](https://civitai.com/articles/31944) · [**🧩 Companion: Animus Sorter**](https://github.com/SillySilk/AnimusSorter)

</div>

---

## ✨ Why AnimaForge

It's **not** a general-purpose SDXL tool with Anima bolted on. It's built for **one** model, and that focus is the whole point. The training config is **hand-tuned from a deep analysis of the Anima model itself**, so you get strong results without becoming a hyperparameter expert.

A true **one-click pipeline** — **Auto-Tag → Describe → Combine → Train** — with everything dialed in.

<div align="center">
<img src="https://raw.githubusercontent.com/SillySilk/AnimaForge/civitai-assets/home.png" alt="Home cockpit" width="80%">
</div>

## 🚀 What it does

| | |
|---|---|
| **🏷 Auto-Tag (WD14)** | Fast, accurate booru-style tags for every image. |
| **📝 Describe (JoyCaption)** | Natural-language captions that capture what tags miss. |
| **🧩 Combine** | Merges both into clean, training-ready captions. |
| **🔥 Train** | Kicks off training with the Anima-tuned settings already set. |
| **🎭 Name Cast** | Name the people in your set once; trigger words stay consistent across the whole dataset. |
| **📦 Batch** | Queue multiple LoRAs and walk away — unattended, survives a restart. |
| **🧰 Low-VRAM mode** | Reaches smaller cards (down to ~8 GB) at the same quality, just slower. |

<div align="center">
<img src="https://raw.githubusercontent.com/SillySilk/AnimaForge/civitai-assets/dataset.png" alt="Dataset gallery" width="49%">
<img src="https://raw.githubusercontent.com/SillySilk/AnimaForge/civitai-assets/characters.png" alt="Name Cast" width="49%">
<img src="https://raw.githubusercontent.com/SillySilk/AnimaForge/civitai-assets/train.png" alt="Train tab" width="49%">
<img src="https://raw.githubusercontent.com/SillySilk/AnimaForge/civitai-assets/batch.png" alt="Batch queue" width="49%">
</div>

## 🧩 Pairs with Animus Sorter

Start clean: **[Animus Sorter](https://github.com/SillySilk/AnimusSorter)** sorts a pile of scraped images into subject bins and renames them `NAME_SERIAL_CATEGORY` — the exact convention AnimaForge reads to pre-fill **Name Cast** automatically. Sort and name once; AnimaForge carries it the rest of the way.

## ⚡ Quick start

```bat
install.bat   :: one-time — builds a self-contained .venv with the full stack
launch.bat    :: start training
```

`install.bat` finds or downloads a compatible Python (3.10/3.11), builds an isolated `.venv`, and installs everything (PySide6 GUI, PyTorch CUDA 12.1, the Kohya `sd-scripts` backend). No system-wide Python changes.

<div align="center">
<img src="https://raw.githubusercontent.com/SillySilk/AnimaForge/civitai-assets/setup.png" alt="Setup tab" width="80%">
</div>

## 📋 Requirements

- **Windows** + an **NVIDIA GPU with a recent driver** (CUDA). There's no CPU/AMD path — Anima training needs CUDA.
- **~16 GB VRAM** comfortable; a low-VRAM mode reaches smaller cards, **8 GB is the practical floor**.
- **The three Anima model files** (the Setup tab can auto-detect them from a models folder):
  - DiT checkpoint — `anima-base-v1.0.safetensors`
  - Qwen3 text encoder — `qwen_3_06b_base.safetensors`
  - Qwen-Image VAE — `qwen_image_vae.safetensors`

  From Hugging Face: [`circlestone-labs/Anima`](https://huggingface.co/circlestone-labs/Anima).
- First install downloads ~2.5 GB (a self-contained Python + PyTorch stack).

> **Why NVIDIA-only? (no AMD on Windows)** — AnimaForge trains through PyTorch + Kohya's `sd-scripts`, and that stack only has a working GPU path on **NVIDIA/CUDA on Windows**. AMD's equivalent (ROCm) is **Linux-only** — there are no Windows ROCm builds of PyTorch — so there's no supported way to run the training backend on an AMD card under Windows. The only experimental option is **ZLUDA** (a CUDA-translation layer), but it's unproven for training workloads like this and would be fragile; **DirectML** exists but is too slow and incomplete for this kind of training. But I'll keep looking at it, and if I find a way to wire it in, I will.

## 🤖 LM Studio (optional)

The core pipeline runs **without any LLM**. If you run [LM Studio](https://lmstudio.ai/), three optional enhancements light up (LLM caption refine, Name Cast "find characters", sample-prompt generation) — each with a non-LLM fallback. Any OpenAI-compatible model works.

## 💛 Free & open source

100% free. No paid tier, no credits, no Buzz, no catch. MIT licensed — install it, use it, train as many LoRAs as you want.

---

<div align="center">

**Built for the Anima community. Forge something great.** 🔥

</div>
