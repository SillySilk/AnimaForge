# AnimaForge — Technical Overview

> **Purpose of this document.** A self-contained briefing on what AnimaForge is, what it
> does, how it is built, and where its boundaries are — written so another large language
> model can hold the full context when you say *"I'm building for this training app."*
> It reflects the current state of the codebase (post the AI-character-naming removal).

---

## 1. What it is, in one paragraph

**AnimaForge** is a local, single-user **Windows desktop GUI** for training **LoRA
adapters** for the **Anima** image-generation model. It is a focused front-end over Kohya
**`sd-scripts`**: it takes a folder of training images, captions them (booru tags +
natural-language description), computes sane training
parameters, generates the `sd-scripts` config, and launches/monitors training — all from
one app, with no cloud dependency. It is **purpose-built for Anima**, not a general SDXL
trainer; the training defaults and the `network_module` are Anima-specific and the SDXL
flags are deliberately absent.

It **consumes/produces LoRAs and serves training**; it is the *producer* side of a LoRA
workflow. (For contrast, the sibling project *ForgeTale* is a fiction-writing engine that
*consumes* LoRA adapters at inference time. AnimaForge is the trainer.)

---

## 2. The model it trains: Anima

Anima (CircleStone Labs / Comfy Org, on Hugging Face as
[`circlestone-labs/Anima`](https://huggingface.co/circlestone-labs/Anima)) is **not SDXL**.
Key facts the trainer is built around:

- A **~2B-parameter DiT** (Diffusion Transformer, NVIDIA Cosmos base).
- Text encoder: **Qwen3-0.6B** (`qwen_3_06b_base.safetensors`).
- VAE: **Qwen-Image VAE** (`qwen_image_vae.safetensors`).
- Base checkpoint: `anima-base-v1.0.safetensors` (the DiT).

Because it is a DiT and not a UNet/SDXL model, the usual SDXL knobs (`v2`, `sdxl`,
`clip_skip`, `xformers`, `no_half_vae`, `min_snr_gamma`) **do not apply and are omitted**.
The LoRA is attached via a dedicated `sd-scripts` network module, **`networks.lora_anima`**.

The three model files are user-supplied; the Setup tab can auto-detect them by scanning
common Stable-Diffusion install layouts (Forge, ComfyUI, A1111).

---

## 3. Tech stack

| Layer | Choice |
|---|---|
| Language | Python **3.10 / 3.11** (pinned; an isolated standalone Python is downloaded if none found) |
| GUI | **PySide6** (Qt 6) |
| Training backend | Kohya **`sd-scripts`** (pinned git submodule, installed editable) |
| ML runtime | **PyTorch CUDA 12.1**, `onnxruntime-gpu` |
| Auto-tagging | **WD14** tagger (ONNX) → booru tags |
| Captioning | **JoyCaption** → natural-language description |
| Optimizer | **Prodigy** (learning-rate-free; converges fast on Anima) |
| Config format | TOML (`sd-scripts` config), JSON (per-dataset/per-run state), QSettings (app config) |
| Test harness | Custom `tests/run_tests.py` (a minimal pytest-free runner; the environment has no pytest) |

**Hardware reality:** NVIDIA GPU + CUDA required (no CPU/AMD path). ~16 GB VRAM is the
comfortable default; an opt-in low-VRAM mode reaches down toward an 8 GB practical floor.

**Install model:** `install.bat` → `scripts/bootstrap.py` builds **one self-contained
`.venv`** with the entire stack (GUI + torch + editable `sd-scripts` + onnxruntime) and
initializes the `sd-scripts` submodule. `launch.bat` runs `main.py`. No system-wide Python
changes. (`requirements.txt` only lists the thin GUI deps — it is **not** enough to train;
the heavy stack comes from the bootstrap.)

---

## 4. Architecture & code structure

The repo follows a strict **pure-core / thin-UI** split. Business logic lives in pure,
testable functions and QProcess/QThread wrappers under `core/`; `ui/` is Qt views that call
into them. Heavy ML work runs in **subprocesses**, not in the GUI process.

```
AnimaForge/
├── main.py                 # entry point: high-DPI setup, builds MainWindow
├── install.bat / launch.bat
├── core/                   # pure logic + subprocess/thread wrappers (no Qt views)
├── ui/                     # PySide6 views (one file per tab + dialogs/widgets)
├── scripts/                # stdlib-only scripts run as subprocesses in the venv
├── sd-scripts/             # pinned Kohya submodule (the actual trainer)
├── tests/                  # custom runner; ~259 tests, GPU-free (mocks/offscreen Qt)
└── docs/superpowers/       # design specs + implementation plans (archival history)
```

### `core/` modules (what each owns)

| Module | Responsibility |
|---|---|
| `config_generator.py` | The **Anima training config**. `HARDCODED_CONFIG` holds the architecture-correct defaults; generates the `sd-scripts` TOML. |
| `step_calculator.py` | Computes epochs/repeats to hit a target step count (default **500 steps**; Anima+Prodigy converges fast). |
| `trainer.py` | `TrainingProcess` — QProcess wrapper that launches & monitors `sd-scripts` training; emits progress/log/finished signals. |
| `tagger.py` | `TaggerProcess` — runs WD14 auto-tagging as a subprocess. |
| `joycaption.py` | `JoyCaptionProcess` — runs JoyCaption natural-language captioning as a subprocess. |
| `dataset_manager.py` | Scans image folders; owns the **sidecar caption model** (`.tags` / `.nl` / `.txt`) and the mechanical Combine (with `normalize_tags`). |
| `naming.py` | The **filename naming convention** (v2) — parse/validate/auto-format (see §6). |
| `characters.py` | Per-dataset **character roster** + style anchor + per-image cast assignments (one JSON per folder). |
| `workflow.py` | Single source of truth for **readiness**: the Load → Name → Caption → Train progress state. |
| `quick_run.py` | Pure planning for the Home "Quick Run" unattended pipeline (which phases to run). |
| `batch.py` | Serializable **run definition** + sequential batch runner (one GPU job at a time). |
| `sets.py` | Named training **presets** ("sets") + crash-recovery marker for interrupted runs. |
| `paths.py` | Per-run output folders (LoRA, `sample/`, logs, configs, resumable state co-located). |
| `lowvram.py` | Opt-in, **quality-neutral** low-VRAM recipe + in-memory session state. |
| `settings.py` | Typed app configuration over QSettings (paths, connections, defaults, advanced knobs). |
| `sample_prompts.py` | Draw random verbatim `.txt` caption blocks to seed the Train tab's sample-prompt box. |
| `forge_api.py` / `forge_worker.py` | Deliver a trained LoRA to Forge/A1111 and test-render it via REST API. |
| `model_locations.py` | First-run guess of where the three Anima model files live. |
| `gpu_check.py` | Best-effort free-VRAM probe (nvidia-smi) for the pre-launch guard. |
| `env.py` | Resolves the Python interpreter for ML subprocesses (the unified `.venv`). |

### `scripts/` (subprocess entry points, stdlib-only where possible)

- `bootstrap.py` — builds the `.venv` and installs the full stack.
- `joycaption_run.py` — JoyCaption captioning; writes `.nl` sidecars.
- `gen_assets.py` — procedurally generates the UI art (Pillow), idempotent.

### `ui/` (the views)

Tabs are a `QStackedWidget`; a slim **progress rail** (`progress_rail.py`) sits pinned above
the content showing Load → Name → Caption → Train. `image_editor.py` is the large per-image
modal (preview + caption editor + cast assignment). `name_validate_view.py` is the filename
validator. `collapsible.py`, `run_progress.py`, `lowvram_dialog.py` are shared widgets.

---

## 5. Features (the six tabs and the workflow)

The window is six tabs, indexed in this order, plus the always-visible progress rail:

0. **Home** — a dashboard + **"Quick Run" cockpit**: pick a folder, name, subject type, and
   target steps, then run an **unattended pipeline** (optionally caption, then train) with a
   live progress widget. Split actions: *Run captioning*, *Run training*, *Add to batch*.
1. **Setup** — model file locations (auto-detect the three Anima files) and environment paths.
2. **Dataset** — load an image folder; run the **captioning pipeline** (Auto-Tag → Describe
   → Combine); browse images as cards; open the per-image editor;
   launch the **filename validator**.
3. **Characters** — roster review (tokens + recognition descriptions), a dataset-wide
   **@-style anchor**, a selectable **frame grid** for bulk cast assignment, and a
   "Character Doctor" for find/replace across captions. Roster can be **detected from
   filenames** or edited manually.
4. **Train** — the source-of-truth training controls: LoRA name, subject type, target steps,
   Anima/optimizer settings (in collapsible sections), sample-preview prompts, **Low-VRAM**
   dialog, start training, and *Add to batch*. Shows the shared run-progress widget.
5. **Batch** — a queue of serialized run definitions executed sequentially (the GPU only ever
   runs one job at a time).

### The captioning pipeline (non-destructive, sidecar-based)

Each image gets **separate sidecar files** so steps never clobber each other:

| Sidecar | Produced by | Content |
|---|---|---|
| `.tags` | WD14 tagger | booru tags |
| `.nl` | JoyCaption | natural-language description |
| `.txt` | the **Combine** step | the **merged caption training actually reads** |

**Combine** is pure Python: it merges `.nl` + `.tags` (with the trigger/prefix and any
per-image character tokens) into the `.txt`, applying `normalize_tags` (lowercase,
spaces-not-underscores, de-dup) to the tag tail. **The entire pipeline runs with no LLM
serving and no network** — captioning is JoyCaption + WD14 locally, and the merge is mechanical.

### Training: how parameters are decided

- `step_calculator.calculate_training_params(image_count, target_steps=500)` solves
  `total_steps = image_count * repeats * epochs / batch_size` for epochs/repeats.
- `config_generator` emits the Anima-correct `sd-scripts` config. Defaults
  (`HARDCODED_CONFIG`): bf16 save/precision, **1024×1024**, batch 4, gradient checkpointing,
  `timestep_sampling=sigmoid`, `vae_chunk_size=64`, **`network_module=networks.lora_anima`**,
  `network_dim=16`, `network_alpha=8`, latent caching to disk, tensorboard logging.
- Optimizer: **Prodigy** (learning-rate-free).
- Subject types (character / object-concept / style) bias the step bands and captioning.

### Other notable features

- **Low-VRAM mode** — opt-in, acknowledged, **quality-neutral**: same precision/resolution/
  effective batch; trades only speed via micro-batching + gradient accumulation + CPU
  block-swap. Presets for 16/12/10/8 GB. Off by default, session-only, with an active
  indicator. 8 GB is the unverified practical floor.
- **Sets** (named presets) + **crash recovery** — a run-in-progress marker lets the app
  offer to resume an interrupted run.
- **Per-run output isolation** — every run gets its own folder (LoRA + `sample/` previews +
  logs + configs + resumable state together).
- **Forge/A1111 delivery** — copy a finished LoRA into Forge and test-render it over the
  REST API (Forge started with `--api`).

---

## 6. The filename naming convention (v2) — important seam

Training captions are driven by filenames. AnimaForge enforces a hard convention:

```
NAME_SERIAL_CATEGORY.ext
```

Three underscore-separated fields; the underscore is **structural** and never appears inside
a field:

- **NAME** — the subject(s). Spaces allowed inside one subject (`Yogi bear`); **multiple
  subjects are joined by a hyphen** (`Homer-Marge-Lisa` = three subjects). Case-insensitive.
- **SERIAL** — zero-padded **3-digit** counter within a group (`001`+). Organizational only.
- **CATEGORY** — `Character`, `Object`, or `Style` (case-insensitive in, canonicalized out).
  **One category per project.**

Examples: `Yogi bear_001_Character.png`, `Homer-Marge_004_Character.jpg`,
`Picnic basket_012_Object.png`, `Morning-Evening_002_Style.gif`.

The captioner reads the **NAME subjects** (hyphen-split) as the trigger tokens. The
**validator screen** (`name_validate_view.py`) flags non-conforming files and offers a
one-click **Auto-format** that bulk-renames a whole folder to the convention (renaming the
caption sidecars alongside, two-pass so re-serialization can't collide). A per-file rename
box remains for stragglers.

**Design philosophy:** training image files are **disposable inputs** — they do not need to
survive past training, so renaming/converting/bulk-rewriting them is fair game. The
authoritative naming is expected to come from a **separate companion app** that renames a set
to exact convention; AnimaForge's validator is the safety-net.

---

## 7. Cross-cutting design principles

- **Pure-core / thin-UI:** logic is in testable `core/` functions; `ui/` only views. Mirrors
  a deliberate, enforced separation across the codebase.
- **Subprocess isolation for ML:** tagging, captioning, and training each run as a
  separate process driven by a `QObject`/QProcess wrapper that emits `log_line` /
  `finished` / `progress` signals. A crash takes down the job, not the GUI.
- **One GPU job at a time:** the batch runner serializes runs; nothing fans out onto the GPU.
- **Anima-specific, not general:** training defaults encode Anima's architecture; SDXL flags
  are intentionally absent.
- **LLM-optional:** the full Tag → Describe → Combine → Train path needs no LLM.
- **State is per-folder / per-run JSON:** the character roster
  (`animaforge_characters.json`), sets, and run state are plain JSON co-located with their
  data, not a central database.

---

## 8. Testing

- Runner: `.venv\Scripts\python.exe tests\run_tests.py` — a minimal pytest-free harness that
  discovers `test_*` functions and injects a `tmp_path` when declared.
- **~259 tests**, all **GPU-free**: ML backends are mocked and Qt runs offscreen
  (`QT_QPA_PLATFORM=offscreen`), so the suite runs anywhere without a model or a card.
- Specs and implementation plans live under `docs/superpowers/` as an archival record of how
  features were designed and built.

---

## 9. Scope boundaries — what AnimaForge does NOT do

State these plainly so an assisting model doesn't assume capabilities that aren't there:

- It **trains LoRAs only** — it does not do full fine-tunes, and it is **not an inference/
  image-generation app** (it can hand a LoRA to Forge and request a test render, but it does
  not generate images itself).
- It is **Anima-only**, not a general SDXL/Flux/SD1.5 trainer.
- It is **Windows + NVIDIA/CUDA only** — no CPU, AMD, or macOS path.
- It is **local and single-user** — no cloud, no multi-tenant, no remote queue.
- **No LLM serving at all.** Earlier versions leaned on an external LM Studio server for an
  AI character-naming feature, a vision-LLM caption "refine" pass, and AI-generated sample
  prompts. All of that is gone: character identity comes from **filenames** (the NAME field)
  and manual roster editing, captions are JoyCaption + WD14 merged mechanically, and sample
  prompts are drawn verbatim from the dataset's own captions. AnimaForge no longer talks to
  any local or remote LLM server.
- It **does not manage the model files** — the three Anima files are user-supplied (it only
  helps locate them).
