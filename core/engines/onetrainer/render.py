"""The final previews: render the shipped LoRA's preview prompts after the run.

OneTrainer's own previews are taken on a timer and stop at the last interval;
nothing renders the state the file is finally saved from. run_H (2026-09-04)
sampled at step 989 of 1139 and looked perfect there, then shipped the state at
1139, which rendered a mosaic. "The samples look fine" had meant "the LoRA was
fine at the last sample".

So once the file has passed every check (and, if the end overshot, once the
snapshot has taken its place), the session runs one more OneTrainer pass:
onetrainer_render.py, under OneTrainer's interpreter from OneTrainer's root,
exactly the way train.py is run. It loads the base model, applies the shipped
file through OneTrainer's own LoRA loader, and renders the run's samples.json
prompts into workspace/samples/final. About five minutes on the 4060 Ti: the
model load is most of it.

It is a courtesy, never a gate. The LoRA is an hour old and verified before this
starts; a render that fails is a note in trainer.log and in the completion
notice, and the run still lands as DONE.
"""
import os
from pathlib import Path

# The OneTrainer-side script. Never imported by the app: it runs where
# OneTrainer's modules are importable and the app's are not.
SCRIPT = str(Path(__file__).resolve().with_name("onetrainer_render.py"))

# Where the previews land -- beside the trainer's own, under the run's workspace,
# so Home's "Samples" button opens onto both.
FINAL_SUBDIR = ("workspace", "samples", "final")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def destination(run_dir: str) -> str:
    return str(Path(run_dir).joinpath(*FINAL_SUBDIR))


def args(config_path: str, lora_path: str, dest_dir: str) -> list:
    """The script's arguments: the run config (which base model, which weight
    types, which samples file), the file to render, and where to put the images."""
    return ["--config-path", config_path, "--lora", lora_path, "--destination", dest_dir]


def command(install, config_path: str, lora_path: str, dest_dir: str) -> list:
    """The full argv, for anyone spawning it by hand."""
    return [install.python, SCRIPT, *args(config_path, lora_path, dest_dir)]


def count_images(dest_dir: str) -> int:
    """How many previews a render actually wrote, whatever its exit code said."""
    folder = Path(dest_dir or "")
    if not folder.is_dir():
        return 0
    return sum(1 for root, _dirs, files in os.walk(folder)
               for name in files if Path(name).suffix.lower() in _IMAGE_EXTS)
