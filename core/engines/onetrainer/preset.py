"""The vendored OneTrainer preset and the patch that turns it into a run config.

assets/krea2_preset.json began as a copy of the preset that proved Krea 2 LoRA
training FITS on the 16 GB card (2026-08-07). "Viable" there meant the run
completed without an OOM -- it was never evidence that the run learned anything,
and the recipe guard in tests/test_trainer_preset.py locked in the memory knobs
while three learning knobs stayed wrong. run_B (2026-08-17) ran the full
1254 steps, saved cleanly, and produced a LoRA whose lora_B sat at ~1.7e-08
against a working LoRA's ~4.0e-02: a file that loads and changes nothing. Fixed
in the same pass:

  learning_rate    3e-4 -> 1.0    PRODIGY reads lr as a multiplier on the step
                                  size it estimates for itself, so 3e-4 scaled
                                  its estimate to nothing instead of setting it.
  output_model_format
     DIFFUSERS_LORA -> COMFY_LORA the image tool reads `diffusion_model.`-prefixed
                                  native names (blocks.N.attn.wq); DIFFUSERS_LORA
                                  writes the diffusers namespace, sharing not one
                                  key, so the loader applied nothing.
  lora_alpha       1.0 -> 16.0    alpha == rank, so the applied scale is 1.0
                                  whether or not the loader honours alpha.
  safeguard_warmup false -> true  this preset always warms up; Prodigy needs this
                                  on or the D estimate is distorted over exactly
                                  the steps that set the rate for the whole run.

Since 2026-09-04 the transformer trains as INT_W8A8_COMPRESSED: the same int8
quantization, with each layer's weights held nvCOMP-compressed on the card and
decoded on the fly (OneTrainer #1630, effective 4-5 bit). The decoded weights
are the int8 ones, so the learning knobs above stand; what changed is VRAM.
offload_fraction stays at 0.3 until a run with less is proven. Needs an
OneTrainer from 2026-08-07 or later (engine.COMPRESSION_MARKER) with
nvidia-nvcomp-cu13 installed in its venv. Nothing in the loader changed: the
system-memory spike while quantizing (core/sysmem.py) is exactly what it was.

What "proven" means for the compression, precisely (2026-09-04, 4060 Ti): two
one-epoch runs over the same 35 images, 17 steps each, identical but for the
data type. Compressed: loss 0.0802, ~3.4 s/step, strongest lora_B 9.44e-05.
Uncompressed: loss 0.0801, ~3.4 s/step, strongest lora_B 9.39e-05. Same
learning, same speed, 44% less VRAM. Both files fail verify.verify_output,
whose 1e-03 floor is sized for a full run -- 17 Prodigy steps move lora_B
four orders of magnitude off zero and no further -- so a short run is not a
verify pass and was wrongly called one in commit 8f24cdb. A full run through
the app is still the only end-to-end proof of the recipe.

Since 2026-09-05 two things guard the END of a run, after run_H (2026-09-04)
trained 67 clean-looking epochs and shipped a file that rendered a mosaic. Its
end-of-epoch loss had bottomed at epoch 52 and finished 38% higher; Prodigy's
step size had settled at 5.8e-04 against 2.9e-04..3.4e-04 on every healthy run.
  save_every 5 EPOCH             a snapshot in the shipped format every five
                                 epochs, so the session can ship the one from
                                 the lowest-loss epoch when verify says the end
                                 of the run overshot (core/engines/onetrainer/session.py).
                                 ~114 MB and ~2 s each, deleted once the run has
                                 settled which file ships.
  optimizer.prodigy_steps        patched per run to three quarters of the step
                                 count (steps.plan): Prodigy stops re-estimating
                                 its step size for the closing stretch. UNPROVEN
                                 as a recipe until a full run has shipped with
                                 it -- run_H's step size had already peaked by
                                 step 989 of 1139, so this alone would not have
                                 saved that run; the snapshots are what would.

Since 2026-10-07 previews render every tenth of the run instead of every 30
minutes: sample_after is patched per run to the epoch count / 10, rounded UP to
a whole epoch (SAMPLE_FRACTION), unit EPOCH. Ten preview passes per run, the
last one landing on or just before the final epoch rather than up to 30 minutes
short of it -- the blind stretch that let run_H's overshoot ship unseen.

Since 2026-10-08 snapshots are taken on the same tenth-of-the-run schedule as the
previews (save_every patched to sample_every_epochs, unit EPOCH): every preview
has a file behind it, so the epoch whose preview looked right is one that can
ship. The preset's fixed save_every 5 is overridden per run.

Also since 2026-10-08 Prodigy's step size is capped at PRODIGY_D_CAP. The preset
uses OneTrainer's plain PRODIGY (prodigyopt), which has no ceiling option -- and
which never receives optimizer.prodigy_steps either: OneTrainer passes that only
to PRODIGY_PLUS_SCHEDULE_FREE, so the per-run patch above has been a no-op. The
cap is applied by core/engines/onetrainer/onetrainer_train.py, which wraps the optimizer's
step before handing over to train.py. run_S (2026-10-08, 147
images) settled at 7.3e-04 and blew up to noise at epoch 11 of 33; run_H's
5.8e-04 overshot late; four healthy runs sat at 2.9e-04..3.4e-04. The cap sits
just above the healthy band. UNPROVEN until a capped run ships a working file.

Only the fields in PATCHED_FIELDS ever change per run (a dotted name is a nested
field); everything else passes through untouched. Any change to a learning knob
needs a generated LoRA whose lora_B magnitude is in the 1e-2 range before it is
called proven.
"""
import json
import math
from pathlib import Path

TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "assets" / "krea2_preset.json"
EXPECTED_TEMPLATE_VERSION = 11
PATCHED_FIELDS = (
    "concept_file_name",
    "sample_definition_file_name",
    "workspace_dir",
    "cache_dir",
    "output_model_destination",
    "epochs",
    "learning_rate_warmup_steps",
    "optimizer.prodigy_steps",
    "sample_after",
    "save_every",
)
# The run sets these (folders, output file, step-derived schedule), so the advanced
# settings editor shows them but never overrides them.
LOCKED_PATHS = PATCHED_FIELDS

# Previews every tenth of the run, rounded up to the next whole epoch.
SAMPLE_FRACTION = 0.1


def sample_every_epochs(epochs: int) -> int:
    """Epochs between preview passes: SAMPLE_FRACTION of the run, rounded up, >= 1."""
    return max(1, math.ceil(max(1, int(epochs or 0)) * SAMPLE_FRACTION))


# Ceiling on Prodigy's self-chosen step size `d` (see the module docstring).
PRODIGY_D_CAP = 4.0e-4


class TemplateError(RuntimeError):
    """The vendored preset is missing or from a different OneTrainer config era."""


def load_template() -> dict:
    try:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise TemplateError(f"Cannot read {TEMPLATE_PATH.name}: {e}") from e
    ver = data.get("__version")
    if ver != EXPECTED_TEMPLATE_VERSION:
        raise TemplateError(
            f"{TEMPLATE_PATH.name} is config version {ver!r}, expected "
            f"{EXPECTED_TEMPLATE_VERSION}. Re-export the preset from OneTrainer and "
            "update EXPECTED_TEMPLATE_VERSION in the same commit.")
    return data


def build_run_config(template: dict, run_dir: str, lora_name: str, epochs: int,
                     warmup_steps: int, prodigy_steps: int) -> dict:
    """A deep-copied run config with exactly PATCHED_FIELDS rewritten, all paths
    absolute under `run_dir`. Pure — writes nothing.

    `warmup_steps` and `prodigy_steps` are required rather than defaulted: both are
    sized from the run's real step count (core/engines/onetrainer/steps.plan). The preset's
    200-step warmup is sized for the ~1880-step reference run, so a caller that
    forgets it would hand a short run a warmup it never finishes; a forgotten
    prodigy_steps would let Prodigy re-estimate to the last step.
    """
    run = Path(run_dir)
    cfg = json.loads(json.dumps(template))
    cfg["concept_file_name"] = str(run / "concepts.json")
    cfg["sample_definition_file_name"] = str(run / "samples.json")
    cfg["workspace_dir"] = str(run / "workspace")
    cfg["cache_dir"] = str(run / "cache")
    cfg["output_model_destination"] = str(run / f"{lora_name}.safetensors")
    cfg["epochs"] = int(epochs)
    cfg["learning_rate_warmup_steps"] = float(warmup_steps)
    cfg["optimizer"]["prodigy_steps"] = int(prodigy_steps)
    cfg["sample_after"] = sample_every_epochs(epochs)
    cfg["save_every"] = sample_every_epochs(epochs)
    return cfg
