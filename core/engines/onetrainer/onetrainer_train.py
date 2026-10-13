"""Run OneTrainer's scripts/train.py with a ceiling on Prodigy's step size.

    python onetrainer_train.py <path to scripts/train.py> <d cap> --config-path <cfg>

Krea Forge runs this under OneTrainer's interpreter, from OneTrainer's root, in
place of train.py itself. It is never imported by the app, and it imports nothing
from the app.

Why: Prodigy picks its own step size (`d`) and never lowers it. On every healthy
run it settled at 2.9e-04..3.4e-04. run_H (2026-09-04) settled at 5.8e-04 and
overshot in its last 15 epochs; run_S (2026-10-08) settled at
7.3e-04 and blew up to noise at epoch 11 of 33. OneTrainer's plain PRODIGY has no
ceiling option -- d_coef scales every run, healthy ones too, and growth_rate only
limits how fast d climbs, not how far -- so the ceiling is applied here: after
every optimizer step, `d` (and the `d_max` it grows toward) are clamped to the
cap. The next step's learning rate is d * lr, read from the clamped value, so no
update ever uses more than the cap. A run that settles below the cap is untouched.
"""
import os
import re
import runpy
import sys
from pathlib import Path

_announced = False

# OneTrainer files each prompt's previews under samples/"<i> - <prompt start>"/.
_PROMPT_DIR = re.compile(r"^(\d+) - ")


def flat_destination(destination: str) -> str:
    """Where a training preview goes instead: loose in samples/, with its prompt
    index carried in the name ("<OneTrainer name>-p<i>") since the folder that
    used to say it is gone. Anything not under a prompt folder is left alone."""
    path = Path(destination)
    m = _PROMPT_DIR.match(path.parent.name)
    if not m or path.parent.parent.name != "samples":
        return destination
    return str(path.parent.parent / f"{path.name}-p{m.group(1)}")


def install_flat_samples(onetrainer_root: str) -> bool:
    """Wrap BaseModelSampler.save_sampler_output so training previews land loose in
    samples/ rather than one folder per prompt. False when the sampler module is not
    where this OneTrainer keeps it (previews then keep OneTrainer's own folders)."""
    sys.path.insert(0, onetrainer_root)
    try:
        from modules.modelSampler.BaseModelSampler import BaseModelSampler
    except ImportError:
        print("[animaforge] OneTrainer's sampler was not found; previews keep their "
              "per-prompt folders", flush=True)
        return False
    original = BaseModelSampler.save_sampler_output

    def save_sampler_output(sampler_output, destination, *args, **kwargs):
        return original(sampler_output, flat_destination(destination), *args, **kwargs)

    BaseModelSampler.save_sampler_output = staticmethod(save_sampler_output)
    return True


def _clamp(optimizer, cap: float):
    global _announced
    for group in optimizer.param_groups:
        d = float(group.get("d", 0.0))
        if d > cap:
            if not _announced:
                print(f"[animaforge] Prodigy step size reached {d:.3e}; capped at "
                      f"{cap:.3e}", flush=True)
                _announced = True
            group["d"] = cap
        if float(group.get("d_max", 0.0)) > cap:
            group["d_max"] = cap


def install_cap(cap: float) -> bool:
    """Wrap prodigyopt.Prodigy.step so `d` never exceeds `cap`. False when the
    optimizer is not installed in this interpreter (there is then no Prodigy run
    to cap)."""
    try:
        import prodigyopt
    except ImportError:
        print("[animaforge] prodigyopt is not installed here; no step-size cap applied",
              flush=True)
        return False
    original = prodigyopt.Prodigy.step

    def step(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        _clamp(self, cap)
        return result

    prodigyopt.Prodigy.step = step
    print(f"[animaforge] Prodigy step size capped at {cap:.3e}", flush=True)
    return True


def _optimizer_name(argv) -> str:
    """The optimizer named in the --config-path file, or "" if it can't be read."""
    import json
    try:
        path = argv[argv.index("--config-path") + 1]
        with open(path, encoding="utf-8") as f:
            return str(json.load(f).get("optimizer", {}).get("optimizer", ""))
    except (ValueError, IndexError, OSError, AttributeError, json.JSONDecodeError):
        return ""


def main() -> None:
    train_script, cap = sys.argv[1], float(sys.argv[2])
    # Only Prodigy picks its own step size; a fixed-rate optimizer has nothing to cap,
    # and announcing a cap there would misdescribe the run.
    if _optimizer_name(sys.argv[3:]) == "PRODIGY":
        install_cap(cap)
    # Launched from OneTrainer's root (process.py sets the cwd), which holds modules/.
    install_flat_samples(os.getcwd())
    # train.py reads its own arguments and imports its siblings (util.import_util)
    # relative to its folder, exactly as if it had been launched directly.
    sys.argv = [train_script, *sys.argv[3:]]
    sys.path.insert(0, str(Path(train_script).resolve().parent))
    runpy.run_path(train_script, run_name="__main__")


if __name__ == "__main__":
    main()
