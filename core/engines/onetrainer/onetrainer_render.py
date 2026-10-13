"""Render a finished LoRA's preview prompts with OneTrainer's own loader and
sampler. Krea Forge runs this under OneTrainer's interpreter, with the current
directory at OneTrainer's root, the same way it runs scripts/train.py. It is
never imported by the app, and it imports nothing from the app.

    python onetrainer_render.py --config-path <run>/config.json
                                --lora <run>/<name>.safetensors
                                --destination <run>/workspace/samples

The run config says which base model, which weight types and which samples
file; the LoRA is applied through OneTrainer's LoRA loader (the COMFY-format
file it wrote is one of the formats that loader reads back); each enabled
sample in samples.json is rendered once, with the same sampler the trainer
uses for its own previews. Model setup mirrors GenericTrainer.start(), minus
the data loader: optimizations (quantization), train-device placement, then
the LoRA wrap that loads the state dict.

Exit 0 with at least one preview written, 2 when none could be rendered. A
traceback on stderr is the record; the app keeps it in trainer.log.
"""
import argparse
import json
import os
import sys
import traceback

# The script's own folder is sys.path[0]; OneTrainer's modules live in the
# current directory, which is its root.
sys.path.insert(0, os.getcwd())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True, dest="config_path")
    parser.add_argument("--lora", required=True, dest="lora")
    parser.add_argument("--destination", required=True, dest="destination")
    args = parser.parse_args()

    import torch
    from modules.util import create
    from modules.util.config.SampleConfig import SampleConfig
    from modules.util.config.TrainConfig import TrainConfig

    config = TrainConfig.default_values()
    with open(args.config_path, "r", encoding="utf-8") as f:
        config.from_dict(json.load(f), migrate=True)
    config.lora_model_name = args.lora
    config.continue_last_backup = False

    train_device = torch.device(config.train_device)
    temp_device = torch.device(config.temp_device)
    if config.quantization.cache_dir is None:
        config.quantization.cache_dir = config.cache_dir + "/quantization"
    os.makedirs(config.quantization.cache_dir, exist_ok=True)

    print(f"[animaforge] loading {config.base_model_name} with {os.path.basename(args.lora)}",
          flush=True)
    model_loader = create.create_model_loader(config.model_type, config.training_method)
    model_setup = create.create_model_setup(
        config.model_type, train_device, temp_device, config.training_method, config.debug_mode)
    model = model_loader.load(
        model_type=config.model_type,
        model_names=config.model_names(),
        weight_dtypes=config.weight_dtypes(),
        quantization=config.quantization,
    )
    model.train_config = config
    model_setup.setup_optimizations(model, config)
    model_setup.setup_train_device(model, config)
    model_setup.setup_model(model, config)
    model.eval()

    sampler = create.create_model_sampler(
        train_device, temp_device, model, config.model_type, config.training_method)

    with open(config.sample_definition_file_name, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    written = 0
    for i, raw in enumerate(raw_samples):
        sample = SampleConfig.default_values(config.model_type).from_dict(raw)
        if not sample.enabled:
            continue
        sample.from_train_config(config)
        dest = os.path.join(args.destination, f"final-{i}")
        try:
            sampler.sample(
                sample_config=sample,
                destination=dest,
                image_format=config.sample_image_format,
                video_format=config.sample_video_format,
                audio_format=config.sample_audio_format,
            )
            written += 1
            print(f"[animaforge] final preview {i} written", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"[animaforge] final preview {i} failed, continuing", flush=True)

    print(f"[animaforge] final previews written: {written}", flush=True)
    return 0 if written else 2


if __name__ == "__main__":
    sys.exit(main())
