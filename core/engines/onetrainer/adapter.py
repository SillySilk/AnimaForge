"""Translate a queued RunDefinition into the OneTrainer session's RunParams."""
from core import added_sets, models
from core.engines.onetrainer.session import RunParams


def to_onetrainer_params(run) -> RunParams:
    if models.get(run.model).engine != models.ONETRAINER:
        raise ValueError(f"model {run.model!r} is not trained by OneTrainer")
    added = added_sets.folders(run.dataset_folder, run.added_folders)
    weight = added_sets.emphasis() if added else 1.0
    return RunParams(
        lora_name=run.lora_name,
        dataset_dir=run.dataset_folder,
        image_count=run.image_count,
        concept=run.subject_type or "character",
        trigger_word=run.trigger_word,
        output_base=run.output_dir,
        onetrainer_root=run.onetrainer_root,
        quality_prefix=run.quality_prefix,
        target_steps=run.target_steps,
        config_overrides=dict(run.config_overrides),
        debug_profiles=bool(run.debug_profiles),
        sample_prompts=list(run.sample_prompts or []),
        added_concepts=[(path, weight) for path, _n in added],
        plan_image_count=(added_sets.weighted_count(run.image_count, run.dataset_folder,
                                                    run.added_folders, weight)
                          if added else 0),
    )
