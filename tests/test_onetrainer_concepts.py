from core.engines.onetrainer.concepts import build_concepts, build_samples, SAMPLE_COUNT

def test_concept_points_at_dataset_and_stays_flat():
    (c,) = build_concepts(r"C:\datasets\aria")
    assert c["path"] == r"C:\datasets\aria"
    assert c["name"] == "aria"
    assert c["enabled"] is True
    assert c["include_subdirectories"] is False
    assert c["type"] == "STANDARD"
    assert c["__version"] == 2

def test_concept_augmentation_matches_proven_run():
    (c,) = build_concepts(r"C:\d")
    assert c["image"]["enable_crop_jitter"] is True
    assert c["image"]["enable_random_flip"] is False
    assert c["text"]["prompt_source"] == "sample"


# ---- sample prompts come from the captions the run is training on ----

def _dataset(tmp_path, captions):
    """A dataset folder of .txt caption sidecars, the file OneTrainer reads."""
    for i, text in enumerate(captions):
        (tmp_path / f"img{i:02d}.txt").write_text(text, encoding="utf-8")
    return str(tmp_path)


def test_sample_prompts_are_drawn_from_the_captions(tmp_path):
    """A preview is only worth rendering if it previews what was taught. A generic
    scene ("a portrait photograph of a person") tells you nothing about a set of
    two-character comic panels — which is exactly what run_B sampled."""
    captions = [f"aria_x, Aria is standing in scene {i}, looking at the camera."
                for i in range(6)]
    samples = build_samples(_dataset(tmp_path, captions), "aria_x")
    assert samples
    for s in samples:
        assert s["prompt"] in captions


def test_samples_spread_across_the_set_rather_than_repeating_one(tmp_path):
    captions = [f"aria_x, Aria in scene number {i}, described at length." for i in range(9)]
    samples = build_samples(_dataset(tmp_path, captions), "aria_x")
    prompts = [s["prompt"] for s in samples]
    assert len(prompts) == SAMPLE_COUNT
    assert len(set(prompts)) == SAMPLE_COUNT


def test_only_captions_carrying_the_trigger_are_previewed(tmp_path):
    """A caption without the trigger was never teaching the trigger, so previewing
    it measures nothing about whether the LoRA answers to it."""
    captions = ["aria_x, Aria sits by the window.",
                "a landscape with no subject at all.",
                "aria_x, Aria stands in the doorway.",
                "another stray caption with no trigger."]
    samples = build_samples(_dataset(tmp_path, captions), "aria_x")
    assert samples
    assert all("aria_x" in s["prompt"] for s in samples)


def test_the_trigger_matches_on_word_boundaries_like_preflight_does(tmp_path):
    """engine._trigger_caption_problem uses \\b; disagreeing with it would let a run
    start and then preview captions it considered untriggered."""
    captions = ["aria_xtra, not the same token at all.",
                "aria_x, Aria is the real subject here."]
    samples = build_samples(_dataset(tmp_path, captions), "aria_x")
    assert [s["prompt"] for s in samples] == [captions[1]]


def test_the_selection_is_deterministic(tmp_path):
    """Samples are the run's progress signal — if the prompt set moved between
    restarts you could not compare one interval to the next."""
    captions = [f"aria_x, Aria in scene {i}." for i in range(12)]
    d = _dataset(tmp_path, captions)
    assert build_samples(d, "aria_x") == build_samples(d, "aria_x")


def test_each_sample_holds_a_fixed_seed_of_its_own(tmp_path):
    """Fixed so intervals are comparable, distinct so the previews are not near
    duplicates of each other."""
    captions = [f"aria_x, Aria in scene {i}." for i in range(6)]
    samples = build_samples(_dataset(tmp_path, captions), "aria_x")
    seeds = [s["seed"] for s in samples]
    assert len(set(seeds)) == len(seeds)
    assert all(s["random_seed"] is False for s in samples)


def test_sample_geometry_matches_the_training_resolution(tmp_path):
    samples = build_samples(_dataset(tmp_path, ["aria_x, Aria stands here."]), "aria_x")
    for s in samples:
        assert s["height"] == 512 and s["width"] == 512
        assert s["enabled"] is True
        assert s["frames"] == 1
        assert s["negative_prompt"] == ""


def test_blank_captions_are_not_previewed(tmp_path):
    captions = ["   \n  ", "aria_x, Aria is here.", ""]
    samples = build_samples(_dataset(tmp_path, captions), "aria_x")
    assert [s["prompt"] for s in samples] == ["aria_x, Aria is here."]


def test_fewer_captions_than_requested_yields_only_what_exists(tmp_path):
    samples = build_samples(_dataset(tmp_path, ["aria_x, only one."]), "aria_x")
    assert len(samples) == 1


def test_a_set_with_no_triggered_caption_previews_nothing(tmp_path):
    """engine.preflight refuses this run outright, so reaching here means something
    upstream changed. Inventing a prompt would fake a preview of an untrained
    token; an empty list simply asks OneTrainer for no samples."""
    samples = build_samples(_dataset(tmp_path, ["a caption with no trigger."]), "aria_x")
    assert samples == []


def test_an_empty_folder_previews_nothing(tmp_path):
    assert build_samples(str(tmp_path), "aria_x") == []


def test_a_missing_folder_previews_nothing(tmp_path):
    assert build_samples(str(tmp_path / "nope"), "aria_x") == []


def test_an_overlong_caption_is_cut_at_a_sentence_end(tmp_path):
    """Krea 2's text encoder window is 512 tokens. A caption past that would be
    silently clipped mid-word by the tokenizer; cutting on a sentence keeps the
    prompt readable and keeps the trigger, which leads the caption."""
    long_caption = "aria_x, " + ("Aria stands in a detailed room. " * 200)
    samples = build_samples(_dataset(tmp_path, [long_caption]), "aria_x")
    (prompt,) = [s["prompt"] for s in samples]
    assert prompt.startswith("aria_x, ")
    assert len(prompt) < len(long_caption)
    assert prompt.endswith(".")


def test_a_caption_with_no_trigger_is_used_when_no_trigger_is_set(tmp_path):
    """No trigger means nothing to filter on, so the captions still drive the
    preview rather than a hardcoded scene."""
    samples = build_samples(_dataset(tmp_path, ["A plain scene description."]), "")
    assert [s["prompt"] for s in samples] == ["A plain scene description."]
