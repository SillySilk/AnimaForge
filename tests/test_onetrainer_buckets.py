"""Aspect-bucket accounting: which images will never be trained.

OneTrainer groups images into aspect buckets, then drops len(bucket) % batch_size
from each one every epoch (mgds AspectBatchSorting.__shuffle). The drop rotates,
so an odd bucket is harmless — but a bucket holding fewer images than the batch
size yields no batches at all, and every image in it sits out FOREVER.

run B (2026-08-17) lost 2 of 47 that way, silently, for a whole 57-epoch
run: A19 at aspect 1.600 and A35 at 0.687 were each alone in a bucket.
"""
from PIL import Image

from core.engines.onetrainer import buckets


def _img(folder, name, w, h):
    Image.new("RGB", (w, h), (128, 128, 128)).save(folder / name, format="PNG")
    return name


def test_an_image_alone_in_its_bucket_is_stranded(tmp_path):
    for i in range(4):
        _img(tmp_path, f"sq{i}.png", 512, 512)
    _img(tmp_path, "tall.png", 820, 1312)          # aspect 1.600, nothing near it
    plan = buckets.bucket_plan(str(tmp_path))
    assert plan["stranded"] == ["tall.png"]


def test_a_pair_at_the_same_aspect_is_not_stranded(tmp_path):
    for i in range(4):
        _img(tmp_path, f"sq{i}.png", 512, 512)
    _img(tmp_path, "tall1.png", 820, 1312)
    _img(tmp_path, "tall2.png", 820, 1312)         # now the bucket can form a batch
    plan = buckets.bucket_plan(str(tmp_path))
    assert plan["stranded"] == []


def test_an_odd_bucket_is_not_stranded_because_the_drop_rotates(tmp_path):
    """3 in a bucket loses 1 per epoch, but AspectBatchSorting reshuffles per epoch
    (current_variation = current_epoch), so a different one sits out each time and
    all three train most epochs. Only a bucket too small to batch is a real loss."""
    for i in range(3):
        _img(tmp_path, f"tall{i}.png", 820, 1312)
    plan = buckets.bucket_plan(str(tmp_path))
    assert plan["stranded"] == []
    assert plan["dropped_per_epoch"] == 1


def test_steps_per_epoch_counts_whole_batches_per_bucket(tmp_path):
    for i in range(4):
        _img(tmp_path, f"sq{i}.png", 512, 512)     # 4 -> 2 steps
    for i in range(3):
        _img(tmp_path, f"tall{i}.png", 820, 1312)  # 3 -> 1 step, 1 dropped
    plan = buckets.bucket_plan(str(tmp_path))
    assert plan["steps_per_epoch"] == 3
    assert plan["dropped_per_epoch"] == 1
    assert plan["total"] == 7


def test_a_stranded_image_is_not_counted_as_trained(tmp_path):
    for i in range(2):
        _img(tmp_path, f"sq{i}.png", 512, 512)
    _img(tmp_path, "tall.png", 820, 1312)
    plan = buckets.bucket_plan(str(tmp_path))
    assert plan["total"] == 3
    assert plan["used"] == 2


def test_an_empty_folder_plans_nothing(tmp_path):
    plan = buckets.bucket_plan(str(tmp_path))
    assert plan["stranded"] == [] and plan["steps_per_epoch"] == 0


def test_a_missing_folder_plans_nothing(tmp_path):
    plan = buckets.bucket_plan(str(tmp_path / "nope"))
    assert plan["stranded"] == []


def test_an_unreadable_file_is_skipped_not_fatal(tmp_path):
    for i in range(2):
        _img(tmp_path, f"sq{i}.png", 512, 512)
    (tmp_path / "broken.png").write_bytes(b"not a png")
    plan = buckets.bucket_plan(str(tmp_path))
    assert plan["total"] == 2                       # counted what it could read


# ---- the user-facing warning ----

def test_crop_report_names_every_fixed_file(tmp_path):
    for i in range(2):
        _img(tmp_path, f"sq{i}.png", 512, 512)
    _img(tmp_path, "tall.png", 820, 1312)
    _img(tmp_path, "wide.png", 1486, 1021)
    report = buckets.crop_report(str(tmp_path))
    assert "tall.png" in report and "wide.png" in report
    assert "cropped" in report.lower()


def test_a_healthy_set_produces_no_report(tmp_path):
    for i in range(4):
        _img(tmp_path, f"sq{i}.png", 512, 512)
    assert buckets.crop_report(str(tmp_path)) == ""


def test_crop_report_still_names_an_uncroppable_remainder(tmp_path):
    # Alone in the whole dataset: no healthy bucket exists to crop toward.
    _img(tmp_path, "only.png", 820, 1312)
    report = buckets.crop_report(str(tmp_path)).lower()
    assert "only.png" in report
    assert "never" in report


def test_the_bev_and_billy_shape_is_pinned(tmp_path):
    """Regression on the real failure. These are the actual dimensions of the set
    that trained 2026-08-17: 38 near-square, a landscape cluster, a portrait
    cluster, plus A19 (820x1312) and A35 (1486x1021) alone at their shapes.
    trainer.log recorded 22 steps per epoch, which is what this must reproduce."""
    for i in range(38):
        _img(tmp_path, f"sq{i:02d}.png", 1024, 1024)      # the 512x512 bucket
    for i, (w, h) in enumerate([(1320, 1023), (1213, 951), (1230, 970), (1200, 940)]):
        _img(tmp_path, f"wide{i}.png", w, h)              # the 576x448 bucket
    for i, (w, h) in enumerate([(760, 1010), (949, 1201), (1016, 1163)]):
        _img(tmp_path, f"port{i}.png", w, h)              # the 448x576 bucket
    _img(tmp_path, "A19 copy.png", 820, 1312)             # alone at aspect 1.600
    _img(tmp_path, "A35 copy.png", 1486, 1021)            # alone at aspect 0.687

    plan = buckets.bucket_plan(str(tmp_path))
    assert plan["total"] == 47
    assert plan["steps_per_epoch"] == 22                  # matches trainer.log
    assert plan["stranded"] == ["A19 copy.png", "A35 copy.png"]
    assert plan["used"] == 44


def test_cropping_the_strays_into_a_populated_bucket_clears_them(tmp_path):
    """The fix that was applied: A19 to aspect 1.286, A35 to 0.778."""
    for i in range(38):
        _img(tmp_path, f"sq{i:02d}.png", 1024, 1024)
    for i, (w, h) in enumerate([(1320, 1023), (1213, 951), (1230, 970), (1200, 940)]):
        _img(tmp_path, f"wide{i}.png", w, h)
    for i, (w, h) in enumerate([(760, 1010), (949, 1201), (1016, 1163)]):
        _img(tmp_path, f"port{i}.png", w, h)
    _img(tmp_path, "A19 copy.png", 820, 1054)             # cropped 1.600 -> 1.285
    _img(tmp_path, "A35 copy.png", 1313, 1021)            # cropped 0.687 -> 0.778

    plan = buckets.bucket_plan(str(tmp_path))
    assert plan["stranded"] == []
    assert plan["steps_per_epoch"] == 23
    assert plan["dropped_per_epoch"] == 1                 # rotates, so harmless


# ---- crop_stranded: doing the crop instead of just reporting it ----

def test_crop_stranded_moves_a_stray_into_a_healthy_bucket(tmp_path):
    for i in range(4):
        _img(tmp_path, f"sq{i}.png", 512, 512)
    _img(tmp_path, "tall.png", 820, 1312)          # aspect 1.600, stranded alone
    fixed = buckets.crop_stranded(str(tmp_path))
    assert [f["name"] for f in fixed] == ["tall.png"]
    assert buckets.bucket_plan(str(tmp_path))["stranded"] == []


def test_crop_stranded_shrinks_height_when_the_stray_is_too_tall(tmp_path):
    for i in range(4):
        _img(tmp_path, f"sq{i}.png", 512, 512)     # nearest healthy bucket: square
    _img(tmp_path, "tall.png", 820, 1312)
    buckets.crop_stranded(str(tmp_path))
    with Image.open(tmp_path / "tall.png") as im:
        w, h = im.size
    assert w == 820             # untouched -- the image was too tall, not too wide
    assert h < 1312              # only the long side gave anything up


def test_crop_stranded_shrinks_width_when_the_stray_is_too_wide(tmp_path):
    for i in range(4):
        _img(tmp_path, f"sq{i}.png", 512, 512)
    _img(tmp_path, "wide.png", 1312, 820)          # mirror of the tall case, aspect 0.625
    buckets.crop_stranded(str(tmp_path))
    with Image.open(tmp_path / "wide.png") as im:
        w, h = im.size
    assert h == 820
    assert w < 1312


def test_crop_stranded_leaves_a_healthy_set_untouched(tmp_path):
    for i in range(4):
        _img(tmp_path, f"sq{i}.png", 512, 512)
    assert buckets.crop_stranded(str(tmp_path)) == []


def test_crop_stranded_does_nothing_without_a_healthy_bucket_to_aim_at(tmp_path):
    _img(tmp_path, "only.png", 820, 1312)          # alone in the whole dataset
    assert buckets.crop_stranded(str(tmp_path)) == []
    with Image.open(tmp_path / "only.png") as im:
        assert im.size == (820, 1312)              # untouched -- nothing to crop toward


def test_crop_stranded_clears_the_bev_and_billy_regression(tmp_path):
    """The real failure, fixed automatically instead of by hand."""
    for i in range(38):
        _img(tmp_path, f"sq{i:02d}.png", 1024, 1024)
    for i, (w, h) in enumerate([(1320, 1023), (1213, 951), (1230, 970), (1200, 940)]):
        _img(tmp_path, f"wide{i}.png", w, h)
    for i, (w, h) in enumerate([(760, 1010), (949, 1201), (1016, 1163)]):
        _img(tmp_path, f"port{i}.png", w, h)
    _img(tmp_path, "A19 copy.png", 820, 1312)
    _img(tmp_path, "A35 copy.png", 1486, 1021)

    fixed = buckets.crop_stranded(str(tmp_path))
    assert {f["name"] for f in fixed} == {"A19 copy.png", "A35 copy.png"}

    plan = buckets.bucket_plan(str(tmp_path))
    assert plan["stranded"] == []
    assert plan["total"] == 47
