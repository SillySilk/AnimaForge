"""The post-run artifact check: does the .safetensors we just spent an hour on
actually contain a LoRA the image tool will apply?

Both failure modes here are real ones, taken from Bev_and_Billy (2026-08-17):
a run that completed, saved, and produced a file that changed nothing.
"""
import struct

from core.engines.onetrainer import verify
from lora_fixture import DEAD_B, write_lora as _write_lora


# ---- the artifact is good ----

def test_a_trained_lora_in_the_readable_namespace_passes(tmp_path):
    out = _write_lora(tmp_path / "good.safetensors")
    assert verify.verify_output(str(out)) is None


# ---- the artifact never learned ----

def test_a_lora_whose_b_side_never_left_zero_is_rejected(tmp_path):
    """lora_B is zero-initialised, so its magnitude IS the training signal.
    1.7e-08 is what Bev_and_Billy actually shipped, against ~4e-02 working."""
    out = _write_lora(tmp_path / "dead.safetensors", b_value=DEAD_B)
    reason = verify.verify_output(str(out))
    assert reason is not None
    assert "did not learn" in reason.lower()


def test_the_rejection_message_quotes_both_magnitudes(tmp_path):
    """The number is the whole diagnosis -- a bare "failed" would send the
    maintainer back to guessing."""
    out = _write_lora(tmp_path / "dead.safetensors", b_value=DEAD_B)
    reason = verify.verify_output(str(out))
    assert "e-0" in reason                       # the measured peak
    assert str(verify.MIN_TRAINED_ABSMAX) in reason or "1e-03" in reason


def test_an_untrained_b_side_is_caught_even_when_a_side_looks_normal(tmp_path):
    """lora_A keeps its random init whatever happens, so a healthy-looking A
    must not rescue the verdict."""
    out = _write_lora(tmp_path / "dead.safetensors", a_value=4.7e-2, b_value=DEAD_B)
    assert verify.verify_output(str(out)) is not None


def test_the_floor_is_a_boundary_not_a_cliff_either_side_of_it(tmp_path):
    """Straddle MIN_TRAINED_ABSMAX rather than sitting on it: 1e-3 has no exact
    BF16 pattern and the fixture's cast truncates toward zero, so writing the
    floor itself stores a hair under it. A weakly-trained LoRA is still a real
    one, so a value clearly above must pass."""
    floor = verify.MIN_TRAINED_ABSMAX
    above = _write_lora(tmp_path / "above.safetensors", b_value=floor * 4)
    assert verify.verify_output(str(above)) is None
    below = _write_lora(tmp_path / "below.safetensors", b_value=floor / 4)
    assert verify.verify_output(str(below)) is not None


# ---- the artifact will not load ----

def test_the_diffusers_namespace_is_rejected(tmp_path):
    """DIFFUSERS_LORA names (transformer.*) share zero keys with the
    diffusion_model.* names the image tool matches on, so it applies nothing
    and says nothing."""
    out = _write_lora(tmp_path / "wrongns.safetensors", prefix="transformer")
    reason = verify.verify_output(str(out))
    assert reason is not None
    assert "diffusion_model" in reason


# ---- the artifact is not a LoRA at all ----

def test_a_file_with_no_lora_tensors_is_rejected(tmp_path):
    out = _write_lora(tmp_path / "nob.safetensors", with_b=False)
    reason = verify.verify_output(str(out))
    assert reason is not None
    assert "lora_b" in reason.lower()


def test_an_unparseable_file_is_rejected_not_passed(tmp_path):
    """A validator that shrugs at bytes it cannot read is worse than none."""
    out = tmp_path / "junk.safetensors"
    out.write_bytes(b"x")
    assert verify.verify_output(str(out)) is not None


def test_a_missing_file_is_rejected(tmp_path):
    assert verify.verify_output(str(tmp_path / "nope.safetensors")) is not None


def test_a_header_longer_than_the_file_is_rejected(tmp_path):
    out = tmp_path / "lying.safetensors"
    out.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    assert verify.verify_output(str(out)) is not None


# ---- the measurement itself ----

def test_peak_abs_reads_magnitude_across_four_orders_of_magnitude(tmp_path):
    """Magnitude ordering of IEEE-754 floats is monotonic in the sign-cleared bit
    pattern, so the peak is found by integer comparison and converted once. The
    only slack here is BF16's own 8-bit mantissa, which the fixture's truncating
    cast rounds toward zero."""
    for value in (4.0e-2, DEAD_B, 1.0, 3.3e-4):
        out = _write_lora(tmp_path / "m.safetensors", b_value=value)
        peak = verify.peak_lora_b_abs(str(out))
        assert value / 2 <= peak <= value * 2, f"{value} -> {peak}"


def test_negative_weights_count_as_magnitude(tmp_path):
    """Sign must be masked off, or a LoRA that trained negative reads as dead."""
    out = _write_lora(tmp_path / "neg.safetensors", b_value=-4.0e-2)
    assert verify.verify_output(str(out)) is None


# ---- the run overshot: the file is the state AFTER the loss turned upward ----
#
# Hailey (2026-09-04): 67 epochs, every trainer-side check green, and a file that
# rendered a mosaic of faces in the image tool. The end-of-epoch smooth loss had
# bottomed at epoch 52 (0.0613) and climbed every epoch after, to 0.0844 (+38%).
# The 13 healthy runs before it all finished within 0-8% of their best.

def _tqdm_log(losses_by_epoch: dict, total: int) -> str:
    """A trainer.log shaped like OneTrainer's: a step bar carrying `smooth loss`,
    then the epoch bar that closes the epoch."""
    lines = ["Quantizing model weights: 100%|##########| 595/595"]
    for epoch in sorted(losses_by_epoch):
        loss = losses_by_epoch[epoch]
        lines.append(f"step: 100%|##########| 17/17 [00:57<00:00,  3.37s/it, "
                     f"loss={loss * 1.3:.4f}, smooth loss={loss:.4f}]")
        pct = int(100 * epoch / total)
        lines.append(f"epoch: {pct:3d}%|{'#' * 10}| {epoch}/{total} [1:12:15<00:00, 62.34s/it]")
    return "\n".join(lines) + "\n"


def _curve(total: int, best_epoch: int, best: float, final: float) -> dict:
    """Falls to `best` at best_epoch, then climbs to `final` at the last epoch."""
    out = {}
    for e in range(1, total + 1):
        if e <= best_epoch:
            out[e] = best + (0.09 - best) * (best_epoch - e) / max(1, best_epoch - 1)
        else:
            out[e] = best + (final - best) * (e - best_epoch) / max(1, total - best_epoch)
    return out


def test_loss_by_epoch_reads_the_smooth_loss_that_closes_each_epoch():
    text = _tqdm_log({1: 0.0800, 2: 0.0750, 3: 0.0700}, total=3)
    assert verify.loss_by_epoch(text) == {1: 0.0800, 2: 0.0750, 3: 0.0700}


def test_a_log_without_loss_lines_has_no_curve():
    assert verify.loss_by_epoch("nvCOMP: backend loaded\nSaving x.safetensors\n") == {}


def test_a_run_that_ends_a_third_above_its_best_is_an_overshoot():
    losses = _curve(total=67, best_epoch=52, best=0.0613, final=0.0844)
    found = verify.overshoot(losses, total_epochs=67)
    assert found is not None
    assert found["best_epoch"] == 52
    assert found["final_epoch"] == 67
    assert 0.35 < found["rise"] < 0.40


def test_a_run_that_ends_within_the_healthy_band_is_not_an_overshoot():
    """Riki finished 3% above its best and works; Melanie 2%."""
    losses = _curve(total=55, best_epoch=47, best=0.0887, final=0.0914)
    assert verify.overshoot(losses, total_epochs=55) is None


def test_the_noisy_opening_epochs_do_not_count_as_the_best():
    """The smooth loss starts as an average of a handful of steps and can sit
    low by chance -- bruises had its minimum at epoch 0 and a healthy +22% "rise"
    from there. Only epochs after the first tenth of the run set the best."""
    losses = {e: 0.100 for e in range(1, 34)}
    losses[1] = 0.0844                     # the fluke
    losses[33] = 0.1030
    assert verify.overshoot(losses, total_epochs=33) is None


def test_overshoot_is_read_from_the_run_log(tmp_path):
    log = tmp_path / "trainer.log"
    log.write_text(_tqdm_log(_curve(67, 52, 0.0613, 0.0844), total=67), encoding="utf-8")
    found = verify.overshoot_from_log(str(log))
    assert found is not None and found["best_epoch"] == 52


def test_a_missing_or_empty_log_is_not_an_overshoot(tmp_path):
    assert verify.overshoot_from_log(str(tmp_path / "nope.log")) is None
    (tmp_path / "trainer.log").write_text("", encoding="utf-8")
    assert verify.overshoot_from_log(str(tmp_path / "trainer.log")) is None


def test_the_overshoot_message_says_where_the_best_was_and_how_far_it_fell():
    losses = _curve(total=67, best_epoch=52, best=0.0613, final=0.0844)
    text = verify.describe_overshoot(verify.overshoot(losses, total_epochs=67))
    assert "epoch 52" in text and "67" in text
    assert "38%" in text


# I_spit_on_your_grave_t (2026-10-08): end-of-epoch smooth loss, epochs 1..32 of 33.
_BLOWUP = [0.129, 0.136, 0.133, 0.134, 0.135, 0.145, 0.135, 0.137, 0.138, 0.129,
           0.215, 0.263, 0.238, 0.221, 0.207, 0.205, 0.206, 0.187, 0.171, 0.167,
           0.162, 0.156, 0.160, 0.162, 0.153, 0.150, 0.152, 0.155, 0.144, 0.148,
           0.143, 0.145]


def _blowup_lines(losses=_BLOWUP, total=33):
    out = []
    for e, loss in enumerate(losses, start=1):
        out.append(f"step: 100%|##########| 72/72 [06:18<00:00, loss={loss}, smooth loss={loss}]")
        out.append(f"epoch: {int(100 * e / total):3d}%|###| {e}/{total} [1:00:00<00:00, 298.93s/it]")
    return out


def test_a_cliff_in_the_loss_is_a_blowup_even_when_the_end_recovers():
    losses = {e: l for e, l in enumerate(_BLOWUP, start=1)}
    assert verify.overshoot(losses, 33) is None          # what let it ship
    found = verify.divergence(losses, 33)
    assert found["kind"] == "diverged"
    assert found["spike_epoch"] == 11 and found["last_good_epoch"] == 10
    assert found["best_epoch"] == 10 and abs(found["rise"] - 0.667) < 0.01


def test_a_healthy_wobble_is_not_a_blowup():
    losses = {e: l for e, l in enumerate(_BLOWUP[:10], start=1)}   # up to +12%
    assert verify.divergence(losses, 33) is None


def test_the_watch_trips_on_the_epoch_that_jumped_and_only_once():
    watch = verify.DivergenceWatch()
    tripped = [(i, f) for i, line in enumerate(_blowup_lines()) if (f := watch.feed(line))]
    assert len(tripped) == 1
    assert tripped[0][1]["spike_epoch"] == 11


def test_a_blowup_in_the_log_outranks_a_healthy_end(tmp_path):
    log = tmp_path / "trainer.log"
    log.write_text("\n".join(_blowup_lines()), encoding="utf-8")
    found = verify.trouble_from_log(str(log))
    assert found["kind"] == "diverged"
    assert "blew up" in verify.describe(found) and "epoch 11" in verify.describe(found)
