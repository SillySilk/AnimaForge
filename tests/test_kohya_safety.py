from lora_fixture import DEAD_B, write_lora

from core.engines.kohya import safety


def _epoch_lines(losses, total=10):
    """sd-scripts' shape: an 'epoch N/M' line opens each epoch, tqdm lines carry avr_loss."""
    lines = []
    for i, loss in enumerate(losses, start=1):
        lines += ["", f"epoch {i}/{total}", ""]
        lines.append(f"steps:  10%|█  | {i * 12}/120 [00:30<04:30,  2.50s/it, avr_loss=0.5]")
        lines.append(f"steps:  10%|█  | {i * 12 + 1}/120 [00:31<04:29,  2.50s/it, avr_loss={loss}]")
    return lines


def test_loss_by_epoch_reads_kohya_lines():
    lines = _epoch_lines([0.11, 0.10, 0.09])
    assert safety.loss_by_epoch(lines) == {1: 0.11, 2: 0.10}
    assert safety.loss_by_epoch(lines, finished=True) == {1: 0.11, 2: 0.10, 3: 0.09}


def test_watch_fires_once_on_the_spike_epoch():
    lines = _epoch_lines([0.11, 0.10, 0.10, 0.10, 0.25, 0.10, 0.10], total=10)
    w = safety.LossWatch()
    hits = [(i, f) for i, f in enumerate(w.feed(l) for l in lines) if f]
    assert len(hits) == 1
    found = hits[0][1]
    assert found["spike_epoch"] == 5 and found["last_good_epoch"] == 4
    assert lines[hits[0][0]] == "epoch 6/10"


HEALTHY = {
    "Barry": [0.1001, 0.0921, 0.0857, 0.0791, 0.0822, 0.0824, 0.0780, 0.0736, 0.0789, 0.0763,
              0.0747, 0.0759, 0.0689, 0.0744, 0.0676, 0.0744, 0.0655, 0.0681, 0.0699, 0.0653],
    "Funnel_Gag": [0.1066, 0.1028, 0.1096, 0.1088, 0.1120, 0.1110, 0.1143, 0.1071, 0.1162, 0.1066,
                   0.0924, 0.1073, 0.1161, 0.1088, 0.1098, 0.1010, 0.1011, 0.0977, 0.0978, 0.1090,
                   0.1027, 0.1005, 0.0890, 0.0988, 0.1102, 0.1055, 0.1089, 0.1037, 0.0998, 0.1092],
    "RebelGirlsFlag": [0.1020, 0.1024, 0.0957, 0.1042, 0.1120, 0.1007, 0.1049, 0.1090, 0.0992,
                       0.1053, 0.1005, 0.0999, 0.1151, 0.1068, 0.1121, 0.0975, 0.1101, 0.1056,
                       0.1105, 0.1120, 0.1008, 0.1095],
    "RickMortyCore": [0.0677, 0.0680, 0.0667, 0.0672, 0.0670, 0.0707, 0.0641, 0.0628, 0.0640,
                      0.0605, 0.0658, 0.0647, 0.0621, 0.0648],
    "RickMortyCore2": [0.0677, 0.0668, 0.0685, 0.0660, 0.0658, 0.0672, 0.0666, 0.0680, 0.0648,
                       0.0662, 0.0645, 0.0661, 0.0639, 0.0674],
    "SteveFriends": [0.0977, 0.0891, 0.0890, 0.0876, 0.0866, 0.0854, 0.0841, 0.0795, 0.0776, 0.0767],
    "whips2": [0.0717, 0.0608, 0.0667, 0.0652, 0.0598, 0.0647, 0.0599, 0.0618],
}


def test_healthy_anima_curves_never_trip():
    for name, curve in HEALTHY.items():
        losses = {i: v for i, v in enumerate(curve, start=1)}
        assert safety.verdict(losses, len(curve)) is None, name
        w = safety.LossWatch()
        assert not any(w.feed(l) for l in _epoch_lines(curve, total=len(curve))), name


def test_epoch_saves_ignore_step_saves_and_finals(tmp_path):
    for n in ("X-000002", "X-000004", "X-step00000250", "X", "Y-000002"):
        (tmp_path / f"{n}.safetensors").write_bytes(b"x")
    assert [s.epoch for s in safety.epoch_saves(str(tmp_path), "X")] == [2, 4]


def _run(tmp_path, epochs=(2, 4, 6, 8, 10), dead=()):
    for e in epochs:
        write_lora(tmp_path / f"X-{e:06d}.safetensors", prefix="lora_unet",
                   b_value=DEAD_B if e in dead else 0.04)
    write_lora(tmp_path / "X.safetensors", prefix="lora_unet")
    return str(tmp_path)


def test_ship_promotes_the_last_good_epoch(tmp_path):
    run = _run(tmp_path)
    found = {"kind": "diverged", "last_good_epoch": 5, "best_epoch": 4, "best": 0.1,
             "spike_epoch": 6, "spike": 0.3, "rise": 2.0, "total_epochs": 10}
    epoch, msg = safety.ship(run, "X", found)
    assert epoch == 4 and "epoch 4" in msg
    assert (tmp_path / "X.overshot.safetensors").is_file()
    assert not (tmp_path / "X-000004.safetensors").exists()


def test_ship_skips_a_dead_snapshot(tmp_path):
    run = _run(tmp_path, dead=(4,))
    found = {"kind": "overshoot", "last_good_epoch": 4, "best_epoch": 4, "best": 0.1,
             "final_epoch": 10, "final": 0.2, "rise": 1.0, "total_epochs": 10}
    epoch, _ = safety.ship(run, "X", found)
    assert epoch == 2


def test_ship_without_a_candidate_keeps_the_final(tmp_path):
    run = _run(tmp_path, epochs=(6, 8))
    before = (tmp_path / "X.safetensors").read_bytes()
    found = {"kind": "overshoot", "last_good_epoch": 4, "best_epoch": 4, "best": 0.1,
             "final_epoch": 10, "final": 0.2, "rise": 1.0, "total_epochs": 10}
    epoch, msg = safety.ship(run, "X", found)
    assert epoch is None and "kept" in msg
    assert (tmp_path / "X.safetensors").read_bytes() == before


def test_keep_last_is_applied_after_the_run_never_to_the_shipped_file(tmp_path):
    run = _run(tmp_path)
    safety.prune(run, "X", keep_last=2, keep_epoch=4)
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == ["X-000004.safetensors", "X-000008.safetensors", "X-000010.safetensors",
                    "X.safetensors"]


def test_keep_all_prunes_nothing(tmp_path):
    run = _run(tmp_path)
    safety.prune(run, "X", keep_last=0, keep_epoch=None)
    assert len(list(tmp_path.iterdir())) == 6


def test_final_check_uses_kohyas_namespace(tmp_path):
    write_lora(tmp_path / "X.safetensors", prefix="lora_unet")
    assert safety.final_problem(str(tmp_path), "X") is None
    write_lora(tmp_path / "X.safetensors", prefix="lora_unet", b_value=DEAD_B)
    assert "did not learn" in safety.final_problem(str(tmp_path), "X")
