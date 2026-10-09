from core.engines.onetrainer import verify


def test_divergence_threshold_is_a_parameter():
    curve = {1: 1.0, 2: 0.5, 3: 0.5, 4: 0.65}          # +30% at epoch 4
    assert verify.divergence(curve, 10) is None          # default 50%
    found = verify.divergence(curve, 10, rise=0.25)
    assert found["spike_epoch"] == 4 and found["last_good_epoch"] == 3


def test_overshoot_threshold_is_a_parameter():
    curve = {1: 1.0, 2: 0.5, 3: 0.55}                    # +10% at the end
    assert verify.overshoot(curve, 3) is None
    assert verify.overshoot(curve, 3, rise=0.05)["best_epoch"] == 2


def test_settling_is_a_parameter():
    curve = {i: 0.5 for i in range(1, 21)}
    curve[1] = 0.2                                       # low by chance at the start
    assert verify.overshoot(curve, 20) is None           # epoch 1 is inside settling
    assert verify.overshoot(curve, 20, settle=0.0)["best_epoch"] == 1
