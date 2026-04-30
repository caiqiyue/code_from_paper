import math

from dp_fedavg.privacy import clip_update, compute_noise_scale


def test_clip_update_caps_l2_norm() -> None:
    update = {"w": [3.0, 4.0]}
    clipped = clip_update(update, clip_norm=2.5)
    norm = math.sqrt(sum(value * value for value in clipped["w"]))
    assert norm <= 2.500001


def test_compute_noise_scale_matches_clip_times_multiplier() -> None:
    assert compute_noise_scale(clip_norm=1.5, noise_multiplier=0.8) == 1.2000000000000002
