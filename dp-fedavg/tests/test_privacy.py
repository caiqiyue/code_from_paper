import math

from dp_fedavg.privacy import add_gaussian_noise, clip_update, compute_noise_scale, estimate_privacy_epsilon


def test_clip_update_caps_l2_norm() -> None:
    update = {"w": [3.0, 4.0]}
    clipped = clip_update(update, clip_norm=2.5)
    norm = math.sqrt(sum(value * value for value in clipped["w"]))
    assert norm <= 2.500001


def test_compute_noise_scale_matches_clip_times_multiplier() -> None:
    assert compute_noise_scale(clip_norm=1.5, noise_multiplier=0.8) == 1.2000000000000002


def test_add_gaussian_noise_is_deterministic_for_fixed_seed() -> None:
    update = {"w": [0.0, 0.0]}
    left = add_gaussian_noise(update, noise_scale=0.1, seed=42)
    right = add_gaussian_noise(update, noise_scale=0.1, seed=42)
    assert left == right


def test_estimate_privacy_epsilon_grows_with_rounds() -> None:
    short = estimate_privacy_epsilon(rounds=1, sample_rate=0.5, noise_multiplier=0.8, delta=1.0e-5)
    long = estimate_privacy_epsilon(rounds=3, sample_rate=0.5, noise_multiplier=0.8, delta=1.0e-5)
    assert long > short
