from dp_fedavg.aggregation import fixed_denominator_average


def test_fixed_denominator_average_uses_expected_user_count() -> None:
    updates = [
        {"w": [1.0, 2.0]},
        {"w": [3.0, 4.0]},
    ]
    averaged = fixed_denominator_average(updates, expected_clients=4)
    assert averaged["w"] == [1.0, 1.5]
