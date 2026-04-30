from dp_fedavg.data import build_client_partitions, detect_partition_mode, load_private_samples


def test_load_private_samples_reads_real_jobs_dataset() -> None:
    samples = load_private_samples(
        dataset_name="jobs",
        train_path="thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json",
        train_limit=12,
    )
    assert len(samples) == 12
    assert all(sample.dataset_name == "jobs" for sample in samples)


def test_detect_partition_mode_returns_supported_value() -> None:
    samples = load_private_samples(
        dataset_name="jobs",
        train_path="thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json",
        train_limit=12,
    )
    mode = detect_partition_mode(samples, natural_user_fields=["speaker", "source_domain"])
    assert mode in {"natural", "pseudo"}


def test_build_client_partitions_returns_multiple_clients() -> None:
    samples = load_private_samples(
        dataset_name="jobs",
        train_path="thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json",
        train_limit=24,
    )
    partitions = build_client_partitions(
        samples,
        partition_mode="pseudo",
        num_clients=4,
        max_samples_per_client=8,
        seed=42,
        natural_user_fields=["speaker", "source_domain"],
    )
    assert len(partitions) == 4
    assert sum(len(partition.samples) for partition in partitions) > 0
