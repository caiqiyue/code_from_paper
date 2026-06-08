#!/usr/bin/env python3
"""Run one E9 federated experiment with PrE-Text-style local clients."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_e9_federated_common import (
    build_client_config_payload,
    build_federated_sidecar,
    build_server_eval_config_payload,
    ensure_partition_manifest,
    export_client_synthetic_texts,
    load_federated_settings,
    resolve_partition_clients,
    resolve_reference_budget,
    resolve_experiment_id,
    resolve_client_prompt_budgets,
    run_client_pipeline_subprocess,
    run_server_eval,
    wait_for_client_vllm_capacity,
    write_aggregated_synthetic_texts,
    write_partition_manifest,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E9 federated PrE-Text-style experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--reference-budget", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg, settings = load_federated_settings(args.config)
    experiment_id = resolve_experiment_id(args.config)
    reference_budget = int(args.reference_budget or resolve_reference_budget(args.config))
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    meta_seed = int(cfg.get("meta", {}).get("seed", 42))
    prebuilt_manifest_path, prebuilt_manifest = ensure_partition_manifest(
        args.config,
        settings=settings,
        seed=meta_seed,
    )
    partition_clients = resolve_partition_clients(prebuilt_manifest)
    prompt_budgets = resolve_client_prompt_budgets(settings)

    client_rows: list[dict[str, object]] = []
    aggregated_texts: list[str] = []
    for idx, client_partition in enumerate(partition_clients):
        client_id = str(client_partition["client_id"])
        client_dir = output_root / "clients" / client_id
        train_path = Path(str(client_partition["train_path"]))
        try:
            wait_for_client_vllm_capacity(timeout_seconds=args.timeout_seconds)
            client_cfg = build_client_config_payload(
                original_config_path=args.config,
                client_output_root=client_dir,
                client_train_path=train_path,
                prompt_budget=prompt_budgets[idx],
                method="e9_pretext",
                reference_budget=reference_budget,
            )
            client_cfg_path = write_yaml(client_dir / "config.yaml", client_cfg)
            summary = run_client_pipeline_subprocess(client_cfg_path, timeout_seconds=args.timeout_seconds)
            client_texts, corpus_path = export_client_synthetic_texts(summary, client_output_root=client_dir)
            aggregated_texts.extend(client_texts)
            client_rows.append(
                {
                    "client_id": client_id,
                    "status": "success",
                    "train_path": str(train_path),
                    "config_path": str(client_cfg_path),
                    "corpus_path": str(corpus_path),
                    "prompt_budget": prompt_budgets[idx],
                    "synthetic_count": len(client_texts),
                }
            )
        except Exception as exc:
            client_rows.append(
                {
                    "client_id": client_id,
                    "status": "failure",
                    "train_path": str(train_path),
                    "prompt_budget": prompt_budgets[idx],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    partition_manifest_path = write_partition_manifest(
        output_root / "partition_manifest.json",
        settings=settings,
        client_rows=client_rows,
    )
    aggregated_texts_path = write_aggregated_synthetic_texts(
        output_root / "aggregated" / "aggregated_synthetic_texts.json",
        aggregated_texts,
    )
    server_cfg = build_server_eval_config_payload(
        original_config_path=args.config,
        server_output_root=output_root / "aggregated",
    )
    server_cfg_path = write_yaml(output_root / "aggregated" / "server_eval_config.yaml", server_cfg)
    if not aggregated_texts:
        raise RuntimeError(
            f"No client synthetic texts were produced for {experiment_id}; source manifest={prebuilt_manifest_path}"
        )
    eval_summary = run_server_eval(
        synthetic_texts=aggregated_texts,
        server_config_path=server_cfg_path,
        output_dir=output_root / "aggregated" / "eval",
    )
    sidecar_path = build_federated_sidecar(
        output_root=output_root,
        experiment_id=experiment_id,
        method="e9_pretext",
        settings=settings,
        partition_manifest_path=partition_manifest_path,
        client_rows=client_rows,
        aggregated_texts=aggregated_texts,
        eval_summary=eval_summary,
        reference_budget=reference_budget,
    )
    print(json.dumps({"experiment_id": experiment_id, "sidecar_path": str(sidecar_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
