import argparse
import glob
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd


PARAM_COLUMNS = [
    "syn_data_path",
    "per_device_train_batch_size",
    "learning_rate",
    "max_steps",
    "num_train",
    "model_name",
    "task_name",
]

METRIC_COLUMNS = [
    "best_valid_acc",
    "best_valid_step",
    "best_valid_per_class_acc",
    "best_test_metric",
    "best_test_step",
    "best_test_per_class_acc",
]


def resolve_path(path_value, base_dir):
    """Resolve a path relative to this script when needed."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def list_finetuning_paths(file_dir, exp_pattern, json_pattern):
    """List filtered synthetic-data files that are ready for fine-tuning."""
    pattern = f"{exp_pattern}*" if exp_pattern else "*"
    return sorted(
        glob.glob(str(Path(file_dir) / pattern / json_pattern))
    )


def collect_results(exp_paths):
    """Load multiple fine-tuning result folders into one DataFrame."""
    df_data = defaultdict(list)
    for exp_path in exp_paths:
        for output_path in sorted(glob.glob(str(Path(exp_path) / "*"))):
            result_path = Path(output_path) / "output" / "main_results.json"
            if not result_path.exists():
                continue
            with result_path.open("r", encoding="utf-8") as handle:
                main_results = json.load(handle)

            for param in PARAM_COLUMNS:
                df_data[param].append(main_results["args"][param])
            for metric in METRIC_COLUMNS:
                df_data[metric].append(main_results[metric])

    return pd.DataFrame(df_data)


def run_self_test():
    """Validate path enumeration and result aggregation on fixtures."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        synthetic_root = tmp_root / "gradmm" / "synthetic_data"
        run_dir = synthetic_root / "demo-run"
        run_dir.mkdir(parents=True, exist_ok=True)
        filtered_path = (
            run_dir
            / "synthetic_data_clean_remove_cls_phi_imdb_positive_negative_instrFalse_fsTrue_top4_score_alpha0.0_per_label_balance_score.jsonl"
        )
        filtered_path.write_text('{"id": 1}\n', encoding="utf-8")

        listed = list_finetuning_paths(
            synthetic_root,
            "demo",
            "synthetic_data_clean_remove_cls_phi_imdb_positive_negative_instrFalse_fsTrue_top*.jsonl",
        )
        assert listed == [str(filtered_path)]

        result_root = tmp_root / "synthetic_data_FT" / "demo-time" / "result"
        exp_dir = result_root / "exp-a"
        output_dir = exp_dir / "seed-0" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "args": {
                "syn_data_path": str(filtered_path),
                "per_device_train_batch_size": 2,
                "learning_rate": 1e-5,
                "max_steps": 2,
                "num_train": 4,
                "model_name": "sshleifer/tiny-gpt2",
                "task_name": "SynIMDB",
            },
            "best_valid_acc": 0.5,
            "best_valid_step": 1,
            "best_valid_per_class_acc": 0.5,
            "best_test_metric": 0.5,
            "best_test_step": 1,
            "best_test_per_class_acc": 0.5,
        }
        (output_dir / "main_results.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        df = collect_results([exp_dir])
        assert df.shape == (1, len(PARAM_COLUMNS) + len(METRIC_COLUMNS))
        assert df.iloc[0]["task_name"] == "SynIMDB"

    print("self-test passed")


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Workflow wrapper for addax fine-tuning notebooks.")
    parser.add_argument("--self-test", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list-paths")
    list_parser.add_argument("--file-dir", default="../gradmm/synthetic_data")
    list_parser.add_argument("--exp-pattern", default="")
    list_parser.add_argument(
        "--json-pattern",
        default="synthetic_data_clean_remove_cls_phi_*_top*.jsonl",
    )

    collect_parser = subparsers.add_parser("collect-results")
    collect_parser.add_argument("--exp-path", action="append", default=[])
    collect_parser.add_argument("--results-root", default=None)
    collect_parser.add_argument("--output-csv", default=None)

    return parser.parse_args()


def main():
    """CLI entrypoint."""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.command:
        raise SystemExit("one command is required unless --self-test is used")

    script_dir = Path(__file__).resolve().parent

    if args.command == "list-paths":
        file_dir = resolve_path(args.file_dir, script_dir)
        paths = list_finetuning_paths(file_dir, args.exp_pattern, args.json_pattern)
        for path in paths:
            print(path)
        print(f"total={len(paths)}")
        return

    exp_paths = [resolve_path(path, script_dir) for path in args.exp_path]
    if args.results_root:
        results_root = resolve_path(args.results_root, script_dir)
        exp_paths.extend(sorted(results_root.glob("*")))
    df = collect_results(exp_paths)
    print(df.to_string(index=False) if not df.empty else "no results found")
    if args.output_csv:
        output_csv = resolve_path(args.output_csv, script_dir)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
