import argparse
import glob
import json
import tempfile
from collections import defaultdict
from pathlib import Path


LAST_LAYERS = ["lm_head", "wte", "embed_out"]


def resolve_path(path_value, base_dir):
    """Resolve a path relative to this script when needed."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def discover_training_dirs(file_dir, exp_pattern):
    """Find generation run directories under one parent directory."""
    pattern = f"{exp_pattern}*" if exp_pattern else "*"
    return sorted(glob.glob(str(Path(file_dir) / pattern)))


def assign_real_ids(training_dirs, json_file, gen_bs):
    """Add real_id fields derived from the synthetic sample id."""
    for run in training_dirs:
        file_path = Path(run) / f"{json_file}.jsonl"
        samples = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                sample = json.loads(line)
                sample["real_id"] = int(sample["id"]) // gen_bs
                samples.append(sample)
        with file_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_filtering_module():
    """Import the filtering module lazily."""
    import filtering as filtering_module

    return filtering_module


def freeze_last_layer(model):
    """Mirror the notebook behavior: keep gradients only for the LM head."""
    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if any(substring in name for substring in LAST_LAYERS):
            named_parameters_to_optim.append((name, param))
        else:
            param.requires_grad = False

    assert named_parameters_to_optim, "no layer found"


def build_filter_args(
    filtering_module,
    *,
    dataset,
    model_name,
    file_dir,
    json_file,
    gen_bs,
    data_root,
    split,
    random_seed,
    pos_label,
    neg_label,
    filter_method,
    coeff_perplexity,
    top_n,
    clean,
    balance_score,
    per_label,
    interleave_label,
):
    """Build a filtering.py-compatible Namespace."""
    argv = [
        "filtering_workflow.py",
        "--dataset",
        dataset,
        "--model_name",
        model_name,
        "--split",
        split,
        "--data_root",
        str(data_root),
        "--pos_label",
        pos_label,
        "--neg_label",
        neg_label,
        "--gen_bs",
        str(gen_bs),
        "--random_seed",
        str(random_seed),
        "--filter_score",
        "cls",
        "--filter_method",
        filter_method,
        "--file_dir",
        str(file_dir),
        "--json_file",
        json_file,
        "--coeff_perplexity",
        str(coeff_perplexity),
        "--top_n",
        str(top_n),
        "--use_instruction",
        "false",
        "--use_fewshot",
        "true",
        "--clean",
        str(clean).lower(),
        "--balance_score",
        str(balance_score).lower(),
        "--per_label",
        str(per_label).lower(),
        "--interleave_label",
        str(interleave_label).lower(),
    ]
    return filtering_module.get_args(argv)


def build_clean_args(filtering_module, workflow_args, file_dir):
    """Build args for the clean/remove stage."""
    return build_filter_args(
        filtering_module,
        dataset=workflow_args.dataset,
        model_name=workflow_args.model_name,
        file_dir=file_dir,
        json_file=workflow_args.json_file,
        gen_bs=workflow_args.gen_bs,
        data_root=workflow_args.data_root,
        split=workflow_args.split,
        random_seed=workflow_args.random_seed,
        pos_label=workflow_args.pos_label,
        neg_label=workflow_args.neg_label,
        filter_method="remove",
        coeff_perplexity=0,
        top_n=workflow_args.top_n,
        clean=True,
        balance_score=workflow_args.balance_score,
        per_label=workflow_args.per_label,
        interleave_label=workflow_args.interleave_label,
    )


def get_clean_output_stem(filtering_module, workflow_args, file_dir):
    """Return the output stem produced by the clean/remove stage."""
    clean_args = build_clean_args(filtering_module, workflow_args, file_dir)
    return filtering_module.get_output_file_name(clean_args)


def load_filtering_stack(model_name):
    """Load model/tokenizer/device once and apply the notebook freeze."""
    filtering_module = load_filtering_module()
    tokenizer, model, device = filtering_module.load_model(model_name)
    freeze_last_layer(model)
    return filtering_module, tokenizer, model, device


def run_clean_remove(workflow_args, training_dirs, filtering_module, tokenizer, model, device):
    """Execute the clean/remove stage over one set of runs."""
    assign_real_ids(training_dirs, workflow_args.json_file, workflow_args.gen_bs)
    clean_args = build_clean_args(
        filtering_module, workflow_args, workflow_args.file_dir
    )
    filtering_module.filtering(
        clean_args,
        training_dirs,
        model,
        tokenizer,
        device,
        num_out=workflow_args.num_out,
    )
    return clean_args


def run_recalc_rec_loss(workflow_args, training_dirs, filtering_module, tokenizer, model, device):
    """Recompute rec_loss_ids on the cleaned JSONL files."""
    clean_output_stem = get_clean_output_stem(
        filtering_module, workflow_args, workflow_args.file_dir
    )
    pos_sequences, neg_sequences, pos_labels, neg_labels = filtering_module.load_real_data(
        dataset_name=workflow_args.dataset,
        split=workflow_args.split,
        device=device,
        n_gen_samples=workflow_args.real_n_gen_samples,
        n_fewshot=0,
        random_seed=workflow_args.random_seed,
        subset=workflow_args.real_subset_size,
        data_root=workflow_args.data_root,
    )

    real_pos_grads = filtering_module.compute_average_grads(
        workflow_args,
        model,
        tokenizer,
        pos_sequences,
        pos_labels,
    )
    real_neg_grads = filtering_module.compute_average_grads(
        workflow_args,
        model,
        tokenizer,
        neg_sequences,
        neg_labels,
    )

    pos_text_label = filtering_module.label_to_text(workflow_args.dataset, 1)
    neg_text_label = filtering_module.label_to_text(workflow_args.dataset, 0)

    for run in training_dirs:
        syn_data_path = Path(run) / f"{clean_output_stem}.jsonl"
        if not syn_data_path.exists():
            raise FileNotFoundError(str(syn_data_path))

        syn_pos_sequences, syn_neg_sequences = filtering_module.load_syn_data(
            str(syn_data_path), workflow_args.dataset
        )
        list_raw_pos_loss = filtering_module.calculate_recon_loss_ids(
            syn_pos_sequences,
            [pos_text_label for _ in syn_pos_sequences],
            real_pos_grads,
            model,
            tokenizer,
            dataset=workflow_args.dataset,
        )
        list_raw_neg_loss = filtering_module.calculate_recon_loss_ids(
            syn_neg_sequences,
            [neg_text_label for _ in syn_neg_sequences],
            real_neg_grads,
            model,
            tokenizer,
            dataset=workflow_args.dataset,
        )

        samples = []
        with syn_data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                samples.append(json.loads(line))

        grouped_samples = defaultdict(list)
        for sample in samples:
            grouped_samples[int(sample["label"])].append(sample)

        loss_dict = {1: list_raw_pos_loss, 0: list_raw_neg_loss}
        list_out_samples = []
        for label, label_samples in grouped_samples.items():
            for sample_idx, sample in enumerate(label_samples):
                sample["rec_loss_ids"] = loss_dict[label][sample_idx]
                list_out_samples.append(sample)

        filtering_module.output_to_jsonl(
            build_clean_args(filtering_module, workflow_args, workflow_args.file_dir),
            list_out_samples,
            str(syn_data_path),
            post_processing=False,
        )

    return clean_output_stem


def run_top_score(workflow_args, training_dirs, filtering_module, tokenizer, model, device):
    """Execute the top-score stage over cleaned files."""
    clean_output_stem = get_clean_output_stem(
        filtering_module, workflow_args, workflow_args.file_dir
    )
    top_args = build_filter_args(
        filtering_module,
        dataset=workflow_args.dataset,
        model_name=workflow_args.model_name,
        file_dir=workflow_args.file_dir,
        json_file=clean_output_stem,
        gen_bs=workflow_args.gen_bs,
        data_root=workflow_args.data_root,
        split=workflow_args.split,
        random_seed=workflow_args.random_seed,
        pos_label=workflow_args.pos_label,
        neg_label=workflow_args.neg_label,
        filter_method="top_score",
        coeff_perplexity=workflow_args.coeff_perplexity,
        top_n=workflow_args.top_n,
        clean=True,
        balance_score=workflow_args.balance_score,
        per_label=workflow_args.per_label,
        interleave_label=workflow_args.interleave_label,
    )
    filtering_module.filtering(
        top_args,
        training_dirs,
        model,
        tokenizer,
        device,
        num_out=workflow_args.num_out,
    )
    return top_args


def run_self_test():
    """Validate path discovery, real_id assignment, and naming behavior."""
    filtering_module = load_filtering_module()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        run_dir = tmp_root / "demo-run"
        run_dir.mkdir(parents=True, exist_ok=True)
        synthetic_path = run_dir / "synthetic_data.jsonl"
        samples = [
            {"id": 0, "inputs": "sample 0 It was", "label": 0},
            {"id": 1, "inputs": "sample 1 It was", "label": 1},
            {"id": 2, "inputs": "sample 2 It was", "label": 0},
            {"id": 3, "inputs": "sample 3 It was", "label": 1},
        ]
        with synthetic_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample) + "\n")

        discovered = discover_training_dirs(tmp_root, "demo")
        assert discovered == [str(run_dir)]

        assign_real_ids(discovered, "synthetic_data", 2)
        updated = [json.loads(line) for line in synthetic_path.read_text(encoding="utf-8").splitlines()]
        assert [row["real_id"] for row in updated] == [0, 0, 1, 1]

        args = argparse.Namespace(
            dataset="imdb",
            model_name="phi",
            file_dir=str(tmp_root),
            json_file="synthetic_data",
            gen_bs=2,
            data_root="../data",
            split="validation",
            random_seed=42,
            pos_label="positive",
            neg_label="negative",
            top_n=4,
            coeff_perplexity=0.0,
            balance_score=True,
            per_label=True,
            interleave_label=False,
            real_n_gen_samples=8,
            real_subset_size=4,
            num_out=None,
        )
        clean_stem = get_clean_output_stem(filtering_module, args, str(tmp_root))
        assert clean_stem.startswith("synthetic_data_clean_remove_cls_phi_imdb")

    print("self-test passed")


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Workflow wrapper for gradmm/filtering.py.")
    parser.add_argument("--self-test", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--dataset", choices=["imdb", "rtpolarity"], required=True)
    parent.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parent.add_argument("--file-dir", default="./synthetic_data")
    parent.add_argument("--exp-pattern", default="")
    parent.add_argument("--json-file", default="synthetic_data")
    parent.add_argument("--data-root", default="../data_smoke")
    parent.add_argument("--split", choices=["train", "validation"], default="validation")
    parent.add_argument("--gen-bs", type=int, default=2)
    parent.add_argument("--random-seed", type=int, default=42)
    parent.add_argument("--pos-label", default="positive")
    parent.add_argument("--neg-label", default="negative")
    parent.add_argument("--top-n", type=int, default=4)
    parent.add_argument("--coeff-perplexity", type=float, default=0.0)
    parent.add_argument(
        "--balance-score",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parent.add_argument(
        "--per-label",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parent.add_argument(
        "--interleave-label",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parent.add_argument("--real-n-gen-samples", type=int, default=8)
    parent.add_argument("--real-subset-size", type=int, default=4)
    parent.add_argument("--num-out", type=int, default=None)

    for command in ["clean-remove", "recalc-rec-loss", "top-score", "all"]:
        subparsers.add_parser(command, parents=[parent])

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
    args.file_dir = str(resolve_path(args.file_dir, script_dir))
    args.data_root = str(resolve_path(args.data_root, script_dir))
    training_dirs = discover_training_dirs(args.file_dir, args.exp_pattern)
    if not training_dirs:
        raise FileNotFoundError(f"no runs found under {args.file_dir} with pattern {args.exp_pattern!r}")

    filtering_module, tokenizer, model, device = load_filtering_stack(args.model_name)

    if args.command == "clean-remove":
        run_clean_remove(args, training_dirs, filtering_module, tokenizer, model, device)
    elif args.command == "recalc-rec-loss":
        run_recalc_rec_loss(args, training_dirs, filtering_module, tokenizer, model, device)
    elif args.command == "top-score":
        run_top_score(args, training_dirs, filtering_module, tokenizer, model, device)
    elif args.command == "all":
        run_clean_remove(args, training_dirs, filtering_module, tokenizer, model, device)
        run_recalc_rec_loss(args, training_dirs, filtering_module, tokenizer, model, device)
        run_top_score(args, training_dirs, filtering_module, tokenizer, model, device)


if __name__ == "__main__":
    main()
