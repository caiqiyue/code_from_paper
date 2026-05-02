# DP-Prompt Smoke and Sweep Experiments

## Goal

Define real, non-mock experiments that validate the full Round 1 DP-Prompt pipeline:

- real dataset splits,
- real local generation model,
- real sanitized document outputs,
- real utility evaluation,
- real static/adaptive text attacks.

This document now covers **two runnable experiment families**:

1. **paper-style document pipeline**
   - original DP-Prompt-shaped document sanitization loop
2. **pretext-style four-dataset pipeline**
   - adapted to the current experimental resources:
     - `jobs`
     - `congressional`
     - `forums`
     - `microblog`
   - server generation through `vllm + thesis_platform/open_model/llama_2_7b_hf`
   - downstream small evaluation through `distilgpt2 / gpt2` via `thesis_platform`

## Required real resources

### Dataset

Provide real IMDb split files through environment variables:

```bash
export DP_PROMPT_IMDB_TRAIN=/absolute/path/to/imdb_train.jsonl
export DP_PROMPT_IMDB_VALIDATION=/absolute/path/to/imdb_validation.jsonl
export DP_PROMPT_IMDB_TEST=/absolute/path/to/imdb_test.jsonl
```

Each split file must contain real records with at least:

- `text`
- `label`
- `author_id`

### Local model

Provide a real local open-source model path:

```bash
export DP_PROMPT_LOCAL_MODEL_PATH=/absolute/path/to/local/model
```

Optional, only if tokenizer files live outside the model directory:

```bash
export DP_PROMPT_LOCAL_TOKENIZER_PATH=/absolute/path/to/tokenizer
```

## Smoke experiments

### 1. Base experiment

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
bash scripts/run_dp_prompt.sh configs/experiments/r1_imdb_base.yaml
```

Purpose:

- verifies the full pipeline runs end to end once
- verifies all output artifacts are created

### 2. Temperature sweep

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
bash scripts/run_dp_prompt.sh configs/experiments/r1_imdb_temp_low.yaml
bash scripts/run_dp_prompt.sh configs/experiments/r1_imdb_temp_mid.yaml
bash scripts/run_dp_prompt.sh configs/experiments/r1_imdb_temp_high.yaml
```

Purpose:

- checks that the privacy-utility sweep surface is wired correctly
- confirms summaries differ by temperature while preserving the same attack/eval structure

## Pretext-style four-dataset experiments

These experiments are the ones aligned with the current comparison environment and should use the same Linux `pretext` setup as the rest of the project's formal runs.

### Required dataset environment variables

```bash
export DP_PROMPT_PRETEXT_JOBS_TRAIN=/mnt/public/caiqiyue_file/code_from_paper/thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
export DP_PROMPT_PRETEXT_JOBS_EVAL=/mnt/public/caiqiyue_file/code_from_paper/thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json

export DP_PROMPT_PRETEXT_FORUMS_TRAIN=/mnt/public/caiqiyue_file/code_from_paper/thesis_platform/datasets/pretext_forums/formatted/forums_train.json
export DP_PROMPT_PRETEXT_FORUMS_EVAL=/mnt/public/caiqiyue_file/code_from_paper/thesis_platform/datasets/pretext_forums/formatted/forums_eval.json

export DP_PROMPT_PRETEXT_MICROBLOG_TRAIN=/mnt/public/caiqiyue_file/code_from_paper/thesis_platform/datasets/pretext_microblog/formatted/microblog_train.json
export DP_PROMPT_PRETEXT_MICROBLOG_EVAL=/mnt/public/caiqiyue_file/code_from_paper/thesis_platform/datasets/pretext_microblog/formatted/microblog_eval.json

export DP_PROMPT_PRETEXT_CONGRESSIONAL_TRAIN=/mnt/public/caiqiyue_file/code_from_paper/thesis_platform/datasets/congressional/formatted/congressional_train.json
export DP_PROMPT_PRETEXT_CONGRESSIONAL_EVAL=/mnt/public/caiqiyue_file/code_from_paper/thesis_platform/datasets/congressional/formatted/congressional_eval.json

export DP_PROMPT_PRETEXT_INIT=/mnt/public/caiqiyue_file/code_from_paper/thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

### Required model/runtime expectations

- conda env: `pretext`
- server generation backend: `vllm`
- server model path: `thesis_platform/open_model/llama_2_7b_hf`
- downstream small eval model path: `thesis_platform/open_model/distilgpt2`

### Run commands

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
bash scripts/run_dp_prompt.sh configs/experiments/p1_jobs_pretext_style.yaml
bash scripts/run_dp_prompt.sh configs/experiments/p1_congressional_pretext_style.yaml
bash scripts/run_dp_prompt.sh configs/experiments/p1_forums_pretext_style.yaml
bash scripts/run_dp_prompt.sh configs/experiments/p1_microblog_pretext_style.yaml
```

### Expected behavior

- input dataset files are plain JSON `list[str]`
- the runner renders one paraphrase prompt per selected private sample
- server generation runs with `vllm`
- sanitized outputs are exported into `stage2/llama7b_text_syn.json`
- evaluation is delegated to `thesis_platform.evaluation.downstream_eval.run_pretext_small_eval`
- final metrics should include the same `best_top1 / best_top3 / best_top5 / best_top10` family used in the rest of the project

## Required output artifacts

Each experiment output directory must contain:

- `sanitized_corpus.json`
- `utility_summary.json`
- `privacy_attack_summary.json`
- `privacy_controls_summary.json`
- `experiment_summary.json`

For pretext-style experiments, the output directory should also contain:

- `stage2/llama7b_text_syn.json`
- `eval/eval_small_summary.json`

## Minimum success checklist

The smoke experiments are considered successful if:

1. sanitized text is generated for train / validation / test records,
2. utility evaluation writes a structured result,
3. both static and adaptive text attacks write structured results,
4. privacy-control metadata is persisted,
5. experiment summary references all artifact paths.

For pretext-style runs, add two more checks:

6. `eval/eval_small_summary.json` exists and contains structured downstream metrics,
7. the synthetic corpus written to `stage2/llama7b_text_syn.json` is a non-empty JSON list of strings.
