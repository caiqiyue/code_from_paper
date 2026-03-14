# PrE-Text Project Manual

## 1. Project Overview

PrE-Text is a two-stage experimental platform for training language models on private federated text data without on-device training.

The pipeline has three major parts:

1. Stage 1: `main.py`
   Generates a small set of differentially private seed texts with the Private Evolution algorithm.
2. Stage 2: `llama_bootstrap.py`
   Expands the DP seed texts into a much larger synthetic corpus with `Llama-2-7b`.
3. Downstream evaluation:
   - `eval_distilgpt2.py` evaluates the small-model setting with `DistilGPT2`.
   - `eval_llama2.py` evaluates the large-model setting with `Llama-2-7b` plus LoRA.

I have added source-level docstrings to every class and function in the project, and added inline comments on the key algorithmic steps, so the Python files themselves can now be read as executable documentation.

## 2. Repository Structure

The repository is intentionally flat. There is no nested Python package directory; the root-level `.py` files are the core modules.

### 2.1 Committed directories and files

| Path | Type | Purpose |
| --- | --- | --- |
| `assets/` | directory | Images used by `README.md` for the paper overview and comparison tables. |
| `README.md` | file | Official repository instructions from the authors. |
| `requirements.txt` | file | Pinned Python dependencies. |
| `custom_datasets.py` | file | Lightweight dataset wrappers used by mutation and training scripts. |
| `variation.py` | file | Candidate mutation logic based on masked token replacement. |
| `similarity.py` | file | Sentence embedding and lookahead embedding helpers. |
| `nn_histogram.py` | file | Differentially private nearest-neighbor histogram computation. |
| `main.py` | file | Stage-1 Private Evolution entry point. |
| `llama_bootstrap.py` | file | Stage-2 synthetic data expansion entry point. |
| `eval_distilgpt2.py` | file | DistilGPT2 downstream evaluation script. |
| `eval_llama2.py` | file | Llama-2 + LoRA downstream evaluation script. |
| `quick_start.md` | file | Earlier high-level quick-start summary. |
| `project_manual.md` | file | This detailed project guide. |

### 2.2 Runtime directories that you need to create yourself

These are not committed to the repository, but the code expects them at runtime:

| Path | Created by | Purpose |
| --- | --- | --- |
| `data/` | user | Stores `initialization.json`, `<dataset>_train.json`, `<dataset>_eval.json`. |
| custom `OUTPUT_DIR/` | user or scripts | Stores experiment outputs. |
| custom `MODEL_DIR/` | user or Hugging Face/vLLM | Stores downloaded model files and caches. |

## 3. Internal Module Breakdown

## 3.1 `custom_datasets.py`

Purpose: provide simple PyTorch dataset wrappers used by the rest of the platform.

### Classes

| Class | Purpose |
| --- | --- |
| `ListDataset` | Wraps a Python list of raw text strings as a `torch.utils.data.Dataset`. |
| `MatrixDataset` | Wraps tokenized `input_ids` and `attention_mask` matrices so candidate sequences can be iterated in batches. |

### Methods

| Symbol | Purpose |
| --- | --- |
| `ListDataset.__init__` | Stores the input text list. |
| `ListDataset.__len__` | Returns dataset size. |
| `ListDataset.__getitem__` | Returns one raw text by index. |
| `MatrixDataset.__init__` | Stores token ids and attention masks. |
| `MatrixDataset.__len__` | Returns number of tokenized sequences. |
| `MatrixDataset.__getitem__` | Returns one sequence and mask in batch-shaped form. |

## 3.2 `variation.py`

Purpose: mutate candidate texts by masking and refilling tokens with a masked language model.

This module is the "variation" operator in Private Evolution.

### Functions

| Function | Purpose |
| --- | --- |
| `top_k_top_p_filtering` | Applies top-k and/or top-p filtering to token logits before sampling. |

### Classes

| Class | Purpose |
| --- | --- |
| `Variation` | Groups mutation-related utilities used by stage 1. |

### Methods

| Method | Purpose |
| --- | --- |
| `Variation.collate_fn_tokenizer` | Builds a batch of parent candidates and randomly masks a fraction of valid tokens. |
| `Variation.sample` | Iteratively fills masked positions by sampling from `roberta-large`. |
| `Variation.produce_variation` | Repeats mask-and-fill for `t_steps` rounds to produce the next candidate population. |

## 3.3 `similarity.py`

Purpose: compute embeddings for private texts and candidate texts.

This module is used by the DP histogram stage to measure how well a candidate population matches the private data distribution.

### Classes

| Class | Purpose |
| --- | --- |
| `Similarity` | Provides embedding helpers for private texts and synthetic candidates. |

### Methods

| Method | Purpose |
| --- | --- |
| `Similarity.sentence_embedding` | Encodes raw text into dense sentence embeddings. |
| `Similarity.concat_embedding` | Computes the MiniLM embeddings used in nearest-neighbor scoring. |
| `Similarity.lookahead_embedding` | Generates several future variations of the same candidate set, embeds them, and averages the embeddings. |

## 3.4 `nn_histogram.py`

Purpose: compute the differentially private quality scores used to select candidates.

This module is where DP noise is injected into the selection process.

### Classes

| Class | Purpose |
| --- | --- |
| `NN_Histogram` | Computes noisy nearest-neighbor vote histograms over synthetic candidates. |

### Methods

| Method | Purpose |
| --- | --- |
| `NN_Histogram.split_given_size` | Splits a large embedding matrix into fixed-size chunks. |
| `NN_Histogram.dp_nn_histogram` | Computes lookahead embeddings, performs FAISS nearest-neighbor search, adds Gaussian noise, applies thresholding, and returns a DP histogram. |

## 3.5 `main.py`

Purpose: stage-1 PrE-Text generation entry point.

High-level flow:

1. Load `roberta-large` and `all-MiniLM-L6-v2`.
2. Load private training texts from `./data/<dataset>_train.json`.
3. Load initialization population from `./data/initialization.json`.
4. Compute privacy accounting with Opacus RDP analysis.
5. Build the first candidate population.
6. Repeat 11 Private Evolution rounds:
   - compute lookahead embeddings,
   - build DP nearest-neighbor histogram,
   - resample survivors,
   - mutate survivors into the next generation.
7. Save intermediate texts and cached private embeddings.

### Functions

| Function | Purpose |
| --- | --- |
| `main` | Runs the full stage-1 Private Evolution loop and writes outputs to disk. |

## 3.6 `llama_bootstrap.py`

Purpose: stage-2 synthetic data expansion entry point.

High-level flow:

1. Read all `surviving_text_it*.json` files from stage 1.
2. Build few-shot prompts by sampling three seed texts per prompt.
3. Use `meta-llama/Llama-2-7b-hf` through `vllm` to generate continuations.
4. Save the expanded synthetic corpus to `llama7b_text_syn.json`.

### Functions

| Function | Purpose |
| --- | --- |
| `build_output_dir` | Reconstructs the experiment directory name from CLI parameters. |
| `load_surviving_seed_texts` | Reads all stage-1 surviving seed files. |
| `build_bootstrap_prompts` | Creates the few-shot prompt list used by Llama 2. |
| `generate_bootstrapped_samples` | Runs batched vLLM generation and returns raw texts. |
| `parse_args` | Parses bootstrap CLI parameters. |
| `main` | Orchestrates stage-2 prompt generation and output saving. |

## 3.7 `eval_distilgpt2.py`

Purpose: downstream evaluation in the small-model setting.

High-level flow:

1. Read synthetic data from `llama7b_text_syn.json`.
2. Read evaluation texts from `./data/<dataset>_eval.json`.
3. Load `distilgpt2`.
4. Load `./c4_checkpoint.pth` as a warm-start checkpoint.
5. Fine-tune on the synthetic corpus.
6. Report cross-entropy and top-k accuracy.
7. Save per-epoch stats and checkpoints.

### Functions

| Function | Purpose |
| --- | --- |
| `evaluate` | Computes loss and top-k token accuracy on the evaluation split. |
| `save_checkpoint` | Saves model, optimizer, and accelerator state. |
| `add_module_prefix` | Normalizes checkpoint key names for wrapped models. |
| `load_checkpoint` | Loads a previously saved checkpoint. |
| `find_latest_checkpoint` | Finds the latest checkpoint file in a directory. |
| `build_output_dir` | Reconstructs the experiment directory name. |
| `main` | Runs DistilGPT2 fine-tuning and evaluation. |
| nested `tokenize` | Converts one text example into `input_ids`, `attention_mask`, and `labels`. |

## 3.8 `eval_llama2.py`

Purpose: downstream evaluation in the large-model setting.

High-level flow:

1. Read synthetic data from `llama7b_text_syn.json`.
2. Read evaluation texts from `./data/<dataset>_eval.json`.
3. Load `meta-llama/Llama-2-7b-hf`.
4. Add LoRA adapters.
5. Fine-tune on the synthetic corpus.
6. Report cross-entropy and top-k accuracy.
7. Save per-epoch stats and checkpoints.

### Functions

| Function | Purpose |
| --- | --- |
| `evaluate` | Computes loss and top-k token accuracy on the evaluation split. |
| `save_checkpoint` | Saves model, optimizer, and accelerator state. |
| `add_module_prefix` | Normalizes checkpoint key names for wrapped models. |
| `load_checkpoint` | Loads a previously saved checkpoint. |
| `find_latest_checkpoint` | Finds the latest checkpoint file in a directory. |
| `build_output_dir` | Reconstructs the experiment directory name. |
| `main` | Runs Llama-2 LoRA fine-tuning and evaluation. |
| nested `tokenize` | Converts one text example into `input_ids`, `attention_mask`, and `labels`. |

## 4. External Python Packages and What They Do

`requirements.txt` contains the full pinned dependency set. Most entries are either direct runtime libraries or transitive dependencies of the ML stack.

The core packages actually used by the project source code are:

| Package | Role in this project | Main files |
| --- | --- | --- |
| `torch` | Tensor operations, model execution, training, loss computation. | all training/generation scripts |
| `transformers` | Loads `roberta-large`, `distilgpt2`, and `Llama-2-7b`. | `main.py`, `variation.py`, `eval_distilgpt2.py`, `eval_llama2.py` |
| `accelerate` | Multi-GPU/distributed preparation and gathering. | `main.py`, `variation.py`, both eval scripts |
| `sentence-transformers` | Loads `all-MiniLM-L6-v2` to embed texts. | `main.py`, `similarity.py` |
| `faiss-cpu` | Nearest-neighbor search for candidate scoring. | `nn_histogram.py` |
| `opacus` | RDP privacy accounting for epsilon estimation. | `main.py` |
| `numpy` | Numerical operations and noisy histogram processing. | `main.py`, `similarity.py`, `nn_histogram.py` |
| `datasets` | Hugging Face dataset utilities for tokenization pipelines. | both eval scripts |
| `peft` | LoRA adapter setup for Llama-2 evaluation. | `eval_llama2.py` |
| `vllm` | Efficient Llama-2 inference for stage-2 expansion. | `llama_bootstrap.py` |
| `json`, `os`, `argparse`, `sys`, `time`, `random`, `re` | Standard-library utilities for file IO, CLI parsing, logging, sampling, and text cleanup. | multiple scripts |

The rest of the packages in `requirements.txt` mainly support:

1. Hugging Face and tokenizer internals
2. CUDA acceleration and low-level kernels
3. vLLM serving/runtime dependencies
4. scientific Python support packages

## 5. How To Create a Virtual Environment

Two practical options are below.

### 5.1 Option A: Conda

```powershell
conda create -n pretext python=3.10 -y
conda activate pretext
```

### 5.2 Option B: Python venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 6. How To Install Third-Party Packages

After activating the environment, install the dependencies from the root of the repository:

```powershell
pip install -r requirements.txt
```

Recommended follow-up:

```powershell
python -m pip install --upgrade pip
accelerate config
```

Notes:

1. `accelerate config` is recommended before any multi-GPU run.
2. `meta-llama/Llama-2-7b-hf` usually requires that your Hugging Face account has accepted the model license.
3. The repository targets GPU execution. The authors mention V100 32GB and A40 48GB in `README.md`.

## 7. Required Input Data

The repository does not ship with real experimental datasets. You need to prepare the input files yourself.

### 7.1 Required files under `data/`

| File | Format | Purpose |
| --- | --- | --- |
| `data/initialization.json` | JSON list of strings | Initial public/seed pool `S1`; README says at least 10000 samples. |
| `data/<dataset_name>_train.json` | JSON list of strings | Aggregated private training samples. |
| `data/<dataset_name>_eval.json` | JSON object like `{"1": ["..."]}` | Evaluation split. |

### 7.2 Required extra file for small-model evaluation

| File | Purpose |
| --- | --- |
| `./c4_checkpoint.pth` | Warm-start checkpoint for `eval_distilgpt2.py`, described in the README as a DistilGPT2 checkpoint trained on a subset of C4. |

### 7.3 Dataset notes from the authors

1. The max number of samples per client should stay small, and the README recommends keeping it below 16.
2. `sensitivity` is the per-client maximum sample count and directly affects privacy noise.
3. If a client has too many samples, subsample before aggregating.

## 8. End-to-End Experiment Workflow

## 8.1 Stage 1: Generate DP seed texts

```powershell
$env:OUTPUT_DIR = "D:\results\pretext"
$env:MODEL_DIR = "D:\models\hf_cache"
$env:DATASET_NAME = "my_dataset"
$env:MAX_SAMPLES = "16"
$env:NOISE = "11.3"
$env:DELTA = "3e-6"
$env:TOKENIZERS_PARALLELISM = "false"

accelerate launch main.py `
  -outputdir $env:OUTPUT_DIR `
  -cachedir $env:MODEL_DIR `
  -datadir $env:DATASET_NAME `
  -sensitivity $env:MAX_SAMPLES `
  -sigma $env:NOISE `
  -delta $env:DELTA
```

What this stage does:

1. Downloads or loads `roberta-large` and `all-MiniLM-L6-v2`.
2. Runs 11 rounds of Private Evolution.
3. Writes DP seed and intermediate outputs into one experiment directory.

## 8.2 Stage 2: Expand the seed texts

```powershell
python .\llama_bootstrap.py `
  -outputdir $env:OUTPUT_DIR `
  -cachedir $env:MODEL_DIR `
  -datadir $env:DATASET_NAME `
  -sensitivity $env:MAX_SAMPLES `
  -sigma $env:NOISE `
  -delta $env:DELTA
```

What this stage does:

1. Reads `surviving_text_it*.json` from stage 1.
2. Creates few-shot prompts from the surviving seeds.
3. Uses `Llama-2-7b` to generate a larger synthetic corpus.
4. Writes `llama7b_text_syn.json`.

## 8.3 Downstream evaluation with DistilGPT2

Precondition: `./c4_checkpoint.pth` must exist.

```powershell
accelerate launch eval_distilgpt2.py `
  -outputdir $env:OUTPUT_DIR `
  -cachedir $env:MODEL_DIR `
  -datadir $env:DATASET_NAME `
  -sensitivity $env:MAX_SAMPLES `
  -sigma $env:NOISE `
  -delta $env:DELTA
```

## 8.4 Downstream evaluation with Llama-2 + LoRA

```powershell
accelerate launch eval_llama2.py `
  -outputdir $env:OUTPUT_DIR `
  -cachedir $env:MODEL_DIR `
  -datadir $env:DATASET_NAME `
  -sensitivity $env:MAX_SAMPLES `
  -sigma $env:NOISE `
  -delta $env:DELTA
```

## 9. Parameter Reference

The scripts share a common naming scheme for experiment directories, so most CLI arguments need to stay consistent across stage 1, stage 2, and evaluation.

### 9.1 Shared parameters

| Parameter | Required | Meaning |
| --- | --- | --- |
| `-datadir` | no, but practically yes | Dataset name prefix. If `myset`, scripts expect `./data/myset_train.json` and `./data/myset_eval.json`. |
| `-outputdir` | yes | Base directory where experiment results are stored. |
| `-cachedir` | yes | Directory where downloaded models are cached. |
| `-sensitivity` | yes | Maximum number of samples per client; also the DP sensitivity. |
| `-delta` | no, default `3e-6` | Delta in `(epsilon, delta)`-DP. |
| `-sigma` | yes | Noise-to-sensitivity ratio. |
| `-mask` | no, default `0.3` | Fraction of valid tokens masked in each mutation round. |
| `-lookahead` | no, default `4` | Number of future variations averaged to score a candidate. |
| `-multiplier` | no, default `4` | Controls the stage-1 synthetic population size. In stage 1, `nsyn = 256 * multiplier * num_gpus`. |
| `-seq_len` | no, default `64` | Maximum token sequence length used during generation/evaluation preprocessing. |
| `-t_steps` | no, default `2` | Number of repeated mask-fill mutation steps per round. |
| `-trial` | no, default `0` | Trial id appended to the experiment directory. |
| `-H_multiplier` | no, default `0.25` | Multiplier that controls threshold `H` in DP histogram post-processing. |

### 9.2 Script-specific practical meaning

| Script | Core role |
| --- | --- |
| `main.py` | Uses all of the above parameters to generate DP seeds. |
| `llama_bootstrap.py` | Uses the same parameters mainly to locate the correct experiment directory, then writes expanded synthetic texts there. |
| `eval_distilgpt2.py` | Uses the same parameters mainly to locate the correct synthetic corpus and output directory. |
| `eval_llama2.py` | Uses the same parameters mainly to locate the correct synthetic corpus and output directory. |

### 9.3 Privacy values mentioned in the README

The README states:

1. `delta = 3e-6`
2. `sigma = 11.3` corresponds to approximately `epsilon = 1.29`
3. `sigma = 2.31` corresponds to approximately `epsilon = 7.58`

## 10. Output Directory Layout

The scripts construct the experiment directory using this pattern:

```text
<OUTPUT_DIR>/<dataset>_<mask>_<lookahead>_<nsyn>_<t_steps>_<H_multiplier>_<sensitivity>_<sigma>_<delta>_<trial>/
```

Typical outputs:

| Artifact | Producer | Purpose |
| --- | --- | --- |
| `private_embeds.npy` | `main.py` | Cached embeddings of the private training texts. |
| `generated_text_it0.json` ... `generated_text_it10.json` | `main.py` | Generated texts for each Private Evolution round. |
| `surviving_text_it0.json` ... `surviving_text_it10.json` | `main.py` | Unique surviving seeds after each selection round. |
| `llama7b_text_syn.json` | `llama_bootstrap.py` | Expanded synthetic training corpus. |
| `log_models_and_accuracies/` | `eval_distilgpt2.py` | DistilGPT2 checkpoints and per-epoch stats. |
| `llama2_models_and_accuracies/` | `eval_llama2.py` | Llama-2 LoRA checkpoints and per-epoch stats. |

## 11. Suggested Reading Order for the Code

If you want to understand the platform from top to bottom, read in this order:

1. `README.md`
2. `main.py`
3. `nn_histogram.py`
4. `similarity.py`
5. `variation.py`
6. `custom_datasets.py`
7. `llama_bootstrap.py`
8. `eval_distilgpt2.py`
9. `eval_llama2.py`

That order follows the actual experiment flow and makes the internal dependencies easier to follow.

## 12. Important Practical Notes

1. The repository itself does not include the original private experimental data from the paper.
2. `eval_distilgpt2.py` depends on `c4_checkpoint.pth`, which you must prepare yourself.
3. `meta-llama/Llama-2-7b-hf` access may require a Hugging Face license agreement.
4. The current `llama_bootstrap.py` default generates `50000` samples, while the README says the paper scaled this much further.
5. Source files now contain class-level and function-level docstrings plus inline comments on key steps, so the code can be used together with this manual.
> Note
> 当前仓库已经平台化重构为 `pretext_platform/`。
> 本文档保留的是旧版扁平脚本结构的细读说明；最新项目结构、配置驱动入口和实验方式请优先参考根目录 `README.md`。
