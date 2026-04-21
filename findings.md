# Findings

## Scope
- Target codebases: `PrE-Text`, `thesis_platform`
- Target docs/configs: `execute/new/*`, especially the formal single-node experiment design and process files
- Output required: step-wise workflow reconstruction, model inventory, synthetic-data generator identification

## Discovery Log
- Planning files reset for the current task on 2026-04-21.
- `execute/new/单节点正式实验设计.md` defines the formal single-node comparison as `SN-C*` (innovation algorithm in `thesis_platform`) versus `SP-C*` (baseline in `PrE-Text`).
- `execute/new/单节点实验流程.md` states the intended single-node commands:
  - `python -m thesis_platform.scripts.run_experiment --config thesis_platform/configs/experiments/single_node_formal/sn_c1_jobs_base.yaml`
  - `python -m pretext_platform.scripts.run_pipeline --config PrE-Text/configs/experiments/single_node_formal/sp_c1_jobs_base.yaml`
  - `python -m pretext_platform.scripts.run_eval_small --config PrE-Text/configs/experiments/single_node_formal/sp_c1_jobs_base.yaml`
- The formal fairness constraints in docs are:
  - same dataset,
  - same privacy budget,
  - same final synthetic sample count (`1500`),
  - same downstream evaluation model (`gpt2`).
- The docs claim the synthetic-data generator is:
  - single-node innovation: `llama_2_7b_hf`
  - single-node pre-text bootstrap: `llama2_7b` via `vllm`

## Single-Node Innovation Workflow
- Formal mainline config: `thesis_platform/configs/experiments/single_node_formal/sn_c1_jobs_base.yaml`.
- Config inheritance:
  - `_base_single_node_formal.yaml`
  - `base/llm_7b_linux.yaml`
  - `methods/generators/pretext_prompt_llm.yaml`
  - `methods/scorers/datainf_real.yaml`
  - `methods/retrievers/knn.yaml`
  - `methods/critics/fedtextgrad_llm.yaml`
  - `methods/aggregators/dbscan_attn_tsgdm.yaml`
  - `methods/downstream_eval/pretext_large_off.yaml`
- Entrypoint: `python -m thesis_platform.scripts.run_experiment --config thesis_platform/configs/experiments/single_node_formal/sn_c1_jobs_base.yaml`.
- Runtime dispatch:
  - `run_experiment.py` calls `thesis_platform.core.pipeline.run_pipeline`.
  - `pipeline.run_pipeline()` detects `execution.mode: single_node` and constructs `SingleNodeRunner`.
- Stage A:
  - load seed corpus from `jobs_train.json`,
  - generate `stage_a.generated_count=100` candidates per iteration with current prompt,
  - score candidates with `DataInfRealScorer`,
  - select worst `stage_a.select_top_k=10`,
  - retrieve real anchors with KNN,
  - generate FedTextGrad critique rules with the client LLM backend,
  - aggregate rules with DBSCAN-Attn-TSGDM,
  - append rules to prompt,
  - stop at score threshold, unchanged prompt, or `max_iterations=10`.
- Stage B:
  - use the final optimized prompt,
  - generate `stage_b.generated_count=1500` final synthetic texts,
  - save `stage_b/llama7b_text_syn.json`.
- Evaluation:
  - export synthetic corpus into `eval/stage2/llama7b_text_syn.json`,
  - call PrE-Text small eval in-process,
  - configured mode is `gpt2`,
  - `run_large_eval=false`.
- Mainline models:
  - generator text backend: `thesis_platform/open_model/llama_2_7b_hf` via Transformers,
  - critic text backend: same `llama_2_7b_hf` client config via Transformers,
  - scorer feature model: `thesis_platform/open_model/roberta_large`,
  - retriever/aggregator embedding model: `thesis_platform/open_model/all_minilm_l6_v2`,
  - downstream small eval mode: `gpt2`, implemented through `pretext_platform.evaluation.gpt2_eval`.
- Supplemental single-node variants:
  - `SN-A1`: replaces scorer with `gradmm_real`,
  - `SN-A2`: replaces scorer with `ira`,
  - `SN-A3`: replaces scorer with `random`,
  - `SN-A4`: replaces aggregator with `uid`,
  - `SN-A5`: replaces aggregator with `summarization`.

## Pre-Text Workflow
- Formal mainline config: `PrE-Text/configs/experiments/single_node_formal/sp_c1_jobs_base.yaml`.
- Config inheritance:
  - `_base_pretext_formal.yaml`,
  - `base/paths.yaml`,
  - `base/models.yaml`,
  - `base/runtime.yaml`,
  - `templates/noise_eps129.yaml`.
- Entrypoint:
  - `python -m pretext_platform.scripts.run_pipeline --config PrE-Text/configs/experiments/single_node_formal/sp_c1_jobs_base.yaml`,
  - then `python -m pretext_platform.scripts.run_eval_small --config PrE-Text/configs/experiments/single_node_formal/sp_c1_jobs_base.yaml`.
- Pipeline:
  - `run_pipeline` runs Stage 1 and Stage 2 because both are enabled,
  - it does not run eval_small inside pipeline for formal configs because `eval_small.enabled=false`,
  - formal eval is run by the separate `run_eval_small` command.
- Stage 1 Private Evolution:
  - load initialization pool and private training texts,
  - load `RobertaForMaskedLM`/`RobertaTokenizer` from `roberta_large`,
  - load `SentenceTransformer` from `all_minilm_l6_v2`,
  - compute private embeddings for training texts,
  - initialize parent population of `nsyn=batch_size*multiplier=64*4=256`,
  - for 25 rounds: compute DP NN histogram, resample parents, use masked-LM variation to create new candidate texts, save generated and surviving texts.
- Stage 2 Bootstrap:
  - load all Stage 1 surviving texts,
  - build 1500 few-shot prompts from triples of surviving texts,
  - choose `bootstrap.generator_model=llama2_7b`,
  - resolve to `../thesis_platform/open_model/llama_2_7b_hf`,
  - require `bootstrap.generator_backend=vllm`,
  - generate final outputs with `vLLM.LLM(..., tensor_parallel_size=1)`,
  - save `stage2/llama7b_text_syn.json`.
- Evaluation:
  - separate `run_eval_small` chooses `run_gpt2_eval` when `eval_small.eval_mode=gpt2`,
  - large eval remains disabled.

## Doc/Code Alignment
- The major doc claims match code: `SN-C1` uses `thesis_platform` single-node runner, `SP-C1` uses `PrE-Text` pipeline + separate small eval, both target 1500 final synthetic samples and small eval only.
- Important ambiguity: the small-eval mode is named `gpt2`, but `PrE-Text/pretext_platform/evaluation/gpt2_eval.py` loads `model_paths.distilgpt2`, which resolves by default to `../thesis_platform/open_model/distilgpt2`. So the code path is GPT-2-style causal LM evaluation, but the concrete local checkpoint depends on what is stored in that `distilgpt2` directory.
- The synthetic-data generator is not ambiguous:
  - innovation `SN-C1`: `thesis_platform/open_model/llama_2_7b_hf` through the Transformers backend,
  - pre-text `SP-C1`: `../thesis_platform/open_model/llama_2_7b_hf` through vLLM in Stage 2 bootstrap.

## Open Questions
- None blocking.
