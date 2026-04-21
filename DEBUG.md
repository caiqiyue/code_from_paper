## Observations

- The project root is `/mnt/public/caiqiyue_files/GATV2-TransDTI`.
- The conda environment `GATV2-TransDTI` exists at `/home/k8smaster/anaconda3/envs/GATV2-TransDTI`.
- `python -V` inside the env is `Python 3.8.20`.
- `python -m pip check` initially reported `No broken requirements found`.
- `which pip` inside an activated shell resolves to `/home/k8smaster/.local/bin/pip` first, even though `/home/k8smaster/anaconda3/envs/GATV2-TransDTI/bin/pip` exists.
- The demo run of `main.py` reached training and evaluation, but printed a fallback warning for `GATv2Conv` and then failed on DGL GPU usage because the installed DGL build was CPU-only.
- Source inspection shows hard imports for `dgl`, `dgllife`, `rdkit`, `fastapi`, `uvicorn`, `yacs`, `prettytable`, `comet_ml`, and `torch_geometric`.
- Source inspection also shows `mcp` imports in the MCP server/client scripts, but that stack is separate from core training.

## Hypotheses

### H1: `pip` is shadowed by `~/.local/bin/pip`
- Supports: `which pip` returns `~/.local/bin/pip` before the env's own `bin/pip`.
- Conflicts: `python -m pip` already points at the env.
- Test: prepend `$CONDA_PREFIX/bin` to `PATH` and re-run `which pip`.

### H2: `torch_geometric` is missing
- Supports: `models.py` imports `GATv2Conv` from `torch_geometric.nn` and the demo logs say it fell back to a dummy implementation.
- Conflicts: the model can still run with the fallback, but not with the intended layer.
- Test: install `torch-geometric` and import `GATv2Conv`.

### H3: DGL is the CPU build, not the CUDA build
- Supports: training failed with `Device API gpu is not enabled. Please install the cuda version of dgl.`
- Conflicts: `torch.cuda.is_available()` is `True`, so the CUDA stack is otherwise available.
- Test: replace the CPU DGL build with the CUDA wheel and rerun a minimal graph-to-GPU operation.

### H4: `mcp` is an optional but unresolved dependency for server/client scripts
- Supports: `mcp_server.py`, `mcp_client_test.py`, and `completion_client.py` import `mcp`.
- Conflicts: core model training does not need MCP.
- Test: try installing `mcp`; if no Python 3.8-compatible wheel exists, treat it as a separate optional stack.

## Experiments

- Demo run with `configs/DrugBAN_Demo.yaml` confirmed the project starts and trains, but exposed the missing `torch_geometric` fallback and CPU-only DGL failure.
- `which pip` confirmed shell resolution favors `~/.local/bin/pip` instead of the env pip.

## Root Cause

- The first environment build was incomplete for this codebase: it used a CPU-only DGL wheel and had no PyTorch Geometric stack, so the demo run fell back to a dummy GATv2 layer and then failed when DGL needed GPU support.
- The shell also resolved `pip` to `~/.local/bin/pip`, so package installs could silently target the wrong location unless `python -m pip` or an env hook was used.
- The MCP tooling is a separate optional stack and requires Python >= 3.10, so it cannot live in the same Python 3.8 training environment.

## Fix

- Kept the training/runtime environment at `Python 3.8.20`.
- Installed and verified:
  - `torch 1.13.1+cu117`
  - `torchvision 0.14.1`
  - `torchaudio 0.13.1`
  - `dgl 1.1.3+cu117`
  - `torch-geometric 2.5.3`
  - `pyg_lib 0.4.0+pt113cu117`
  - `torch_scatter 2.1.1+pt113cu117`
  - `torch_sparse 0.6.17+pt113cu117`
  - `torch_cluster 1.6.1+pt113cu117`
  - `torch_spline_conv 1.2.2+pt113cu117`
  - `nvidia-cusparse-cu11 11.7.5.86`
- Added a conda activation hook so `pip` resolves to the env bin and CUDA wheel libraries are present on `LD_LIBRARY_PATH`.
- Verified:
  - `which pip` -> `/home/k8smaster/anaconda3/envs/GATV2-TransDTI/bin/pip`
  - `python -m pip check` -> `No broken requirements found`
  - `torch_geometric.nn.GATv2Conv` imports successfully
  - DGL can move a graph to CUDA without error
- Remaining boundary:
  - The `mcp` SDK is only available for Python >= 3.10 on official PyPI, so the MCP server/client scripts need a separate environment if you want those to run too.


## Formal Linux GPU Mapping

### Observations

- The formal Linux queue was failing with `torch.OutOfMemoryError` on `GPU 0`.
- The queue launcher had been exporting `CUDA_VISIBLE_DEVICES=1`.
- The server has two physical GPUs, but their visible-device numbering was not the same as the earlier assumption.

### Hypotheses

### H1: `CUDA_VISIBLE_DEVICES=1` selects the A6000
- Supports: this was the earlier assumption.
- Conflicts: the remote probe contradicted it.
- Test: run a remote probe with `CUDA_VISIBLE_DEVICES=1` and inspect the device name.

### H2: `CUDA_VISIBLE_DEVICES=0` selects the A6000
- Supports: remote probe reported `NVIDIA RTX A6000` and ~48 GiB total memory.
- Conflicts: none after the probe.
- Test: run a remote probe with `CUDA_VISIBLE_DEVICES=0` and inspect the device name.

### H3: `backends.py` is still spreading the model across GPUs
- Supports: older code used `device_map="auto"`.
- Conflicts: current `backends.py` now uses explicit single-device mapping for `cuda`.
- Test: inspect the loaded-device branch and verify `device_map={"": self._device}` for CUDA inputs.

### Experiments

- `python old_automation/gpu_visibility_probe.py --visible-device 0`
  - Result: `True`, `1`, `NVIDIA RTX A6000`, `51041271808`
- `python old_automation/gpu_visibility_probe.py --visible-device 1`
  - Result: `True`, `1`, `NVIDIA GeForce RTX 2080 Ti`, `11545280512`
- `rg -n "CUDA_VISIBLE_DEVICES=1|VISIBLE_DEVICE_INDEX|CUDA_VISIBLE_DEVICES=0" old_automation`
  - Result: only `VISIBLE_DEVICE_INDEX = "0"` remains in the queue launcher.

### Root Cause

- The Linux server maps `CUDA_VISIBLE_DEVICES=0` to the A6000 and `CUDA_VISIBLE_DEVICES=1` to the RTX 2080 Ti.
- The earlier OOM happened because the formal queue was launching on the RTX 2080 Ti, not on the A6000.

### Fix

- Keep the formal Linux queue launcher on `CUDA_VISIBLE_DEVICES=0`.
- Keep `backends.py` on explicit single-device CUDA mapping so `cuda` does not fan out across visible GPUs.
- Result: the formal experiment queue targets the A6000 only and does not do two-GPU distributed loading.

## Formal Config Audit

### Observations

- The formal single-node and federated base configs already use `runtime.device: cuda`.
- `run_large_eval` is disabled in the formal single-node and federated base configs.
- `PrE-Text` evaluation helpers still defaulted to `cuda:1` when `runtime.device` was missing.
- `old_automation` already used `CUDA_VISIBLE_DEVICES=0` for remote launches and only polls / advances the queue.

### Fixes

- Updated the stale `CUDA_VISIBLE_DEVICES=1` comment in `thesis_platform/configs/base/llm_7b_linux.yaml`.
- Updated the formal Linux launch example in `thesis_platform/docs/one/单节点版本创新算法完整流程.md` to `CUDA_VISIBLE_DEVICES=0`.
- Changed `PrE-Text` eval defaults in `gpt2_eval.py`, `distilgpt2_eval.py`, and `llama2_eval.py` from `cuda:1` to `cuda`.
- Changed `old_automation/gpu_visibility_probe.py` default visible device from `1` to `0`.
- Updated `old_automation/README.md` so it matches the current behavior: 30-minute polling, manual code sync, queue advancement only.

### Verification

- `thesis_platform/configs/experiments/single_node_formal/_base_single_node_formal.yaml` stays on `runtime.device: cuda` with `run_large_eval: false`.
- `PrE-Text/configs/experiments/single_node_formal/_base_pretext_formal.yaml` stays on `runtime.device: cuda` with `eval_large.enabled: false`.
- `PrE-Text/configs/experiments/federated_formal/_base_federated_formal.yaml` stays on `runtime.device: cuda` with `eval_large.enabled: false`.
- `old_automation/old_experiment_queue.py` now keeps `VISIBLE_DEVICE_INDEX = "0"`.
- `python -m py_compile` passed for the modified Python files.

## Full Smoke Test on RTX 2080 Ti

### Observations

- The `GATV2-TransDTI` environment now resolves `pip` to `/home/k8smaster/anaconda3/envs/GATV2-TransDTI/bin/pip`.
- `python -m pip check` reports `No broken requirements found`.
- `CUDA_VISIBLE_DEVICES=1` on the old server maps to `NVIDIA GeForce RTX 2080 Ti`.
- A full project smoke test was run with:
  - `python main.py --cfg configs/DrugBAN_Demo.yaml --data biosnap --split random`
  - `CUDA_VISIBLE_DEVICES=1`
- The run printed `Running on: cuda:0`, completed training, and finished final test evaluation.

### Experiments

- Remote run log: `/mnt/public/caiqiyue_files/GATV2-TransDTI/result/demo_smoke_2080.log`
- Output directory: `/mnt/public/caiqiyue_files/GATV2-TransDTI/result/demo`
- Generated artifacts:
  - `model_epoch_0.pth`
  - `model_epoch_1.pth`
  - `model_best.pth`
  - `model_architecture.txt`

### Result

- The project now runs end-to-end in the `GATV2-TransDTI` environment on the RTX 2080 Ti without falling back to the dummy `GATv2Conv` implementation.
- The log still shows `torch_geometric` warnings that `pyg-lib` and `torch-sparse` are disabled because the wheel requires `GLIBC_2.29`, but those warnings did not block the run.
- Final test metrics reported by the demo run:
  - `AUROC: 0.6610`
  - `AUPRC: 0.6532`
  - `Accuracy: 0.6037`
  - `F1: 0.6347`

---

# Debug: SP-C1 eval_small device mismatch

## Observations

- The failed experiment is `SP-C1`, started on the old server at `2026-04-21 14:12:26` with PID `8642`, according to `old_automation/old_experiment_queue.log`.
- The local queue state still marks `SP-C1` as `running` with PID `8642`, according to `old_automation/old_experiment_queue_state.json`.
- A direct remote status check at `2026-04-21 14:58:07+08:00` showed no live process for PID `8642`, so the experiment had already exited.
- The remote log `/mnt/public/caiqiyue_file/code_from_paper/old_automation/SP-C1.remote.log` ends with:
  `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`
- The stack trace ends inside `transformers.models.gpt2.modeling_gpt2.GPT2Model.forward`, specifically at the embedding lookup (`self.wte(input_ids)`), which means the failure happens before the loss computation.
- The remote output directory contains `stage1_summary.json`, `stage2_summary.json`, and `metrics_summary.json`, but no `eval_small_summary.json`, so the failure happened inside `run_eval_small`.
- `pretext_platform.evaluation.gpt2_eval.run_gpt2_eval()` moves the model to `device`, and the training loop moves each training batch to `device`, but `evaluate()` calls `model(**batch)` without moving `batch`.
- `pretext_platform.evaluation.distilgpt2_eval.run_distilgpt2_eval()` has the same `evaluate()` pattern: model moved to `device`, training batches moved to `device`, evaluation batches passed as-is.
- The formal pre-text configs for both single-node and federated formal experiments set `eval_small.eval_mode: gpt2`, so the shared `gpt2_eval` path affects official formal runs.

## Hypotheses

### H1: `evaluate()` leaves eval batches on CPU while the model is on CUDA (ROOT HYPOTHESIS)
- Supports: the server error is exactly a CUDA/CPU mismatch during GPT-2 embedding lookup; `gpt2_eval.evaluate()` and `distilgpt2_eval.evaluate()` both call `model(**batch)` without moving `batch`; the training loops in both files do move batches to `device`, so the bug is isolated to evaluation.
- Conflicts: none found so far.
- Test: run a minimal local experiment around `evaluate()` that proves the batch stays on the loader device instead of being moved to the model device.

### H2: the GPT-2 model itself is only partially moved to CUDA
- Supports: mixed-device failures can happen if some submodules remain on CPU.
- Conflicts: `run_gpt2_eval()` explicitly does `model = model.to(device)` before the crash, and the stack trace points to `input_ids` vs embedding device mismatch at the embedding layer, which more strongly suggests CPU inputs against CUDA weights.
- Test: inspect the code path and confirm there is no later reassignment of GPT-2 weights back to CPU after `model.to(device)`.

### H3: the checkpoint or tokenizer path creates CPU-only labels/inputs for eval while training uses a different collation path
- Supports: tokenization creates Python lists that are later converted to tensors by the dataset formatter and DataLoader.
- Conflicts: the same formatting pattern is used for both train and eval datasets, but only the training path moves tensors to `device` before model invocation; the crash occurs before loss calculation, so labels are not the primary trigger.
- Test: compare the train and eval call sites to confirm only eval lacks the device transfer.

## Experiments

### E1: Remote minimal reproduction against the shared eval functions
- Change: no production code change. Ran a synthetic diagnostic in the server `pretext` environment with a fake model that expects `cuda:0`.
- Result: confirmed. Both `gpt2_eval.evaluate()` and `distilgpt2_eval.evaluate()` forwarded `input_ids`, `attention_mask`, and `labels` on `cpu`.
- Evidence:
  - `gpt2_eval RuntimeError batch stayed on {'input_ids': 'cpu', 'attention_mask': 'cpu', 'labels': 'cpu'}, expected cuda:0`
  - `distilgpt2_eval RuntimeError batch stayed on {'input_ids': 'cpu', 'attention_mask': 'cpu', 'labels': 'cpu'}, expected cuda:0`

## Root Cause

`pretext_platform.evaluation.gpt2_eval.evaluate()` and `pretext_platform.evaluation.distilgpt2_eval.evaluate()` passed evaluation batches straight from the DataLoader to the model without moving them onto the model device, so formal pre-text runs on CUDA crashed during GPT-2 embedding lookup with mixed `cuda:0` and `cpu` tensors.

## Fix

- Added `pretext_platform.evaluation.device_utils` with shared `model_device()` and `move_batch_to_model_device()` helpers.
- Updated both `gpt2_eval.evaluate()` and `distilgpt2_eval.evaluate()` to move each eval batch onto the model device before calling `model(**batch)`.
- Added a regression test file `PrE-Text/tests/test_eval_device_transfer.py` that fails if either eval path forwards CPU batch tensors to a CUDA model.
- Synced the fixed evaluation files to the old server copy under `/mnt/public/caiqiyue_file/code_from_paper/PrE-Text/...` so the reset formal experiment will use the patched code.
