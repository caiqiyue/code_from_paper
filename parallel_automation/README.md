# Parallel Automation

This folder is tracked by git.

Queued experiments:

| Label | Config | Conda env |
| --- | --- | --- |
| SP-C6 | `PrE-Text/configs/experiments/single_node_formal/sp_c6_jobs_eps758.yaml` | `pretext` |
| SP-C7 | `PrE-Text/configs/experiments/single_node_formal/sp_c7_jobs_no_privacy.yaml` | `pretext` |
| SP-C8 | `PrE-Text/configs/experiments/single_node_formal/sp_c8_jobs_seed123.yaml` | `pretext` |
| SP-C9 | `PrE-Text/configs/experiments/single_node_formal/sp_c9_jobs_seed456.yaml` | `pretext` |

Execution order:

1. `SP-C6`
2. `SP-C7`
3. `SP-C8`
4. `SP-C9`

Run once to start or advance the queue:

```powershell
.\parallel_automation\run_parallel_experiment_queue.ps1
```

The script starts only one remote experiment at a time and polls the server every 30 minutes to advance the queue. It does not sync code; code updates are handled manually. The remote launch uses `pretext` for all queued experiments, and binds the old server to the physical `A6000` with `CUDA_DEVICE_ORDER=PCI_BUS_ID` plus `CUDA_VISIBLE_DEVICES=1`. For each `SP-*`, it runs `run_pipeline` first and then `run_eval_small` as a separate step.
