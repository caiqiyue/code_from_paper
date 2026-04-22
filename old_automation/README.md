# Old Server Automation

This folder is versioned in git.

Queued experiments:

| Label | Config | Conda env |
| --- | --- | --- |
| SN-C1 | `thesis_platform/configs/experiments/single_node_formal/sn_c1_jobs_base.yaml` | `caiqiyue-vllm` |
| SP-C1 | `PrE-Text/configs/experiments/single_node_formal/sp_c1_jobs_base.yaml` | `pretext` |
| SN-C2 | `thesis_platform/configs/experiments/single_node_formal/sn_c2_congressional_base.yaml` | `caiqiyue-vllm` |
| SP-C2 | `PrE-Text/configs/experiments/single_node_formal/sp_c2_congressional_base.yaml` | `pretext` |
| SN-C3 | `thesis_platform/configs/experiments/single_node_formal/sn_c3_forums_base.yaml` | `caiqiyue-vllm` |
| SP-C3 | `PrE-Text/configs/experiments/single_node_formal/sp_c3_forums_base.yaml` | `pretext` |
| SN-C4 | `thesis_platform/configs/experiments/single_node_formal/sn_c4_microblog_base.yaml` | `caiqiyue-vllm` |
| SP-C4 | `PrE-Text/configs/experiments/single_node_formal/sp_c4_microblog_base.yaml` | `pretext` |
| SN-C5 | `thesis_platform/configs/experiments/single_node_formal/sn_c5_jobs_eps05.yaml` | `caiqiyue-vllm` |
| SP-C5 | `PrE-Text/configs/experiments/single_node_formal/sp_c5_jobs_eps05.yaml` | `pretext` |
| SN-C6 | `thesis_platform/configs/experiments/single_node_formal/sn_c6_jobs_eps758.yaml` | `caiqiyue-vllm` |
| SP-C6 | `PrE-Text/configs/experiments/single_node_formal/sp_c6_jobs_eps758.yaml` | `pretext` |
| SN-C7 | `thesis_platform/configs/experiments/single_node_formal/sn_c7_jobs_no_privacy.yaml` | `caiqiyue-vllm` |
| SP-C7 | `PrE-Text/configs/experiments/single_node_formal/sp_c7_jobs_no_privacy.yaml` | `pretext` |
| SN-C8 | `thesis_platform/configs/experiments/single_node_formal/sn_c8_jobs_seed123.yaml` | `caiqiyue-vllm` |
| SP-C8 | `PrE-Text/configs/experiments/single_node_formal/sp_c8_jobs_seed123.yaml` | `pretext` |
| SN-C9 | `thesis_platform/configs/experiments/single_node_formal/sn_c9_jobs_seed456.yaml` | `caiqiyue-vllm` |
| SP-C9 | `PrE-Text/configs/experiments/single_node_formal/sp_c9_jobs_seed456.yaml` | `pretext` |

Execution order:

1. `SN-C1`
2. `SP-C1`
3. `SN-C2`
4. `SP-C2`
5. `SN-C3`
6. `SP-C3`
7. `SN-C4`
8. `SP-C4`
9. `SN-C5`
10. `SP-C5`
11. `SN-C6`
12. `SP-C6`
13. `SN-C7`
14. `SP-C7`
15. `SN-C8`
16. `SP-C8`
17. `SN-C9`
18. `SP-C9`

Run once to start or advance the queue:

```powershell
.\old_automation\run_old_experiment_queue.ps1
```

The script starts only one remote experiment at a time and polls the server every 30 minutes to advance the queue. It does not sync code; code updates are handled manually. The remote launch uses `caiqiyue-vllm` for `SN-*` and `pretext` for `SP-*`, and binds the old server to the physical `A6000` with `CUDA_DEVICE_ORDER=PCI_BUS_ID` plus `CUDA_VISIBLE_DEVICES=1`. For `SP-*`, it runs `run_pipeline` first and then `run_eval_small` as a separate step.
