# Old Server Automation

This folder is versioned in git.

Queued experiments:

| Label | Config | Conda env |
| --- | --- | --- |
| NS-C1 | `paper-new/configs/experiments/single_node_formal/ns_c1_jobs_base.yaml` | `pretext` |
| SP-C1 | `PrE-Text/configs/experiments/single_node_formal/sp_c1_jobs_base.yaml` | `pretext` |
| NS-C2 | `paper-new/configs/experiments/single_node_formal/ns_c2_congressional_base.yaml` | `pretext` |
| SP-C2 | `PrE-Text/configs/experiments/single_node_formal/sp_c2_congressional_base.yaml` | `pretext` |
| NS-C3 | `paper-new/configs/experiments/single_node_formal/ns_c3_forums_base.yaml` | `pretext` |
| SP-C3 | `PrE-Text/configs/experiments/single_node_formal/sp_c3_forums_base.yaml` | `pretext` |
| NS-C4 | `paper-new/configs/experiments/single_node_formal/ns_c4_microblog_base.yaml` | `pretext` |
| SP-C4 | `PrE-Text/configs/experiments/single_node_formal/sp_c4_microblog_base.yaml` | `pretext` |
| NS-C5 | `paper-new/configs/experiments/single_node_formal/ns_c5_jobs_eps05.yaml` | `pretext` |
| SP-C5 | `PrE-Text/configs/experiments/single_node_formal/sp_c5_jobs_eps05.yaml` | `pretext` |
| NS-C6 | `paper-new/configs/experiments/single_node_formal/ns_c6_jobs_eps758.yaml` | `pretext` |
| SP-C6 | `PrE-Text/configs/experiments/single_node_formal/sp_c6_jobs_eps758.yaml` | `pretext` |
| NS-C7 | `paper-new/configs/experiments/single_node_formal/ns_c7_jobs_no_privacy.yaml` | `pretext` |
| SP-C7 | `PrE-Text/configs/experiments/single_node_formal/sp_c7_jobs_no_privacy.yaml` | `pretext` |
| NS-C8 | `paper-new/configs/experiments/single_node_formal/ns_c8_jobs_seed123.yaml` | `pretext` |
| SP-C8 | `PrE-Text/configs/experiments/single_node_formal/sp_c8_jobs_seed123.yaml` | `pretext` |
| NS-C9 | `paper-new/configs/experiments/single_node_formal/ns_c9_jobs_seed456.yaml` | `pretext` |
| SP-C9 | `PrE-Text/configs/experiments/single_node_formal/sp_c9_jobs_seed456.yaml` | `pretext` |

Execution order:

1. `NS-C1`
2. `SP-C1`
3. `NS-C2`
4. `SP-C2`
5. `NS-C3`
6. `SP-C3`
7. `NS-C4`
8. `SP-C4`
9. `NS-C5`
10. `SP-C5`
11. `NS-C6`
12. `SP-C6`
13. `NS-C7`
14. `SP-C7`
15. `NS-C8`
16. `SP-C8`
17. `NS-C9`
18. `SP-C9`

Run once to start or advance the queue:

```powershell
.\old_automation\run_old_experiment_queue.ps1
```

The script starts only one remote experiment at a time and polls the server every 30 minutes to advance the queue. It does not sync code; code updates are handled manually. The remote launch uses `pretext` for both `NS-*` and `SP-*`, and binds the old server to the physical `A6000` with `CUDA_DEVICE_ORDER=PCI_BUS_ID` plus `CUDA_VISIBLE_DEVICES=1`. For `SP-*`, the formal config now enables `eval_small` inside `run_pipeline`, so the queue launches a single pipeline command instead of a separate `run_eval_small` step.
