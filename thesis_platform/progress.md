# Progress

## 2026-04-23
- 初始化本次调研的持久化计划文件。
- 已确认项目顶层结构，下一步读取 README、主运行器和正式实验配置。
- 已确认真实实验主入口是 `run_experiment.py -> core/pipeline.py -> core/experiment_runner.py`，单节点入口是 `run_single_node.py -> core/single_node_runner.py`。
- 已确认 README 把 `jobs_real_datainf_v3.yaml` 视为当前正式 v3 算法主线。
- 已确认配置系统是 `inherits` 递归拼装，当前所谓“创新算法”并非单模块，而是多模块组合。
- 已确认单节点 formal 只覆盖 Stage A/B + small eval，不等于完整联邦 v3 主线。
- 已定位关键方法模块：`pretext_prompt_generator`、`datainf_real_scorer`、`fedtextgrad_llm`、`dbscan_attn_tsgdm`、`minilm_mean`、`downstream_eval`。
- 已初步判断收益来源是多模块共振而不是单一创新点。
- 已确认单节点 formal 对比主要围绕 scorer 与 aggregator 变体展开，但评测仍通过 `PrE-Text` small eval 排序，且主线收益解释仍被多模块耦合遮蔽。
- 已形成最终判断：适合收缩成 PrE-Text 增量补丁的是 `real scorer`、更稳的 critique 聚合、少量 prompt 优化控制；不适合直接搬运的是联邦 routing/prototype/privacy/downstream orchestration 整套联动。
