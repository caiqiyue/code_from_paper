# Findings

## Initial Structure
- 平台主目录包含 `core`、`adapters`、`algorithms`、`configs`、`evaluation`、`scripts`、`tests`、`docs`。
- 目录名显示该平台是“配置驱动 + 插件式方法组合”的实验框架，而非单一固定算法实现。

## Entry Points
- `scripts/run_experiment.py` 直接调用 `core/pipeline.py::run_pipeline`，是联邦/多轮真实实验主入口。
- `core/experiment_runner.py` 是真实实验总编排器，负责 preflight、数据划分、轮次调度、checkpoint、privacy ledger、downstream eval、cross-domain eval。
- `scripts/run_single_node.py` 调用 `core/single_node_runner.py::SingleNodeRunner`，是单节点 fine 分支入口。
- `SingleNodeRunner` 明确采用 Stage A -> Stage B -> Evaluation 的单节点流程；其 Stage A 本质是“生成-打分-选坏样本-检索锚点-批评-聚合-改 prompt”的迭代优化。

## README-Level Method Framing
- README 将当前“创新算法第三版主链”描述为：
  `global/cluster generation -> real scoring -> bad-sample selection -> anchor retrieval -> critique -> prototype extraction -> clustering + aggregation -> prompt update -> downstream large-eval`
- README 指向的当前唯一正式 v3 主线配置是 `configs/experiments/v3/jobs_real_datainf_v3.yaml`。
- README 中列出的默认方法组合为：
  `generator=pretext_prompt_llm`
  `scorer=datainf_real`
  `retriever=knn`
  `critic=fedtextgrad_llm`
  `aggregator=dbscan_attn_tsgdm`
  `prototype=minilm_mean`
  `routing=personalized_v3`
  `privacy=jobs_eps129`
  `downstream_eval=pretext_large`

## Config And Orchestration
- `core/config.py` 通过 `inherits` 递归合并 YAML，说明“一个实验方法”通常不是单文件，而是多个 generator/scorer/retriever/critic/aggregator/prototype/routing/privacy/downstream_eval 片段的拼装结果。
- `core/pipeline.py` 根据 `execution.mode` 在 federated 与 single_node 两条执行链之间切换。
- `configs/experiments/v3/jobs_real_datainf_v3.yaml` 只定义了少量顶层超参，真正的方法主体来自一串 method config 继承，因此当前“创新算法”是一个配置组合体。
- 该 v3 正式配置的关键设定：
  `num_clients=16`
  `max_samples_per_client=8`
  `generated_per_round=24`
  `rounds=3`
  `top_k_bad=2`
  `aggregator.cluster_eps=0.35`
  `aggregator.cluster_min_samples=2`
  `aggregator.momentum_beta=0.7`
- `configs/experiments/single_node_formal/_base_single_node_formal.yaml` 则把实验压缩成：
  `execution.mode=single_node`
  `stage_a` 迭代 prompt 优化
  `stage_b` 大规模合成
  `downstream_eval.run_small_eval=true`
  `privacy.enabled=false`
  这说明单节点 formal 不是完整 v3，只是抽取了其中部分模块做离线对照。

## Round-Level Flow
- `core/round_runner.py` 展示了真实联邦每轮的核心顺序：
  1. 服务器按 global prompt 生成全局样本池
  2. 若 routing 开启，再按 cluster prompt 生成 cluster pool
  3. 按 `personalized_mix_ratio` 给每个 client 分配 global/cluster 混合候选
  4. client 本地 scorer 打分并选 bad samples
  5. retriever 找 real anchors
  6. critic 生成 critiques
  7. 若 routing 开启，从 real anchors 提 prototype
  8. server aggregator 聚合 critiques/prototypes，形成 cluster assignments、cluster prompts 与新的 prompt update
  9. 记录 privacy ledger、round metrics、routing summary、各类中间产物

## Key Method Modules
- `adapters/generators/pretext_prompt_generator.py`
  生成器本身较薄，只是把当前 prompt 与 seed exemplars 组装后调用 LLM；它不是创新核心，更像承载 prompt update 的执行器。
- `adapters/scorers/datainf_real_scorer.py`
  用真实 transformer feature encoder 编码 train/val/synthetic 文本，调用 `algorithms/scorers/datainf_core.py` 计算 influence，再叠加 `domain_gap`；这是当前优于纯启发式打分的重要来源之一。
- `adapters/critics/fedtextgrad_llm.py`
  只是 LLM critic 包装层，真正逻辑在 `algorithms/critics/fedtextgrad_core.py`；模块角色是把 bad sample 与 real anchor 差异转成文本规则。
- `adapters/aggregators/dbscan_attn_tsgdm.py`
  通过 `algorithms/aggregators/dbscan_core.py` 做 critique 聚类、记忆动量、原型聚类与 cluster prompt 生成；这是“第三版方法感”最强的模块之一。
- `algorithms/prototypes/minilm_mean.py`
  从检索到的 real anchors 上提客户端 prototype，并可经 `PrivacyLedger` 做 DP 扰动；这是路由/个性化链路的输入，不是独立方法闭环。
- `evaluation/downstream_eval.py`
  不是新算法本体，而是把导出的 synthetic corpus 直接送入 `PrE-Text/pretext_platform` 的 small/large eval，说明平台最终收益仍通过 PrE-Text 评测链路兑现。

## Method Coupling Signals
- `datainf_real` 的分数 = `influence_score + domain_gap`，收益并不只来自“DataInf”本身，而是同时混入了分布接近性项。
- `dbscan_attn_tsgdm` 不仅聚合 critiques，还显式使用：
  `memory`
  `prototype_feedbacks`
  `personalized_mix_ratio`
  `prototype_cluster_method`
  所以它不是单纯“换一个聚合器”，而是在吃多路上游信号。
- `minilm_mean` 与 DP 是绑在一起设计的，说明 prototype 的收益、路由效果和隐私扰动在正式主线里并未严格解耦。
