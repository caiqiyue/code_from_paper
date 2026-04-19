# 联邦编排版 PrE-Text 设计方案

日期：2026-04-16
范围：`PrE-Text/pretext_platform`
状态：设计确认版

## 1. 背景与目标

当前仓库中的 `PrE-Text` 主要提供单节点执行链路：

- `stage1`
- `bootstrap`
- `stage2`
- 可选下游评测

它已经具备一定的联邦数据语义基础，例如：

- 能读取 client-bucketed 数据
- 保留 `max_samples_per_client` 等字段

但它还没有联邦实验编排层，缺少：

- 多客户端分区驱动
- 外层联邦轮次编排
- 联邦实验级 artifact
- 联邦实验级 privacy ledger

本设计的目标不是重构现有单节点 `PrE-Text`，而是**在 `PrE-Text/pretext_platform` 内新增一条联邦架构版本的 PrE-Text 实验线**。

核心原则：

- 保持 `PrE-Text` 的 `stage1/bootstrap/stage2` 算法本体不变
- 只新增联邦编排外壳
- 第一版目标是联邦生成，不做联邦下游评测
- 最终仍然只产出一份全局合成语料，供现有单机下游评测复用

## 2. 设计边界

### 2.1 本设计必须做到

1. 在 `PrE-Text/pretext_platform` 内新增 federated runner
2. 完全复用联邦创新算法的数据分区规则
3. 客户端每轮只执行 `stage1`
4. 服务端每轮统一执行 `bootstrap -> stage2`
5. 客户端每轮只上传 `surviving texts`
6. 每个 federated round 产出一份全局合成语料
7. 实验结束后只保留最后一轮全局合成语料作为最终语料
8. 保留 `PrE-Text` 原生隐私口径，同时新增联邦实验级 `privacy_ledger.json`

### 2.2 本设计明确不做

1. 不修改单节点 `PrE-Text` 主流程
2. 不重写 `stage1/bootstrap/stage2` 内部算法
3. 不做联邦下游任务评测
4. 不做个性化生成
5. 不做路由、原型、聚类、服务端记忆等联邦创新算法机制
6. 第一版不做 checkpoint / resume / failure recovery

## 3. 总体架构

联邦版 `PrE-Text` 的结构定义如下：

- **客户端侧**
  - 每个客户端持有固定私有文本分区
  - 每个 federated round 只运行一次 `stage1`
  - 本轮只向服务端贡献 surviving texts

- **服务端侧**
  - 汇总全部客户端 surviving texts
  - 作为全局 seed 输入一次 `bootstrap`
  - 随后统一执行一次 `stage2`
  - 生成本轮全局合成语料

- **实验结束**
  - 只使用最后一轮的全局合成语料
  - 下游小模型评测继续沿用当前单机评测口径

### 3.1 外层联邦轮次定义

一个 federated round 固定等于一次完整服务端生成周期：

1. 客户端分别执行一次 `stage1`
2. 服务端收集 surviving texts
3. 服务端执行一次全局 `bootstrap -> stage2`
4. 产出本轮全局合成语料

### 3.2 上传对象定义

客户端每轮上传内容固定为：

- **只上传每轮 surviving texts**

第一版不上传完整 stage1 目录，也不把中间生成文本全部上传到服务端。

## 4. 推荐文件落点

### 4.1 新增核心模块

建议新增以下文件：

- `PrE-Text/pretext_platform/core/federated_runner.py`
  - 联邦编排主入口
  - 负责 round loop、客户端 stage1 调度、服务端 bootstrap/stage2 调度、summary 写出

- `PrE-Text/pretext_platform/core/federated_partition.py`
  - 联邦数据分区适配层
  - 负责复用联邦创新算法分区规则
  - 负责把原始文本转成 `PrE-Text` 需要的 per-client 数据输入

- `PrE-Text/pretext_platform/core/federated_artifacts.py`
  - round artifact、experiment artifact 写出
  - 包含 `metrics_summary.json`、per-round 目录、客户端 stage1 结果、服务端 stage2 结果的落盘逻辑

- `PrE-Text/pretext_platform/core/federated_privacy.py`
  - 联邦实验级 `privacy_ledger.json` 汇总
  - 不替换 `PrE-Text` 原生隐私计算，只做联邦实验层记录

### 4.2 新增脚本入口

建议新增：

- `PrE-Text/pretext_platform/scripts/run_federated.py`

用途：

- 显式运行联邦版 `PrE-Text` 配置
- 便于早期调试

同时推荐后续把 `pretext_platform.core.pipeline.run_pipeline()` 扩展为按 `execution.mode` 分发：

- `single_node` 继续走当前单节点链路
- `federated_pretext` 走新的 `FederatedPretextRunner`

### 4.3 保持不动的现有模块

以下文件仅被复用，不作为本轮重写目标：

- `PrE-Text/pretext_platform/algorithms/stage1.py`
- `PrE-Text/pretext_platform/algorithms/bootstrap.py`
- `PrE-Text/pretext_platform/core/pipeline.py` 中现有单节点流程
- `PrE-Text/pretext_platform/evaluation/*`

## 5. 配置字段设计

### 5.1 新增顶层 execution 区块

推荐新增：

```yaml
execution:
  mode: federated_pretext
```

合法值：

- `single_node`
- `federated_pretext`

说明：

- 单节点 `PrE-Text` 继续使用 `single_node`
- 联邦版 `PrE-Text` 使用 `federated_pretext`

### 5.2 新增 federation 区块

推荐新增：

```yaml
federation:
  rounds: 10
  num_clients: 16
  max_samples_per_client: 16
  validation_ratio: 0.25
  partition_strategy: preserve_buckets
```

字段语义要求与联邦创新算法保持一致：

- `rounds`
  - federated round 数量
- `num_clients`
  - 客户端数量
- `max_samples_per_client`
  - 每个客户端最多使用的私有样本数
- `validation_ratio`
  - 分区后每个客户端本地验证占比
- `partition_strategy`
  - **完全复用联邦创新算法的分区规则**

### 5.3 现有区块的联邦用法

以下区块继续保留，但语义变成“联邦版中由 runner 在不同层调用”：

- `stage1`
  - 客户端本地使用
- `bootstrap`
  - 服务端全局使用
- `eval_small`
  - 不在联邦 runner 内执行，但保留以供后续单机评测复用
- `eval_large`
  - 同上

### 5.4 隐私字段

不新增一套新的算法隐私字段。

联邦版做法是：

- 客户端 `stage1` 继续使用 `PrE-Text` 现有隐私计算逻辑
- 联邦 runner 从每个客户端 `stage1_summary` 中提取关键字段
- 汇总成实验级 `privacy_ledger.json`

## 6. 每轮数据流

每个 federated round 固定为以下步骤：

1. 从固定客户端分区中取出每个客户端的本地数据
2. 客户端各自执行一次 `stage1`
3. 每个客户端提取本轮 surviving texts
4. 服务端合并全部 surviving texts
5. 服务端执行一次全局 `bootstrap`
6. 服务端执行一次全局 `stage2`
7. 写出本轮 round artifacts
8. 更新实验级 `privacy_ledger.json`

### 6.1 客户端输出

每个客户端 round 目录至少包含：

- `stage1_summary.json`
- `surviving_text_it*.json`

### 6.2 服务端输出

每个 round 的服务端目录至少包含：

- `server_stage2/llama7b_text_syn.json`
- `server_stage2/stage2_summary.json`

### 6.3 最终语料规则

实验结束后：

- **只取最后一轮的全局 `llama7b_text_syn.json`**
- 作为后续下游小模型评测输入

这条规则必须和联邦创新算法当前“最终轮产物进入下游评测”的口径一致。

## 7. 实验产物设计

### 7.1 推荐输出目录结构

推荐沿用 `PrE-Text` 当前输出根目录，不新增新的输出系统：

- `PrE-Text/outputs/pretext_platform/<experiment_id>/`

内部结构：

```text
<experiment_id>/
  resolved_config.json
  metrics_summary.json
  privacy_ledger.json
  round_000/
    client_000/
      stage1_summary.json
      surviving_text_it0.json
      ...
    client_001/
      ...
    server_stage2/
      llama7b_text_syn.json
      stage2_summary.json
  round_001/
    ...
```

### 7.2 metrics_summary.json

推荐包含：

- `experiment_id`
- `experiment_dir`
- `status`
- `round_count`
- `completed_rounds`
- `final_synthetic_corpus_path`
- `final_synthetic_sample_count`
- `round_summaries`
- `privacy_summary`

### 7.3 privacy_ledger.json

推荐包含：

- `schema_version`
- `experiment_id`
- `rounds`
- 每轮客户端记录
- 每轮服务端汇总 surviving texts 数量
- 每轮全局 stage2 产出数量
- 最终汇总统计

每个 round 至少记录：

- `round_id`
- `participating_clients`
- `client_stage1_stats`
- `merged_surviving_count`
- `server_stage2_sample_count`

## 8. 实验配置命名规则

### 8.1 配置目录

推荐新建：

- `PrE-Text/configs/experiments/federated/`

### 8.2 命名规则

推荐使用统一前缀：

- `fpt_` = federated pretext

例如：

- `fpt_jobs_eps129.yaml`
- `fpt_jobs_eps05.yaml`
- `fpt_jobs_eps758.yaml`
- `fpt_jobs_no_privacy.yaml`
- `fpt_congressional_eps129.yaml`
- `fpt_forums_eps129.yaml`
- `fpt_microblog_eps129.yaml`

### 8.3 tiny / validate / smoke 命名

推荐同步准备一套最小链路配置：

- `fpt_jobs_tiny.yaml`
- `fpt_jobs_validate.yaml`

用于：

- 联邦链路连通性验证
- artifact 完整性检查

## 9. 文档输出位置

### 9.1 本设计文档

固定放在：

- `PrE-Text/docs/fed/2026-04-16-federated-pretext-design.md`

### 9.2 后续实现计划文档

建议放在：

- `PrE-Text/docs/fed/2026-04-16-federated-pretext-implementation-plan.md`

### 9.3 后续正式实验设计文档

建议放在：

- `PrE-Text/docs/fed/联邦版PrE-Text正式实验设计.md`

### 9.4 后续实验记录表

建议放在：

- `PrE-Text/docs/fed/联邦版PrE-Text实验记录表.md`

## 10. 第一版最小实现范围

第一版只要求：

1. 联邦 tiny 配置能跑通
2. 多客户端分区生效
3. 每轮客户端 `stage1` 能独立执行
4. 服务端每轮能成功执行一次全局 `bootstrap -> stage2`
5. 能写出完整 round artifacts
6. 能写出联邦实验级 `metrics_summary.json`
7. 能写出联邦实验级 `privacy_ledger.json`
8. 最终最后一轮语料能被现有单机评测脚本直接读取

## 11. 测试与验收标准

### 11.1 配置验收

- 新 federated YAML 能成功加载
- `execution.mode = federated_pretext`
- `federation.*` 字段齐全

### 11.2 流程验收

- 一个 tiny federated 配置可完整跑通：
  - 分区
  - 多客户端 stage1
  - surviving texts 合并
  - 服务端 bootstrap
  - 服务端 stage2

### 11.3 产物验收

- 每个 round 都存在：
  - 客户端 stage1 目录
  - 服务端 stage2 目录
- 实验根目录存在：
  - `resolved_config.json`
  - `metrics_summary.json`
  - `privacy_ledger.json`

### 11.4 一致性验收

- 分区规则与联邦创新算法一致
- 最终语料只取最后一轮
- 现有下游评测入口能直接读取最后一轮 `llama7b_text_syn.json`

## 12. 总结

本设计给 `PrE-Text` 增加的是一个**联邦编排外壳**，而不是新的算法本体。

最终形态是：

- 客户端：只跑 `stage1`
- 服务端：统一跑 `bootstrap -> stage2`
- 联邦 round：每轮完成一次完整服务端生成周期
- 最终语料：只取最后一轮
- 隐私记录：保留 `PrE-Text` 原生口径，并新增联邦实验级 ledger

这样可以最大限度保持 `PrE-Text` 的“血统”，同时让它具备和联邦创新算法做公平实验对照的联邦实验外壳。
