# PrE-Text Platform

`PrE-Text` 已重构为一个独立的实验平台包 `pretext_platform/`。  
它保留原论文的核心算法主链：

1. Stage 1: Private Evolution 生成 DP seed texts
2. Stage 2: 基于 LLaMA-2-7B 的 bootstrap 扩增
3. Downstream evaluation:
   - `DistilGPT2`
   - `LLaMA-2-7B + LoRA`

同时，项目结构和配置方式对齐 `thesis_platform`，但运行时不依赖 `thesis_platform` 的 Python 代码，只复用当前工作区中已经准备好的数据集和模型目录。

## 目录结构

```text
PrE-Text/
  pretext_platform/
    algorithms/
    core/
    data/
    evaluation/
    scripts/
  configs/
    base/
    templates/
    experiments/
  tests/
  main.py
  llama_bootstrap.py
  eval_distilgpt2.py
  eval_llama2.py
```

说明：

- `pretext_platform/` 是新的主实现。
- 根目录四个旧脚本仍保留，但已经变成兼容包装器。
- `variation.py`、`similarity.py`、`nn_histogram.py`、`custom_datasets.py` 也已转为兼容导出，统一指向新包实现。

## 默认数据和模型来源

平台默认直接复用当前工作区中已有资源：

- 数据集根目录：`../datasets`
- 模型根目录：`../thesis_platform/open_model`

默认会使用：

- `../datasets/initial_set.json`
- `../datasets/congressional_train.json`
- `../datasets/congressional_eval.json`
- `../datasets/bioarxiv_train.json`
- `../datasets/bioarxiv_eval.json`
- `../thesis_platform/open_model/all_minilm_l6_v2`
- `../thesis_platform/open_model/roberta_large`
- `../thesis_platform/open_model/llama_2_7b_hf`
- `../thesis_platform/open_model/distilgpt2`

注意：

- `DistilGPT2` 下游评测严格要求显式提供 `c4_checkpoint.pth`。
- 当前仓库默认配置把它写成 `./c4_checkpoint.pth`。
- 如果该文件不存在，`eval_small` 会直接失败，不会自动退化到无 warm-start 模式。

## 配置驱动运行

推荐入口：

```bash
python -m pretext_platform.scripts.run_pipeline --config configs/experiments/smoke_congressional_eps758.yaml
```

分阶段入口：

```bash
python -m pretext_platform.scripts.run_stage1 --config configs/experiments/smoke_congressional_eps758.yaml
python -m pretext_platform.scripts.run_bootstrap --config configs/experiments/smoke_congressional_eps758.yaml
python -m pretext_platform.scripts.run_eval_large --config configs/experiments/smoke_congressional_eps758.yaml
python -m pretext_platform.scripts.run_eval_small --config configs/experiments/full_congressional_eps129.yaml
```

## 内置实验配置

当前内置 4 套配置：

- `configs/experiments/smoke_congressional_eps758.yaml`
- `configs/experiments/full_congressional_eps129.yaml`
- `configs/experiments/smoke_bioarxiv_eps758.yaml`
- `configs/experiments/full_bioarxiv_eps129.yaml`

设计约定：

- `smoke_*`：轻量验证配置，默认关闭 `eval_small`
- `full_*`：更接近原论文默认超参的完整配置
- `eps758` 对应 `sigma=2.31`
- `eps129` 对应 `sigma=11.3`

## 输出结构

所有结果都写入：

```text
outputs/pretext_platform/<experiment_id>/
```

实验目录下分为：

- `stage1/`
- `stage2/`
- `eval_small/`
- `eval_large/`

其中保留原论文关键产物命名：

- `stage1/private_embeds.npy`
- `stage1/generated_text_it*.json`
- `stage1/surviving_text_it*.json`
- `stage2/llama7b_text_syn.json`

同时新增平台摘要文件：

- `resolved_config.json`
- `stage1_summary.json`
- `stage2_summary.json`
- `eval_small_summary.json`
- `eval_large_summary.json`
- `metrics_summary.json`

## 旧命令兼容

以下旧命令仍可使用：

```bash
python main.py ...
python llama_bootstrap.py ...
python eval_distilgpt2.py ...
python eval_llama2.py ...
```

它们现在只负责：

1. 解析旧参数
2. 映射成新配置
3. 调用 `pretext_platform` 中对应 stage

## 测试

运行：

```bash
python -m unittest discover -s tests -p "test_*.py"
```

测试覆盖：

- YAML 继承与路径解析
- 训练/评测数据双格式加载
- Stage 1 编排与产物命名
- pipeline 输出摘要
- legacy CLI 映射
- `eval_small` 缺失 checkpoint 的显式失败

## 环境依赖

保留当前仓库的 `requirements.txt`。  
核心依赖仍然是：

- `torch`
- `transformers`
- `accelerate`
- `sentence-transformers`
- `faiss-cpu`
- `opacus`
- `vllm`
- `datasets`
- `peft`
- `PyYAML`

## 引用

如果你在研究中使用 PrE-Text，请继续引用原论文：

```bibtex
@misc{hou2024pretext,
      title={PrE-Text: Training Language Models on Private Federated Data in the Age of LLMs},
      author={Charlie Hou and Akshat Shrivastava and Hongyuan Zhan and Rylan Conway and Trang Le and Adithya Sagar and Giulia Fanti and Daniel Lazar},
      year={2024},
      eprint={2406.02958},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
