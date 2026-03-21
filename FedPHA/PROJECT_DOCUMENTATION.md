# FedPHA 项目文档

## 项目概述

**FedPHA: Federated Prompt Learning for Heterogeneous Client Adaptation**

这是 ICML 2025 论文的实现，专注于使用 CLIP (Contrastive Language-Image Pre-training) 的提示学习来解决异构客户端联邦学习问题。

**论文引用:**
```
@inproceedings{fedpha2025,
  title={FedPHA: Federated Prompt Learning for Heterogeneous Client Adaptation},
  author={Fang, Chengying and Huang, Wenke and Wan, Guancheng and Yang, Yihao and Ye, Mang},
  booktitle={Forty-second International Conference on Machine Learning}
  year={2025}
}
```

---

## 项目结构

```
FedPHA/
├── .git/                          # Git 版本控制
├── Dassl/                         # DASSL 训练框架
│   └── dassl/
│       ├── config/                # 配置系统 (yacs)
│       ├── data/                  # 数据处理、变换
│       ├── engine/                # 训练引擎
│       ├── evaluation/            # 评估器
│       ├── metrics/               # 准确率、距离度量
│       ├── modeling/              # 骨干网络 (ViT, ResNet)
│       ├── optim/                 # 优化器、学习率调度
│       └── utils/                 # 工具 (日志、计数器)
├── clip/                          # CLIP 模型实现
│   ├── clip.py                    # CLIP 加载/下载
│   ├── model.py                   # CLIP 模型架构
│   └── simple_tokenizer.py        # 分词器
├── configs/                       # 配置文件
│   └── datasets/                  # 数据集配置 (yaml)
├── datasets/                      # 数据集工具
├── scripts/                       # 运行脚本
├── trainers/                      # 训练器实现
├── utils/                         # 工具函数
├── federated_main.py              # 主入口
├── FedPHA-pipeline.jpg            # 流程图
├── README.md                      # 文档
└── requirements.txt               # 依赖
```

---

## 模块功能详解

### 1. 主入口 (federated_main.py)

**位置:** `FederatedPHA/federated_main.py`

**功能:**
- 加载并合并配置（数据集配置 yaml + 命令行参数 + 默认配置）
- 实现联邦学习训练循环（全局轮次迭代、客户端本地训练、权重聚合）
- 支持多种训练模式切换

**关键参数:**
| 参数 | 说明 |
|------|------|
| `--trainer` | 训练器名称 (GL_SVDMSE, PROMPTFL, etc.) |
| `--dataset` | 数据集名称 (caltech101, cifar10, etc.) |
| `--num_users` | 联邦客户端数量 |
| `--num_shots` | 小样本学习设置 |
| `--n_ctx` | 提示向量数量 (prompt length) |
| `--alpha` | push_loss 间隔参数 |
| `--ratio` | SVD 零空间投影比例 |
| `--iid` | IID vs non-IID 数据分布 |
| `--beta` | Dirichlet 非均匀分布参数 |
| `--specify` | 异构提示长度启用 |
| `--prompts_lens` | 各客户端提示长度列表 |

### 2. 配置系统 (configs/datasets/)

**位置:** `configs/datasets/`

**可用数据集配置:**
- `caltech101.yaml`
- `cifar10.yaml` - 包含100个客户端的 USER_PROMPT_LENGTHS
- `cifar100.yaml`
- `domainnet.yaml`
- `dtd.yaml` (可描述纹理)
- `food101.yaml`
- `oxford_flowers.yaml`
- `oxford_pets.yaml`
- `office31.yaml`
- `officehome.yaml`
- `pacs.yaml`
- `office.yaml`

**配置结构示例:**
```yaml
DATASET:
  NAME: "Caltech101"

DATALOADER:
  TRAIN_X:
    BATCH_SIZE: 32
  TEST:
    BATCH_SIZE: 100
  NUM_WORKERS: 8

INPUT:
  SIZE: (224, 224)
  INTERPOLATION: "bicubic"
  PIXEL_MEAN: [0.48145466, 0.4578275, 0.40821073]
  PIXEL_STD: [0.26862954, 0.26130258, 0.27577711]
  TRANSFORMS: ["random_resized_crop", "random_flip", "normalize"]

OPTIM:
  NAME: "sgd"
  LR: 0.001
  ROUND: 50
  MAX_EPOCH: 1
```

### 3. 训练器 (trainers/)

#### 3.1 GL_SVDMSE.py (FedPHA 核心)
**核心实现:**
- **Global Context (`ctx_global`)**: 跨客户端聚合的可学习提示
- **Local Context (`ctx_local`)**: 客户端本地的提示，保持私有
- **SVD 正交投影**: 计算全局提示的零空间来投影局部特征

**损失函数:**
1. **pull_loss**: MSE(局部特征, 投影后的局部特征) - 保持局部接近其投影
2. **push_loss**: ReLU(alpha - ||局部特征 - 全局特征||) - 推动局部远离全局
3. **Cross-Entropy**: 标准分类损失

**关键算法:**
```python
# 通过 SVD 计算零空间
V2 = compute_null_space(ctx_global, ratio=0.8)
# 通过零空间投影局部上下文
projected_ctx_local = ctx_local @ (V2 @ V2.T)
```

#### 3.2 GL_SVDMSE_HE.py (异构扩展)
- 支持**不同提示长度**的客户端
- 全局提示固定长度，本地提示可变
- padding 到最大长度 (77 tokens)

#### 3.3 PROMPTFL.py
基线方法：单一全局提示，无本地/全局分离

#### 3.4 GLP_OT.py
- 使用**最优传输** (Sinkhorn/COT) 聚合
- 仅通过 OT 聚合全局提示

#### 3.5 FEDPGP.py
- 高斯提示参数化: `sigma, U, V` 分解
- 只聚合 sigma (全局)，U,V 保持本地

#### 3.6 CLIP.py
无联邦学习的 CLIP 基线对比

### 4. 数据集处理 (datasets/)

**文件:**
- `data_utils.py` (~53KB): 数据下载与准备
- `dataloader.py` (~15KB): DataLoader 创建与变换
- `dataset.py` (~34KB): 数据集类
- `datasplit.py` (~21KB): Non-IID 数据划分

**支持数据集:**
- 图像分类: Caltech101, OxfordPets, Flowers102, Food101, DTD
- CIFAR10/100
- DomainNet, Office-Caltech10
- PACS

**数据划分策略:**
- IID (独立同分布)
- Non-IID Label Dirichlet: `noniid-labeldir`, `noniid-labeluni`, `noniid-labeldir100`
- Feature Skew 支持

### 5. 工具函数 (utils/)

#### fed_utils.py
| 函数 | 功能 |
|------|------|
| `average_weights()` | 联邦平均加权 |
| `moment_aggre_weights()` | 基于聚类的聚合 |
| `show_results()` | 计算并显示准确率、错误率、macro_f1 |
| `count_parameters()` | 统计模型参数量 |
| `save_acc_csv()` | 保存准确率历史 |
| `KMEANS` | 自定义 K-means 实现 |

#### dataloader.py
- DataLoader 创建与图像变换
- 高斯噪声注入 (隐私/增强)

#### datasplit.py
- `partition_data()`: Non-IID 数据划分
- `record_net_data_stats()`: 联邦数据统计

---

## 核心算法: FedPHA (GL_SVDMSE)

### 架构

```
全局上下文 (跨客户端聚合)
    ↓ SVD 分解
零空间 (正交补)
    ↓
局部上下文投影 (客户端特定)
    ↓ 拼接
提示文本特征 → 分类
```

### 训练流程

1. **全局轮次循环**
   - 客户端选择
   - 本地训练 (MAX_EPOCH 次)
   - 权重聚合

2. **本地训练**
   - 计算全局提示的零空间
   - 将本地提示投影到零空间
   - 最小化: CE + pull_loss + push_loss

3. **聚合**
   - 使用 FedAvg 聚合全局提示

---

## 联邦学习流程图

```
┌─────────────┐     ┌─────────────┐          ┌─────────────┐
│  Client 1   │     │  Client 2   │   ...    │  Client N   │
│ ┌─────────┐ │     │ ┌─────────┐ │          │ ┌─────────┐ │
│ │Local    │ │     │ │Local    │ │          │ │Local    │ │
│ │Prompt   │ │     │ │Prompt   │ │          │ │Prompt   │ │
│ └─────────┘ │     │ └─────────┘ │          │ └─────────┘ │
│      ↓      │     │      ↓      │          │      ↓      │
│ ┌─────────┐ │     │ ┌─────────┐ │          │ ┌─────────┐ │
│ │Global   │ │     │ │Global   │ │          │ │Global   │ │
│ │Prompt   │ │     │ │Prompt   │ │          │ │Prompt   │ │
│ │Projected│ │     │ │Projected│ │          │ │Projected│ │
│ └─────────┘ │     │ └─────────┘ │          │ └─────────┘ │
└──────┬──────┘     └──────┬──────┘          └──────┬──────┘
       │                   │                       │
       └───────────────────┼───────────────────────┘
                           ↓
                  ┌─────────────────┐
                  │  Server:        │
                  │  FedAvg         │
                  │  Aggregation    │
                  └────────┬────────┘
                           ↓
                  ┌─────────────────┐
                  │  New Global      │
                  │  Prompt          │
                  └─────────────────┘
```

---

## 实验数据集

### 图像分类数据集

| 数据集 | 类别数 | 描述 |
|--------|--------|------|
| Caltech101 | 101 | 物体分类 |
| CIFAR10 | 10 | 小图像分类 |
| CIFAR100 | 100 | 小图像多类 |
| OxfordPets | 37 | 宠物品种 |
| OxfordFlowers | 102 | 花卉分类 |
| Food101 | 101 | 食品分类 |
| DTD | 47 | 纹理描述 |
| Office31 | 31 | 域适应 (Amazon, DSLR, Webcam) |
| OfficeHome | 65 | 域适应 (Art, Clipart, Product, Real) |
| PACS | 7 | 域适应 (Photo, Art, Cartoon, Sketch) |
| DomainNet | 345 | 域适应 (Real, Sketch, Quickdraw, etc.) |

### 数据划分

- **IID**: 数据均匀分布到各客户端
- **Non-IID Label Dirichlet**: 使用 Dirichlet 分布 (beta 参数) 划分标签
- **Non-IID Label Uniform**: 标签均匀分配

---

## 实验参数配置

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_shots` | 2 | 每类样本数 (小样本设置) |
| `--num_users` | 10 | 客户端数量 |
| `--n_ctx` | 16 | 全局/本地提示长度 |
| `--alpha` | 1.0 | push_loss 间隔参数 |
| `--ratio` | 0.8 | SVD 零空间保留比例 |
| `--max_epoch` | 1 | 本地训练轮次 |
| `--round` | 50 | 全局通信轮次 |
| `--lr` | 0.001 | 学习率 |
| `--batch_size` | 32 | 训练批次大小 |
| `--seed` | 1 | 随机种子 |

### 非均匀数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--iid` | False | 是否使用 IID 分布 |
| `--beta` | 0.5 | Dirichlet 分布参数 (越小越不均匀) |
| `--noniid` | "labeldir" | Non-IID 类型 |

### 异构提示长度参数

| 参数 | 说明 |
|------|------|
| `--specify` | 启用异构提示长度 |
| `--prompts_lens` | 客户端提示长度列表 |

---

## 运行脚本

### 1. 小样本联邦学习 (FedPHA_few_shot.sh)

```bash
TRAINER="GL_SVDMSE"          # 训练器
DATASET="caltech101"         # 数据集
SHOTS=2                      # 小样本数
BACKBONE="rn50"              # 骨干网络 (ViT: vit_b32, ResNet: rn50)
USERS=10                     # 客户端数
SEED=1                       # 随机种子
```

### 2. 异构提示长度 (FedPHA_HE_prompts.sh)

```bash
TRAINER="GL_SVDMSE_HE"       # 异构扩展训练器
DATASET="cifar10"            # 数据集
SHOTS=2                      # 小样本数
USERS=6                      # 客户端数
PROMPT_LENS="[4,8,12,16,20,24]"  # 各客户端提示长度

# 提示长度分配模式: [A,A,B,B,C,C]
# 6个客户端: 2个A长度, 2个B长度, 2个C长度
```

---

## 输出格式

结果保存到:
```
output/{dataset}/{trainer}/shot_{shots}/beta_{beta}/ep{epochs}_r{rounds}/alpha{alpha}_ratio{ratio}/seed_{seed}/
```

**输出文件:**
- `acc.csv`: 每轮各客户端准确率
- 模型检查点
- 训练日志

---

## 环境依赖 (requirements.txt)

```
Python 3.8+
PyTorch 1.10.0+
CLIP (openai)
yacs (配置系统)
timm (骨干网络)
scikit-learn
numpy, pandas, matplotlib
prettytable (参数量统计)
```

---

## 使用示例

### 基本运行
```bash
bash scripts/FedPHA_few_shot.sh
```

### 异构提示长度
```bash
bash scripts/FedPHA_HE_prompts.sh
```

### 手动运行
```bash
python federated_main.py \
  --trainer GL_SVDMSE \
  --dataset caltech101 \
  --num_shots 2 \
  --num_users 10 \
  --n_ctx 16 \
  --alpha 1.0 \
  --ratio 0.8 \
  --iid False \
  --seed 1
```

---

## 项目创新点总结

1. **全局/本地提示分离**: 全局提示跨客户端聚合，本地提示保持私有
2. **SVD 正交投影**: 通过 SVD 分解将本地提示投影到全局提示的零空间
3. **Pull-Push 损失**: 保持本地特征的判别性，同时推动不同客户端的特征分离
4. **异构提示长度支持**: GL_SVDMSE_HE 支持不同客户端使用不同长度的提示
5. **最优传输聚合**: GLP_OT 使用条件最优传输进行更优的聚合