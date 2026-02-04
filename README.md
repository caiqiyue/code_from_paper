# 联邦学习算法合集（caiqiyue_file）

本仓库收集多种联邦学习/隐私学习相关算法的实现与实验代码。下面按子文件夹列出当前已包含的算法及其简要说明，方便快速定位与使用。

**算法子目录一览**

| 子目录 | 算法/论文 | 简要说明 |
| --- | --- | --- |
| `DPGA-TextSyn` | DPGA-TextSyn | 文本合成方向的算法实现，包含数据处理、下游评测与隐私分析相关内容。 |
| `FedCOG` | FedCOG: Federated Learning with Consensus-Oriented Generation | 通过“共识导向生成”缓解联邦学习中的异质性，结合互补数据生成与知识蒸馏训练。 |
| `KnowledgeSG` | KnowledgeSG: Privacy-Preserving Synthetic Text Generation with Knowledge Distillation from Server | 服务器侧知识蒸馏驱动的隐私文本合成，包含基线生成与后处理流程。 |
| `PCEvolve` | PCEvolve: Private Contrastive Evolution for Synthetic Dataset Generation | 使用少量私有数据与生成式 API，基于对比演化与 DP 选择器生成高质量合成数据集。 |
| `POPri` | POPri: Private Federated Learning using Preference-Optimized Synthetic Data | 将合成数据生成转化为偏好优化/强化学习问题，提升 DP 合成数据质量。 |
| `PrE-Text` | PrE-Text: Training Language Models on Private Federated Data in the Age of LLMs | 服务器侧两阶段 DP 合成文本：先生成 DP seed，再扩展为大规模合成数据。 |
| `PrivateKT` | PrivateKT | 联邦知识迁移框架，包含重要性采样、知识缓冲、自训练、随机响应与无偏聚合机制。 |
| `RewardDS` | RewardDS: Privacy-Preserving Fine-Tuning via Reward Driven Data Synthesis | 客户端训练生成/奖励代理模型，用奖励引导过滤与自优化提升合成数据质量。 |
| `WASP` | Contrastive Private Data Synthesis via Weighted Multi-PLM Fusion | 通过多预训练语言模型加权融合，进行对比式私有数据合成。 |

**非算法目录说明**

- `datasets`: 数据集存放与处理相关内容。
- `models`: 模型或权重相关内容。
- `outputs`: 训练/评测输出结果。
- `clash_for_linux`: 工具相关目录（非算法）。

#每次拉取代码注意子模块
如需更细节的算法说明或使用方法，请进入对应子目录查看其 `README.md` 或代码注释。
