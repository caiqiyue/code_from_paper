# Start Scripts 说明

本目录包含用于测试和下载的脚本。

## 环境测试脚本

| 脚本 | 功能 |
|------|------|
| `1_check_env.sh` | 检查 conda 环境、PyTorch、CUDA、关键包 |
| `2_check_imports.sh` | 检查 thesis_platform 模块导入 |
| `3_list_experiments.sh` | 列出所有可用实验 |
| `4_run_simple_test.sh` | 运行简单测试（验证 runner 流程） |
| `5_check_models.sh` | 检查模型文件状态 |

## 下载脚本

| 脚本 | 功能 |
|------|------|
| `6_list_available.sh` | 列出所有可用的数据集和模型 |
| `7_download_datasets.sh` | **后台下载所有数据集** |
| `8_download_models.sh` | **后台下载所有模型（不含大模型）** |
| `9_download_specific.sh` | 下载指定的数据集或模型 |
| `A_download_large_models.sh` | 下载大模型（>15B参数） |
| `B_check_download_status.sh` | 检查下载状态和日志 |

## 使用方法

### 1. 先测试环境
```bash
conda activate caiqiyue
cd /root/autodl-tmp/caiqiyue/code_from_paper
bash start/1_check_env.sh
bash start/2_check_imports.sh
```

### 2. 查看可用数据集/模型
```bash
bash start/6_list_available.sh
```

### 3. 下载所有数据集
```bash
bash start/7_download_datasets.sh
# 查看日志: tail -f outputs/download_logs/datasets_*.log
```

### 4. 下载所有模型（不含大模型）
```bash
bash start/8_download_models.sh
# 查看日志: tail -f outputs/download_logs/models_*.log
```

### 5. 下载指定的数据集或模型
```bash
# 下载指定数据集
bash start/9_download_specific.sh dataset gsm8k imdb

# 下载指定模型
bash start/9_download_specific.sh model opt-125m distilgpt2
```

### 6. 下载大模型（需要确认）
```bash
bash start/A_download_large_models.sh
```

### 7. 检查下载状态
```bash
bash start/B_check_download_status.sh
```

## 日志位置

所有下载日志保存在: `outputs/download_logs/`

- `datasets_YYYYMMDD_HHMMSS.log` - 数据集下载日志
- `models_YYYYMMDD_HHMMSS.log` - 模型下载日志
- `large_models_YYYYMMDD_HHMMSS.log` - 大模型下载日志

## 报告文件

下载完成后会生成报告:
- `thesis_platform/datasets/download_report.json` - 数据集下载报告
- `thesis_platform/models/download_report.json` - 模型下载报告