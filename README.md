# BrainAuth: EEG-based Personal Identification

## 项目简介
BrainAuth 是一个基于脑电（EEG）信号的个体身份识别系统，集成了数据预处理、深度学习模型训练与超参数自动优化等完整流程。该项目支持多种模型结构，SiameseICAConvNet，在本项目实验中取得了最佳效果。

## 主要特性
- EEG 数据批量化整理与预处理（滤波、重采样、归一化、窗口切分等）
- Optuna 自动超参数调优
- 训练与评估流程可视化
- 配置化实验管理，易于复现

## 安装与环境
1. 克隆本仓库
2. 安装依赖（推荐使用虚拟环境）
   ```bash
   pip install -r requirements.txt
   ```
3. 准备原始 EEG 数据并放置于 `data/raw/` 目录

## 快速开始

### 1. 数据整理
```bash
python src/organize_file.py
```

### 2. 数据预处理
```bash
python src/preprocess.py --config configs/config.yaml
```

### 3. 模型训练（推荐 SiameseICAConvNet）
```bash
python src/train.py --config configs/config.yaml
```

### 4. 超参数自动调优
```bash
python src/hyperparameter_tuning.py
```

## 目录结构
```
BrainAuth/
├── configs/                # 配置文件
├── data/                   # 数据目录
│   ├── raw/                # 原始数据
│   ├── organized/          # 整理后的数据
│   └── processed/          # 预处理后数据
├── outputs/                # 训练与调优结果
├── src/                    # 源代码
│   ├── preprocess.py
│   ├── train.py
│   ├── hyperparameter_tuning.py
│   └── ...
└── README.md
```

## 配置说明
- 所有流程均通过 YAML 配置文件控制，详见 `configs/config.yaml`。
- 支持命令行参数覆盖配置项。
- 推荐在 `dataloader.mode` 选择 `siamese`，并使用 SiameseICAConvNet 结构。

## 贡献指南
欢迎提交 issue 或 pull request 以改进本项目。请遵循标准的代码风格和提交规范。

## 引用
如在学术研究中使用本项目，请引用相关论文或本仓库。

## 联系方式
如有问题或合作意向，请联系项目维护者。

