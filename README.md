# 开发目标

本项目后续开发计划如下：

## 1. 模型参数调优
- **实施计划**：
  - 引入 Optuna 框架，实现超参数自动优化。
  - 设计合理的搜索空间，重点优化模型结构和训练参数。
  - 实现并行实验管理，提高实验效率。
- **文件创建/修改**：
  - `src/hyperparameter_tuning.py`：Optuna 调参主程序。
  - `configs/optuna_config.yaml`：调参配置文件。
- **主要任务**：
  1. 定义参数搜索空间（如学习率、网络层数、卷积核大小等）。
  2. 实现目标函数，使用验证集性能作为优化指标。
  3. 设计实验跟踪与结果可视化逻辑。
  4. 运行调参任务并选择最佳模型配置。

## 2. 设备适配训练（Emotiv EPOC X）
- **实施计划**：
  - 针对 Emotiv EPOC X 的通道配置进行数据适配。
  - 通过通道裁剪模拟目标设备的数据结构。
  - 设计和评估专用于少通道设备的轻量级模型。
- **文件创建/修改**：
  - 扩展 `src/dataset.py`，添加通道适配功能。
  - 修改 `configs/config.yaml`，增加设备适配配置选项。
- **主要任务**：
  1. 分析 Emotiv EPOC X 通道配置和数据格式。
  2. 在现有数据集处理流程中添加通道映射和裁剪功能。
  3. 修改 `BrainAuthDataset` 类以支持通道自适应。
  4. 训练和评估在裁剪数据上的模型性能。

## 3. 实际应用场景验证
- **实施计划**：
  - 采集真实用户的 EEG 数据进行离线验证。
  - 评估模型在实际数据上的识别准确率。


# BrainAuth 模型开发指南

本文档为 `brainauth_model.py` 文件中定义的 PyTorch 模型提供开发指南，旨在帮助开发者理解、使用和扩展这些模型。

## 1. 概述

`brainauth_model.py` 模块包含用于 BrainAuth 项目的深度学习模型，主要用于基于脑电图（EEG）数据的身份认证（验证任务）。这些模型接收成对的EEG空-频特征图作为输入，并判断它们是否来自同一个体。

## 2. 模型架构

目前实现了以下几种模型：

### 2.1 `P3DCNN`

*   **描述**: 基于伪三维卷积神经网络（Pseudo-3D CNN）的模型，用于处理EEG空-频特征图。它通过分别处理两个输入样本，提取特征后计算特征差异，最后通过全连接层进行分类。
*   **主要参数**:
    *   `input_shape` (tuple): 输入空-频特征图的形状，默认为 `(110, 100, 10)`，代表（深度/频带数, 高度, 宽度）。注意，PyTorch的 `Conv3d` 期望输入为 `(N, C, D, H, W)`。在模型内部，单通道输入会自动扩展维度。
    *   `num_classes` (int): 输出类别数，对于验证任务通常为2（同一人/非同一人）。
*   **核心逻辑**:
    1.  `_forward_conv(x)`: 包含一系列3D卷积层（`conv1` 至 `conv6`）、批归一化（BatchNorm）、ReLU激活和Dropout，用于从单个EEG特征图中提取深层特征。
    2.  `forward(x1, x2)`:
        *   分别将 `x1` 和 `x2` 通过 `_forward_conv` 提取特征。
        *   计算两个特征向量之间的绝对差值。
        *   将差值向量展平并通过全连接层（`fc`）和输出层（`out`）得到分类结果。

### 2.2 `SiameseBrainAuth`

*   **描述**: 一个基于孪生网络（Siamese Network）架构的深度学习模型，专为EEG身份认证设计。此架构的核心在于其**双路处理机制**：它使用**同一个共享的特征提取器**（权重共享）并行处理两个输入的EEG空-频特征图。这种设计使得模型能够学习一个有效的度量空间，在该空间中，来自同一个体的样本对的特征表示会更接近，而来自不同个体的样本对则会更疏远。
*   **主要参数**:
    *   `input_shape` (tuple): 单个输入空-频特征图的形状，默认为 `(110, 100, 10)`，代表（深度/频带数, 高度, 宽度）。
*   **核心逻辑**:
    1.  **共享特征提取器 (`feature_extractor`)**: 这是一个 `nn.Sequential` 模块，由一系列3D卷积层、激活函数（如ReLU）、批归一化（BatchNorm）和Dropout层组成。此提取器的权重在处理两个输入样本时是共享的，确保两个样本通过完全相同的变换。
    2.  **单路前向传播 (`forward_one(x)`)**: 此辅助方法负责将单个EEG特征图 `x` 输入到共享的 `feature_extractor` 中，提取其高维特征表示，并随后将特征图展平（flatten）为一维向量。
    3.  **双路并行处理与比较 (`forward(x1, x2)`)**: 
        *   两个输入样本 `x1` 和 `x2` 分别独立地通过 `forward_one` 方法，生成各自的特征向量 `feat1` 和 `feat2`。
        *   这两个特征向量随后被拼接（`torch.cat`）形成一个组合特征向量。
        *   拼接后的向量被送入一个或多个全连接层（`fc`），这些层学习如何根据组合特征来判断原始样本对的相似性。
        *   最终，输出层通常使用Sigmoid激活函数，产生一个介于0和1之间的相似度得分，表示 `x1` 和 `x2` 来自同一个体的概率。

### 2.3 `LightP3DCNN`

*   **描述**: `P3DCNN` 的轻量级版本，旨在减少模型参数量和计算复杂度。它通过减少卷积层通道数、使用更小的卷积核以及简化网络结构来实现轻量化。
*   **主要参数**: 与 `P3DCNN` 类似。
*   **与 `P3DCNN` 的主要区别**:
    *   `_forward_conv` 方法中的卷积层（如 `conv1`, `conv2`, `conv3`, `conv6`）具有更少的输出通道。
    *   跳过了 `P3DCNN` 中的 `conv4` 和 `conv5` 层。
    *   Dropout比例可能有所调整。

### 2.4 `LightSiameseBrainAuth`

*   **描述**: `SiameseBrainAuth` 的轻量级版本。其共享的 `feature_extractor` 结构参考了 `LightP3DCNN` 的卷积部分设计，全连接层也相应简化。
*   **主要参数**: 与 `SiameseBrainAuth` 类似。
*   **与 `SiameseBrainAuth` 的主要区别**:
    *   `feature_extractor` 采用了更少的卷积层和更少的通道数。
    *   全连接层（`fc`）的神经元数量减少。

### 2.5 数据输入与预处理

BrainAuth模型的训练和评估依赖于经过精心预处理的EEG数据。数据处理流程通常包括以下步骤，主要涉及 `src/dataset.py`, `src/preprocess.py`, `src/preprocess_data.py` 和 `src/EEGSpectralConverter.py` 等模块：

1.  **数据加载**:
   *   原始EEG数据通常以特定格式（如EDF, BDF, FIF, 或预处理后的 `.h5` 文件）存储。
   *   使用 `src/dataset.py` 中的 `BrainAuthDataset` 类来加载数据。这些类负责从原始数据文件中读取EEG信号和对应的标签（例如受试者ID）。

2.  **数据分段 (Epoching)**:
   * 使用 `preprocess.py` 脚本将原始数据按照不同的实验条件（如 `eyes_open` 和 `eyes_closed`）组织到相应目录结构。
   * `BrainAuthDataset` 类在初始化时会自动将连续的EEG数据分割成固定长度的段，使用 `segment_duration` 参数控制每个段的持续时间（默认为2秒）。
   * 通过设置 `overlap` 参数（默认为0.5）可控制相邻段之间的重叠率，增加可用于训练的数据量。
   * 分段过程会为每个受试者创建多个EEG段，并记录每段的起始和结束时间。

3.  **预处理**:
   * 使用 `_load_and_preprocess` 方法对每个EEG段进行预处理，包括：
     * 通道选择：可通过 `channels` 参数指定使用特定通道。
     * 重采样：通过 `resample` 参数可将信号降采样到指定频率（默认160Hz）。
     * 滤波：应用高通滤波（默认1.0Hz）和可选的低通滤波，去除基线漂移和高频噪声。
     * 标准化：默认情况下对每个通道进行标准化处理（减去均值并除以标准差）。
   * 错误处理机制：预处理过程中包含多重异常处理，确保在数据异常（如NaN值、文件缺失等）情况下返回零填充数据而不中断程序。
   * 三级缓存策略：为提高效率，实现了内存缓存、HDF5文件缓存和即时计算的三级缓存机制，避免重复处理相同的数据段。

4.  **空-频特征图转换**:
   *   预处理后的EEG片段（通常是多通道时域信号）被转换为模型所需的空-频特征图。
   *   `src/EEGSpectralConverter.py` 模块负责此转换过程：
       * 使用Welch方法计算每个通道的功率谱密度（PSD）。
       * 将频谱划分为多个频带（默认10个），并计算每个频带的能量。
       * 基于电极的2D位置，将每个频带的能量值映射到2D平面上，使用插值（默认cubic）生成连续的空间分布图。
       * 输出形状为 (110, 100, 10) 的特征图，表示空间-频率的三维表示。

5.  **数据对生成 (Pairing)**:
   *   对于验证任务，模型需要成对的EEG特征图 (`x1`, `x2`) 作为输入。
   *   `src/dataset.py` 中的 `BrainAuthDataset` 类负责生成这些数据对：
       *   **正样本对**: `x1` 和 `x2` 来自同一个受试者，标签为1。
       *   **负样本对**: `x1` 和 `x2` 来自不同的受试者，标签为0。
   *   通过 `pos_ratio` 参数（默认0.5）控制正负样本对的比例。
   *   数据加载器确保在每个批次中以适当的比例混合正负样本对，并在训练阶段随机打乱样本顺序。
   * 使用 `get_dataloaders` 函数可自动划分训练集、验证集和测试集（默认比例为70%/15%/15%），确保模型评估的可靠性。

### 2.6 工具函数说明（src/utils.py）

`src/utils.py` 文件包含了项目中常用的工具函数，主要包括：
- EEG文件探索：如`explore_eeg_file`，用于获取EEG文件的基本信息。
- EEG数据绘图：如`plot_eeg_channels`，支持对指定通道和时间段的EEG信号进行可视化。
- 随机种子设置：`set_seed`，保证实验结果可复现。
- 配置加载：`load_config`，便于读取YAML格式的配置文件。
- 设备选择：`get_device`，自动检测并选择GPU或CPU。
- 目录创建：`create_dirs`，批量创建所需目录。
- 模型保存与加载：`save_checkpoint`、`load_checkpoint`，用于模型训练过程中的断点保存与恢复。
- 性能指标计算：`calculate_metrics`，支持准确率、精确率、召回率、F1、AUC、EER等常用指标。
- 指标可视化：`plot_metrics`，可绘制训练/验证损失、准确率、F1等历史曲线。

## 3. 文件结构

```
BrainAuth/
├── README.md                # 项目说明文档
├── requirements.txt         # Python依赖包列表
├── configs/                 # 配置文件目录（如config.yaml）
├── data/                    # 数据存放目录（原始/预处理/特征等）
├── models/                  # 深度学习模型定义与相关代码
│   ├── brainauth_model.py   # 主要模型实现文件（P3DCNN等）
│   └── __init__.py
├── src/                     # 核心源码（数据处理、训练等）
│   ├── dataset.py           # 数据集加载与样本对生成
│   ├── preprocess.py        # 数据预处理脚本
│   ├── preprocess_data.py   # 数据批量预处理入口
│   ├── EEGSpectralConverter.py # 空-频特征图转换工具
│   ├── EEGHDFCache.py       # HDF5缓存管理
│   ├── train.py             # 训练主程序
│   └── utils.py             # 辅助工具函数
├── notebooks/               # Jupyter/可视化分析笔记本
│   └── visualize_eeg.py     # EEG数据可视化示例
├── outputs/                 # 输出结果（模型、日志、图表等）
│   ├── checkpoints/         # 训练中保存的模型权重
│   ├── logs/                # 日志文件
│   └── results/             # 结果图表与评估指标
│       └── training_metrics.png
└── .venv/                   # Python虚拟环境（可选）
```

各目录及文件说明：
- **configs/**：存放项目运行所需的配置文件，如模型参数、日志设置等。
- **data/**：用于存放原始EEG数据、预处理数据及生成的特征文件。
- **models/**：包含所有模型结构定义及相关代码。
- **src/**：核心代码，包括数据集加载、预处理、特征转换、训练与工具函数。
- **notebooks/**：用于实验、可视化和分析的Jupyter笔记本。
- **outputs/**：保存训练过程中的模型、日志和结果图表。
- **requirements.txt**：项目依赖的Python包清单。
- **README.md**：项目说明文档，包含开发指南、模型介绍等。

## 4. 配置文件说明（configs/config.yaml）

`configs/config.yaml` 是BrainAuth项目的核心配置文件，集中管理模型训练、数据处理、日志记录等各项参数，便于实验复现和灵活调整。主要内容如下：

### 4.1 主要作用
- 统一管理项目的各类参数，避免硬编码。
- 支持不同实验场景下的快速切换（如模型结构、数据路径、预处理方式、训练策略等）。
- 便于团队协作和实验复现。

### 4.2 主要配置项说明

- **seed**：随机种子，保证实验可复现性。
- **data**：数据相关配置，包括：
  - `data_dir`：数据目录路径。
  - `condition`：实验条件（如eyes_open/eyes_closed）。
  - `batch_size`：每批次样本数。
  - `num_workers`：数据加载线程数。在使用hdf5缓存时，设置为1以避免多线程冲突。
  - `cache_data`、`use_hdf5_cache`、`hdf5_cache_dir`：数据缓存相关设置。
  - `pos_ratio`：正负样本对比例。
  - `preprocess_params`：预处理参数（如滤波频段、分段长度、重叠率、标准化、通道选择、重采样、频带划分等）。
- **model**：模型结构配置，包括：
  - `name`：模型名称（如LightSiameseBrainAuth、P3DCNN等）。
  - `input_shape`：输入特征图形状。
  - `num_classes`：类别数。
  - `pretrained`：预训练权重路径（如有）。
- **train**：训练相关参数，包括：
  - `epochs`：训练轮数。
  - `learning_rate`：学习率。
  - `weight_decay`：权重衰减。
  - `lr_scheduler`及其参数：学习率调度策略。
  - `early_stopping`及其耐心值：早停策略。
  - `loss_function`：损失函数类型。
- **evaluation**：评估设置，包括：
  - `metrics`：评估指标（如accuracy、f1、auc、eer等）。
  - `best_metric`：用于模型选择的主指标。
  - `subject_split`：受试者划分比例（训练/验证/测试）。
  - `monitor_subject_performance`：是否监控个体表现。
- **logging**：日志相关配置，如是否记录张量形状等。
- **output**：结果保存路径及策略，包括模型权重、日志、结果图表的保存目录、保存频率等。

### 4.3 配置调整示例

- 若需切换模型结构，只需修改 `model.name` 字段，如：
  ```yaml
  model:
    name: "P3DCNN"
  ```
- 调整数据分段长度或重叠率，可修改：
  ```yaml
  data:
    preprocess_params:
      segment_duration: 2.0
      overlap: 0.5
  ```
- 更改训练轮数和学习率：
  ```yaml
  train:
    epochs: 100
    learning_rate: 0.001
  ```
- 更改日志详细程度：
  ```yaml
  logging:
    log_tensor_shapes: true
  ```

通过合理配置`configs/config.yaml`，可灵活适配不同实验需求，提升开发效率和实验可控性。
