"""
BrainAuth Model: 基于空-频特征图学习的脑电图验证模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import yaml
import os
from typing import Tuple, List, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BrainAuth')

# 加载配置文件
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
_log_tensor_shapes = True  # 默认开启
try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    _log_tensor_shapes = config.get('logging', {}).get('log_tensor_shapes', True)
except FileNotFoundError:
    logger.warning(f"Configuration file {CONFIG_PATH} not found. Using default logging settings.")
except yaml.YAMLError as e:
    logger.warning(f"Error parsing configuration file {CONFIG_PATH}: {e}. Using default logging settings.")


class P3DCNN(nn.Module):
    """
    基于空-频特征图学习的3D卷积神经网络
    针对验证(verification)场景的脑电图身份认证模型
    """
    
    def __init__(self, input_shape=(110, 100, 10), num_classes=2):
        """
        初始化网络结构
        
        参数:
            input_shape: 输入空-频特征图的形状 (高度, 宽度, 频带数)
            num_classes: 分类数（通常为2，表示同一人或不同人）
        """
        super(P3DCNN, self).__init__()
        self.input_shape = input_shape
        
        # 空间-频域伪三维卷积模块
        # 卷积层1 - 空间方向
        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=16,
            kernel_size=(5, 5, 1),  # 空间方向卷积
            stride=1,
            padding='same'
        )
        
        # 卷积层2 - 频域方向
        self.conv2 = nn.Conv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=(1, 1, 5),  # 频域方向卷积
            stride=1,
            padding='same'
        )
        
        # 三维卷积-池化模块
        # 卷积层3
        self.conv3 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=2,  # 步长为2实现降采样/池化
            padding=1
        )
        self.bn1 = nn.BatchNorm3d(64)
        
        # 卷积层4
        self.conv4 = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3, 3),
            stride=1,
            padding='same'
        )
        
        # 卷积层5
        self.conv5 = nn.Conv3d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3, 3),
            stride=1,
            padding='same'
        )
        
        # 卷积层6
        self.conv6 = nn.Conv3d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3, 3),
            stride=2,  # 步长为2实现降采样/池化
            padding=1
        )
        self.bn2 = nn.BatchNorm3d(256)
        self.dropout = nn.Dropout(0.5)
        
        # 计算卷积后的特征尺寸
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # 全连接层
        self.fc = nn.Linear(conv_output_size, 256)
        
        # 输出层
        self.out = nn.Linear(256, num_classes)
    
    def _get_conv_output_size(self, shape):
        """计算卷积层输出的特征尺寸"""
        batch_size = 1
        input_tensor = torch.zeros(batch_size, 1, *shape)
        output = self._forward_conv(input_tensor)
        return output.numel() // batch_size
    
    def _forward_conv(self, x):
        """前向传播卷积部分"""
        if _log_tensor_shapes:
            logger.info(f"_forward_conv input shape: {x.shape}")
        x = F.relu(self.conv1(x))
        if _log_tensor_shapes:
            logger.info(f"After conv1 shape: {x.shape}")
        x = F.relu(self.conv2(x))
        if _log_tensor_shapes:
            logger.info(f"After conv2 shape: {x.shape}")
        
        # 三维卷积-池化模块
        x = F.relu(self.bn1(self.conv3(x)))
        if _log_tensor_shapes:
            logger.info(f"After conv3 shape: {x.shape}")
        x = F.relu(self.conv4(x))
        if _log_tensor_shapes:
            logger.info(f"After conv4 shape: {x.shape}")
        x = F.relu(self.conv5(x))
        if _log_tensor_shapes:
            logger.info(f"After conv5 shape: {x.shape}")
        x = F.relu(self.bn2(self.conv6(x)))
        if _log_tensor_shapes:
            logger.info(f"After conv6 shape: {x.shape}")
        x = self.dropout(x)
        return x
    
    def forward(self, x1, x2):
        """前向传播"""
        # 打印输入形状
        if _log_tensor_shapes:
            logger.info(f"P3DCNN forward - x1 shape: {x1.shape}")
            logger.info(f"P3DCNN forward - x2 shape: {x2.shape}")
        
        # 确保输入形状正确 (batch_size, channels, depth, height, width)
        if x1.dim() == 4:
            if _log_tensor_shapes:
                logger.info("Adding channel dimension to x1")
            x1 = x1.unsqueeze(1)
            if _log_tensor_shapes:
                logger.info(f"After unsqueeze, x1 shape: {x1.shape}")
        if x2.dim() == 4:
            if _log_tensor_shapes:
                logger.info("Adding channel dimension to x2")
            x2 = x2.unsqueeze(1)
            if _log_tensor_shapes:
                logger.info(f"After unsqueeze, x2 shape: {x2.shape}")
        
        # 分别通过卷积层提取特征
        if _log_tensor_shapes:
            logger.info("Starting forward conv for x1")
        x1_conv = self._forward_conv(x1)
        if _log_tensor_shapes:
            logger.info(f"x1_conv shape: {x1_conv.shape}")
        
        if _log_tensor_shapes:
            logger.info("Starting forward conv for x2")
        x2_conv = self._forward_conv(x2)
        if _log_tensor_shapes:
            logger.info(f"x2_conv shape: {x2_conv.shape}")
        
        # 特征拼接或计算差异
        # 方法1: 欧氏距离
        x_diff = torch.abs(x1_conv - x2_conv)
        if _log_tensor_shapes:
            logger.info(f"x_diff shape: {x_diff.shape}")
        
        x_flat = x_diff.view(x_diff.size(0), -1)
        if _log_tensor_shapes:
            logger.info(f"x_flat shape: {x_flat.shape}")
        
        # 全连接层
        x = F.relu(self.fc(x_flat))
        if _log_tensor_shapes:
            logger.info(f"After fc shape: {x.shape}")
        
        # 输出层
        output = self.out(x)
        if _log_tensor_shapes:
            logger.info(f"Output shape: {output.shape}")
        
        return output

class SiameseBrainAuth(nn.Module):
    """孪生网络模型，用于EEG身份验证
    
    使用共享权重的特征提取器处理一对EEG特征图，然后比较它们的相似性
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (110, 100, 10), 
                conv_channels: Optional[List[int]] = None,
                hidden_size: int = 256,
                dropout_rate: float = 0.5,
                use_batch_norm: bool = True):
        """初始化模型
        
        Args:
            input_shape: 输入特征图形状 (D, H, W)
            conv_channels: 卷积层通道数列表 [conv1,...,conv6]，如果为None则使用默认值
            hidden_size: 全连接层隐藏单元数量
            dropout_rate: Dropout比例
            use_batch_norm: 是否使用批归一化
        """
        super(SiameseBrainAuth, self).__init__()
        
        # 如果没有指定卷积通道数，使用默认值
        if conv_channels is None or len(conv_channels) < 6:
            conv_channels = [16, 32, 64, 128, 256, 512]
        
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # 输入形状
        self.input_shape = input_shape
        
        # 构建特征提取器
        layers = []
        
        # Conv1 层：1 -> conv_channels[0]
        layers.append(nn.Conv3d(1, conv_channels[0], kernel_size=3, stride=1, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(conv_channels[0]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout3d(p=dropout_rate))
        
        # Conv2 层：conv_channels[0] -> conv_channels[1]
        layers.append(nn.Conv3d(conv_channels[0], conv_channels[1], kernel_size=3, stride=1, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(conv_channels[1]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout3d(p=dropout_rate))
        
        # Conv3 层：conv_channels[1] -> conv_channels[2]，步长2
        layers.append(nn.Conv3d(conv_channels[1], conv_channels[2], kernel_size=3, stride=2, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(conv_channels[2]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout3d(p=dropout_rate))
        
        # Conv4 层：conv_channels[2] -> conv_channels[3]
        layers.append(nn.Conv3d(conv_channels[2], conv_channels[3], kernel_size=3, stride=1, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(conv_channels[3]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout3d(p=dropout_rate))
        
        # Conv5 层：conv_channels[3] -> conv_channels[4]，步长2
        layers.append(nn.Conv3d(conv_channels[3], conv_channels[4], kernel_size=3, stride=2, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(conv_channels[4]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout3d(p=dropout_rate))
        
        # Conv6 层：conv_channels[4] -> conv_channels[5]
        layers.append(nn.Conv3d(conv_channels[4], conv_channels[5], kernel_size=3, stride=1, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(conv_channels[5]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout3d(p=dropout_rate))
        
        # 创建特征提取器
        self.feature_extractor = nn.Sequential(*layers)
        
        # 计算特征维度 - 使用一个临时tensor计算输出尺寸
        with torch.no_grad():
            # 计算卷积网络输出的特征维度
            dummy_input = torch.zeros(1, 1, *input_shape)
            dummy_output = self.feature_extractor(dummy_input)
            self.flat_features = dummy_output.numel() // dummy_output.size(0)
            print(f"卷积输出特征的形状: {dummy_output.shape}")
            print(f"展平后的特征维度: {self.flat_features}")
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.flat_features * 2, hidden_size),  # 两倍特征大小，因为我们将连接两个特征向量
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, 1)  # 输出一个相似度得分
        )
    
    def forward_one(self, x):
        """前向传播单个样本
        
        Args:
            x: 脑电图特征图，形状 (batch_size, depth, height, width)
                
        Returns:
            特征向量
        """
        # 确保输入是5D张量 [batch, channel, depth, height, width]
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        
        # 通过共享的特征提取器
        x = self.feature_extractor(x)
        
        # 展平特征
        x = x.view(x.size(0), -1)
        
        return x
    
    def forward(self, x1, x2):
        """前向传播两个样本
        
        Args:
            x1: 第一个脑电图特征图，形状 (batch_size, depth, height, width)
            x2: 第二个脑电图特征图，形状 (batch_size, depth, height, width)
                
        Returns:
            相似度得分，形状 (batch_size, 1)
        """
        # 分别提取特征
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)
        
        # 连接特征向量
        combined = torch.cat((feat1, feat2), 1)
        
        # 通过全连接层计算相似度
        similarity = self.fc(combined)
        
        return similarity
    
    def get_embedding(self, x):
        """获取输入的嵌入向量表示
        
        这个方法可用于提取特征，用于可视化或其他分析
        
        Args:
            x: 脑电图特征图，形状 (batch_size, depth, height, width)
                
        Returns:
            特征嵌入向量
        """
        return self.forward_one(x)


class LightSiameseBrainAuth(nn.Module):
    """轻量级孪生网络模型，用于EEG身份验证
    
    SiameseBrainAuth的轻量级版本，减少卷积层数和通道数以降低计算复杂度
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (110, 100, 10), 
                conv_channels: Optional[List[int]] = None,
                hidden_size: int = 128,
                dropout_rate: float = 0.3,
                use_batch_norm: bool = True):
        """初始化模型
        
        Args:
            input_shape: 输入特征图形状 (D, H, W)
            conv_channels: 卷积层通道数列表 [conv1,conv2,conv3,conv6]，如果为None则使用默认值
            hidden_size: 全连接层隐藏单元数量
            dropout_rate: Dropout比例
            use_batch_norm: 是否使用批归一化
        """
        super(LightSiameseBrainAuth, self).__init__()
        
        # 如果没有指定卷积通道数，使用默认值
        if conv_channels is None or len(conv_channels) < 4:
            conv_channels = [8, 16, 32, 64]
        
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # 输入形状
        self.input_shape = input_shape
        
        # 构建特征提取器
        layers = []
        
        # Conv1 层：1 -> conv_channels[0]
        layers.append(nn.Conv3d(1, conv_channels[0], kernel_size=3, stride=1, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(conv_channels[0]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout3d(p=dropout_rate))
        
        # Conv2 层：conv_channels[0] -> conv_channels[1]，步长2
        layers.append(nn.Conv3d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(conv_channels[1]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout3d(p=dropout_rate))
        
        # Conv3 层：conv_channels[1] -> conv_channels[2]，步长2
        layers.append(nn.Conv3d(conv_channels[1], conv_channels[2], kernel_size=3, stride=2, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(conv_channels[2]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout3d(p=dropout_rate))
        
        # Conv6 层（跳过4和5）：conv_channels[2] -> conv_channels[3]
        layers.append(nn.Conv3d(conv_channels[2], conv_channels[3], kernel_size=3, stride=1, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(conv_channels[3]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout3d(p=dropout_rate))
        
        # 创建特征提取器
        self.feature_extractor = nn.Sequential(*layers)
        
        # 计算特征维度 - 使用一个临时tensor计算输出尺寸
        with torch.no_grad():
            # 计算卷积网络输出的特征维度
            dummy_input = torch.zeros(1, 1, *input_shape)
            dummy_output = self.feature_extractor(dummy_input)
            self.flat_features = dummy_output.numel() // dummy_output.size(0)
            print(f"轻量级模型卷积输出特征的形状: {dummy_output.shape}")
            print(f"轻量级模型展平后的特征维度: {self.flat_features}")
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.flat_features * 2, hidden_size),  # 两倍特征大小，因为我们将连接两个特征向量
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, 1)  # 输出一个相似度得分
        )
    
    def forward_one(self, x):
        """前向传播单个样本
        
        Args:
            x: 脑电图特征图，形状 (batch_size, depth, height, width)
                
        Returns:
            特征向量
        """
        # 确保输入是5D张量 [batch, channel, depth, height, width]
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        
        # 通过共享的特征提取器
        x = self.feature_extractor(x)
        
        # 展平特征
        x = x.view(x.size(0), -1)
        
        return x
    
    def forward(self, x1, x2):
        """前向传播两个样本
        
        Args:
            x1: 第一个脑电图特征图，形状 (batch_size, depth, height, width)
            x2: 第二个脑电图特征图，形状 (batch_size, depth, height, width)
                
        Returns:
            相似度得分，形状 (batch_size, 1)
        """
        # 分别提取特征
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)
        
        # 连接特征向量
        combined = torch.cat((feat1, feat2), 1)
        
        # 通过全连接层计算相似度
        similarity = self.fc(combined)
        
        return similarity
    
    def get_embedding(self, x):
        """获取输入的嵌入向量表示
        
        这个方法可用于提取特征，用于可视化或其他分析
        
        Args:
            x: 脑电图特征图，形状 (batch_size, depth, height, width)
                
        Returns:
            特征嵌入向量
        """
        return self.forward_one(x)


# 辅助函数：通道适配函数，用于Emotiv EPOC X的适配
def adapt_channels_for_emotiv(original_channels, target_channels=None):
    """
    将原始EEG通道适配到目标通道配置
    
    Args:
        original_channels: 原始EEG通道列表
        target_channels: 目标通道列表，默认为Emotiv EPOC X通道
        
    Returns:
        映射字典和保留的通道索引
    """
    if target_channels is None:
        # Emotiv EPOC X 通道配置
        target_channels = [
            "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", 
            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
        ]
    
    # 创建通道映射
    channel_map = {}
    kept_indices = []
    
    # 对每个目标通道，找到原始通道中的匹配项
    for i, channel in enumerate(original_channels):
        # 标准化通道名称（去除空格和转换为大写）
        norm_channel = channel.strip().upper()
        
        if norm_channel in [ch.strip().upper() for ch in target_channels]:
            channel_map[channel] = channel
            kept_indices.append(i)
        else:
            # 尝试模糊匹配
            for target in target_channels:
                norm_target = target.strip().upper()
                if norm_target in norm_channel or norm_channel in norm_target:
                    channel_map[channel] = target
                    kept_indices.append(i)
                    break
    
    return channel_map, kept_indices


# 测试函数
def test_siamese_models():
    """测试孪生网络模型"""
    # 设置随机种子
    torch.manual_seed(42)
    
    # 输入形状
    input_shape = (110, 100, 10)
    batch_size = 4
    
    # 创建随机输入数据
    x1 = torch.randn(batch_size, *input_shape)
    x2 = torch.randn(batch_size, *input_shape)
    
    print("=== 测试 SiameseBrainAuth 模型 ===")
    print(f"输入 x1 形状: {x1.shape}")
    print(f"输入 x2 形状: {x2.shape}")
    
    # 创建模型
    model = SiameseBrainAuth(input_shape=input_shape)
    
    # 前向传播
    output = model(x1, x2)
    print(f"输出形状: {output.shape}")
    print(f"输出样本: {output[:2]}")
    
    print("\n=== 测试 LightSiameseBrainAuth 模型 ===")
    # 创建轻量级模型
    light_model = LightSiameseBrainAuth(input_shape=input_shape)
    
    # 前向传播
    light_output = light_model(x1, x2)
    print(f"轻量级模型输出形状: {light_output.shape}")
    print(f"轻量级模型输出样本: {light_output[:2]}")
    
    # 提取嵌入向量
    embedding = light_model.get_embedding(x1)
    print(f"嵌入向量形状: {embedding.shape}")
    
    # 打印模型信息
    print("\n=== 模型信息 ===")
    print(f"SiameseBrainAuth 参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"LightSiameseBrainAuth 参数数量: {sum(p.numel() for p in light_model.parameters())}")
    
    print("测试完成!")

def test_p3dcnn():
    """测试P3DCNN模型"""
    print("\n=== 测试 P3DCNN 模型 ===")
    
    # 获取当前设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型实例
    input_shape = (110, 100, 10)  # 高度, 宽度, 频带数
    model = P3DCNN(input_shape=input_shape, num_classes=2).to(device)
    
    # 创建一批测试数据并移至设备
    batch_size = 4
    x1 = torch.randn(batch_size, *input_shape).to(device)
    x2 = torch.randn(batch_size, *input_shape).to(device)
    
    # 前向传播
    print(f"输入 x1 形状: {x1.shape}")
    print(f"输入 x2 形状: {x2.shape}")
    
    output = model(x1, x2)
    print(f"输出形状: {output.shape}")
    
    # 测试后向传播
    labels = torch.randint(0, 2, (batch_size,)).to(device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, labels)
    print(f"损失值: {loss.item()}")
    
    # 梯度反向传播
    loss.backward()
    print("梯度反向传播成功")
    
    # 测试获取参数梯度
    total_params = sum(p.numel() for p in model.parameters())
    params_with_grad = sum(p.numel() for p in model.parameters() if p.grad is not None)
    print(f"总参数数量: {total_params}")
    print(f"有梯度的参数数量: {params_with_grad}")

def inspect_model_architecture(model, name):
    """检查并打印模型架构"""
    print(f"\n=== {name} 架构 ===")
    print(model)
    
    # 计算模型参数总量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 计算模型大小(MB)
    model_size = total_params * 4 / (1024 * 1024)  # 假设每个参数是4字节
    print(f"模型大小: {model_size:.2f} MB")


if __name__ == "__main__":
    # 测试两个模型
    print("开始测试BrainAuth模型...")
    
    # 检测CUDA设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # 创建模型实例用于架构检查
    input_shape = (110, 100, 10)
    # p3dcnn_model = P3DCNN(input_shape=input_shape, num_classes=2).to(device)
    light_siamese_model = LightSiameseBrainAuth(input_shape=input_shape).to(device)
    siamese_model = SiameseBrainAuth(input_shape=input_shape).to(device)
    
    # 检查模型架构
    # inspect_model_architecture(p3dcnn_model, "P3DCNN模型")
    inspect_model_architecture(siamese_model, "SiameseBrainAuth模型")
    inspect_model_architecture(light_siamese_model, "LightSiameseBrainAuth模型")
    
    # 运行模型测试
    # test_p3dcnn()
    # test_siamese_brainauth()
    test_siamese_models()
    
    print("\n所有测试完成!")
    
    # 清理GPU内存
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU内存已清理，当前使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")