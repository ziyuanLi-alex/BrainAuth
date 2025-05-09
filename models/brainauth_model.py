"""
BrainAuth Model: 基于空-频特征图学习的脑电图验证模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import yaml
import os

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
    """
    孪生网络结构的BrainAuth模型
    使用共享权重的网络处理一对EEG样本
    """
    
    def __init__(self, input_shape=(110, 100, 10)):
        """
        初始化孪生网络
        
        参数:
            input_shape: 输入空-频特征图的形状 (高度, 宽度, 频带数)
        """
        super(SiameseBrainAuth, self).__init__()
        
        # 共享的特征提取网络
        self.feature_extractor = nn.Sequential(
            # 空间方向卷积
            nn.Conv3d(1, 16, kernel_size=(5, 5, 1), padding='same'),
            nn.ReLU(),
            
            # 频域方向卷积
            nn.Conv3d(16, 32, kernel_size=(1, 1, 5), padding='same'),
            nn.ReLU(),
            
            # 三维卷积-池化模块
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding='same'),
            nn.ReLU(),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            
            nn.Dropout(0.5)
        )
        # self.feature_extractor = nn.Sequential(
        #     # 空间方向卷积（保持原设计）
        #     nn.Conv3d(1, 16, kernel_size=(5, 5, 1), padding='same'),
        #     nn.ReLU(),
            
        #     # 频域方向卷积（保持原设计）
        #     nn.Conv3d(16, 32, kernel_size=(1, 1, 5), padding='same'),
        #     nn.ReLU(),
            
        #     # 三维卷积模块（简化通道数并减少重复结构）
        #     nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=2, padding=1),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(),
            
        #     # 使用深度可分离卷积替代常规卷积
        #     nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding='same', groups=64),
        #     nn.Conv3d(64, 128, kernel_size=1),
        #     nn.ReLU(),
            
        #     # 最终下采样层（减少输出通道数）
        #     nn.Conv3d(128, 192, kernel_size=(3, 3, 3), stride=2, padding=1),
        #     nn.BatchNorm3d(192),
        #     nn.ReLU(),
            
        #     nn.Dropout(0.4)  # 微调dropout率
        # )
        
        # 计算特征提取后的尺寸
        dummy_input = torch.zeros(1, 1, *input_shape)
        dummy_output = self.feature_extractor(dummy_input)
        feature_size = dummy_output.numel()
        
        # 全连接层进行特征比较
        self.fc = nn.Sequential(
            nn.Linear(feature_size * 2, 512),  # 两个特征向量拼接
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 二分类问题（同一人/不同人）
        )
    
    def forward_one(self, x):
        """处理单个输入样本"""
        if x.dim() == 4:
            x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # 展平
        return x
    
    def forward(self, x1, x2):
        """
        处理一对输入样本
        
        参数:
            x1, x2: 一对EEG样本
            
        返回:
            output: 相似度得分，表示两个样本属于同一人的概率
        """
        # 提取特征
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)
        
        # 拼接特征
        combined = torch.cat((feat1, feat2), 1)
        
        # 计算相似度
        output = torch.sigmoid(self.fc(combined))
        
        return output

#TODO: 生成dummy数据测试向量维度

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

class LightP3DCNN(nn.Module):
    """轻量级P3DCNN模型，减少90%以上参数"""
    
    def __init__(self, input_shape=(110, 100, 10), num_classes=2):
        super(LightP3DCNN, self).__init__()
        self.input_shape = input_shape
        
        # 减少通道数和卷积核尺寸
        # 空间方向卷积: 从16通道降至8通道，核尺寸从5x5降至3x3
        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=8,  # 原16，减少一半
            kernel_size=(3, 3, 1),  # 原(5, 5, 1)
            stride=1,
            padding='same'
        )
        
        # 频域方向卷积: 从32通道降至16通道，核尺寸从5降至3
        self.conv2 = nn.Conv3d(
            in_channels=8,  # 原16
            out_channels=16,  # 原32
            kernel_size=(1, 1, 3),  # 原(1, 1, 5)
            stride=1,
            padding='same'
        )
        
        # 简化三维卷积-池化模块
        # 从64通道降至32通道
        self.conv3 = nn.Conv3d(
            in_channels=16,  # 原32
            out_channels=32,  # 原64
            kernel_size=(3, 3, 2),  # 减小卷积核尺寸
            stride=2,
            padding=1
        )
        self.bn1 = nn.BatchNorm3d(32)  # 原64
        
        # 直接跳过中间卷积层(conv4, conv5)
        # 最后一个卷积层: 从256通道降至64通道
        self.conv6 = nn.Conv3d(
            in_channels=32,  # 原128
            out_channels=64,  # 原256
            kernel_size=(3, 3, 2),  # 减小卷积核尺寸
            stride=2,
            padding=1
        )
        self.bn2 = nn.BatchNorm3d(64)  # 原256
        self.dropout = nn.Dropout(0.3)  # 降低dropout比例以加快训练
        
        # 计算卷积后的特征尺寸
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # 全连接层减少神经元数量
        self.fc = nn.Linear(conv_output_size, 64)  # 原256，减少到64
        
        # 输出层
        self.out = nn.Linear(64, num_classes)  # 原256->2，现在64->2
    
    def _get_conv_output_size(self, shape):
        """计算卷积层输出的特征尺寸"""
        batch_size = 1
        input_tensor = torch.zeros(batch_size, 1, *shape)
        output = self._forward_conv(input_tensor)
        return output.numel() // batch_size
    
    def _forward_conv(self, x):
        """前向传播卷积部分"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # 三维卷积-池化模块
        x = F.relu(self.bn1(self.conv3(x)))
        # 去掉原来的conv4和conv5
        x = F.relu(self.bn2(self.conv6(x)))
        x = self.dropout(x)
        return x
    
    def forward(self, x1, x2):
        """前向传播"""
        # 确保输入形状正确 (batch_size, channels, depth, height, width)
        if x1.dim() == 4:
            x1 = x1.unsqueeze(1)
        if x2.dim() == 4:
            x2 = x2.unsqueeze(1)
        
        # 分别通过卷积层提取特征
        x1_conv = self._forward_conv(x1)
        x2_conv = self._forward_conv(x2)
        
        # 特征差异计算
        x_diff = torch.abs(x1_conv - x2_conv)
        x_flat = x_diff.view(x_diff.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc(x_flat))
        
        # 输出层
        output = self.out(x)
        
        return output



def test_siamese_brainauth():
    """测试SiameseBrainAuth模型"""
    print("\n=== 测试 SiameseBrainAuth 模型 ===")
    
    # 获取当前设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型实例
    input_shape = (110, 100, 10)
    model = SiameseBrainAuth(input_shape=input_shape).to(device)
    
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
    labels = torch.randint(0, 2, (batch_size, 1)).float().to(device)
    criterion = nn.BCELoss()
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
    p3dcnn_model = P3DCNN(input_shape=input_shape, num_classes=2).to(device)
    siamese_model = SiameseBrainAuth(input_shape=input_shape).to(device)
    
    # 检查模型架构
    inspect_model_architecture(p3dcnn_model, "P3DCNN模型")
    inspect_model_architecture(siamese_model, "SiameseBrainAuth模型")
    
    # 运行模型测试
    test_p3dcnn()
    test_siamese_brainauth()
    
    print("\n所有测试完成!")
    
    # 清理GPU内存
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU内存已清理，当前使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")