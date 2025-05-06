"""
BrainAuth Model: 基于空-频特征图学习的脑电图验证模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # 三维卷积-池化模块
        x = F.relu(self.bn1(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.bn2(self.conv6(x)))
        x = self.dropout(x)
        return x
    
    def forward(self, x1, x2):
        """
        前向传播
        
        参数:
            x1: 第一个EEG样本的空-频特征图
            x2: 第二个EEG样本的空-频特征图
            
        返回:
            output: 分类输出（同一人/不同人）
        """
        # 确保输入形状正确 (batch_size, channels, depth, height, width)
        if x1.dim() == 4:
            x1 = x1.unsqueeze(1)
        if x2.dim() == 4:
            x2 = x2.unsqueeze(1)
        
        # 分别通过卷积层提取特征
        x1_conv = self._forward_conv(x1)
        x2_conv = self._forward_conv(x2)
        
        # 特征拼接或计算差异
        # 方法1: 欧氏距离
        x_diff = torch.abs(x1_conv - x2_conv)
        x_flat = x_diff.view(x_diff.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc(x_flat))
        
        # 输出层
        output = self.out(x)
        
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