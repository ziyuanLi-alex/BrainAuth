"""
EEG空间-频率转换模块: 将时序EEG数据转换为空间-频率特征图
"""

import numpy as np
import torch
import logging
import mne
from scipy import signal
from scipy.interpolate import griddata
from typing import List, Dict, Tuple, Optional, Union
import time
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BrainAuth')

class EEGSpectralConverter:
    """
    将时序EEG数据转换为空间-频率特征图
    基于论文《基于空-频特征图学习三维卷积神经网络的运动想象脑电解码方法》
    """
    
    def __init__(
        self, 
        channel_positions: Optional[Dict[str, Tuple[float, float]]] = None,
        freq_bands: List[Tuple[float, float]] = None,
        fs: float = 160.0,
        output_shape: Tuple[int, int, int] = (110, 100, 10),
        mapping_method: str = 'cubic',
        use_log: bool = True
    ):
        """
        初始化EEG空间-频率转换器
        
        参数:
            channel_positions: 电极位置字典，键为通道名，值为(x,y)坐标
            freq_bands: 频段列表，每个元素为(最低频率, 最高频率)
            fs: 采样频率
            output_shape: 输出特征图形状(高度, 宽度, 频带数)
            mapping_method: 插值方法，可选'cubic', 'linear', 'nearest'
            use_log: 是否对功率谱应用对数变换
        """
        self.fs = fs
        self.output_shape = output_shape
        self.mapping_method = mapping_method
        self.use_log = use_log
        
        # 默认使用10-15Hz频段，以0.5Hz为步长
        if freq_bands is None:
            self.freq_bands = [(10.0 + i * 0.5, 10.0 + (i + 1) * 0.5) for i in range(10)]
        else:
            self.freq_bands = freq_bands
        
        # 初始化电极位置
        if channel_positions is None:
            self.channel_positions = self._get_default_channel_positions()
        else:
            self.channel_positions = channel_positions
            
        # 为插值创建网格
        self.grid_x, self.grid_y = np.mgrid[0:output_shape[0], 0:output_shape[1]]
        
        logger.info(f"初始化EEG空间-频率转换器，频段数: {len(self.freq_bands)}")
        
    def _get_default_channel_positions(self) -> Dict[str, Tuple[float, float]]:
        """获取默认的电极位置（基于10-10系统）"""
        # 创建标准10-10系统的电极位置
        montage = mne.channels.make_standard_montage('standard_1020')
        pos_dict = {}
        
        # 获取电极3D位置并投影到2D平面
        pos_3d = montage.get_positions()['ch_pos']
        for ch_name, pos in pos_3d.items():
            # 简化的2D投影，使用x和y坐标
            pos_dict[ch_name] = (pos[0], pos[1])
            
        return pos_dict
    
    def _compute_psd(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算EEG信号的功率谱密度
        
        参数:
            eeg_data: EEG数据，形状为[通道数, 时间点数]
            
        返回:
            freqs: 频率点
            psd: 功率谱密度，形状为[通道数, 频率点数]
        """
        n_channels = eeg_data.shape[0]
        psds = []
        
        # 为每个通道计算PSD
        for ch in range(n_channels):
            # 使用Welch方法估计PSD
            freqs, psd = signal.welch(
                eeg_data[ch], fs=self.fs, nperseg=min(256, eeg_data.shape[1]),
                scaling='density'
            )
            psds.append(psd)
            
        # 转换为numpy数组
        psds = np.array(psds)
        
        # 应用对数变换（可选）
        if self.use_log:
            # 防止log(0)
            psds = np.log10(psds + 1e-10)
            
        return freqs, psds
    
    def _extract_band_power(self, freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
        """
        提取特定频段的功率
        
        参数:
            freqs: 频率点
            psd: 功率谱密度，形状为[通道数, 频率点数]
            band: 频段范围(low, high)
            
        返回:
            band_power: 频段功率，形状为[通道数]
        """
        # 找到频段对应的索引
        idx = np.logical_and(freqs >= band[0]-0.3, freqs <= band[1]+0.5)
        
        # 确保有足够的频率点
        if np.sum(idx) <= 1:
            logger.warning(f"频段 {band} 内频率点太少，可能导致功率估计不准确")
            #扩大搜索范围
            margin = 0.5  # Hz
            idx = np.logical_and(freqs >= band[0]-margin, freqs <= band[1]+margin)
        
        # 计算频段功率（积分）
        band_power = np.trapz(psd[:, idx], freqs[idx], axis=1)
        
        return band_power
    
    def _create_topo_map(self, channel_values: np.ndarray) -> np.ndarray:
        """
        创建地形图
        
        参数:
            channel_values: 通道值，形状为[通道数]
            
        返回:
            topo_map: 地形图，形状为[height, width]
        """
        height, width = self.output_shape[0], self.output_shape[1]
        
        # 获取通道位置
        channels = list(self.channel_positions.keys())[:len(channel_values)]  # 确保匹配数量
        points = np.array([self.channel_positions[ch] for ch in channels])
        
        # 归一化点坐标到网格范围
        points_norm = np.zeros_like(points)
        points_norm[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * (height - 1)
        points_norm[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * (width - 1)
        
        # 使用插值创建地形图
        topo_map = griddata(
            points_norm, channel_values, (self.grid_x, self.grid_y), 
            method=self.mapping_method, fill_value=0
        )
        
        return topo_map
    
    def convert(self, eeg_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        将EEG数据转换为空间-频率特征图
        
        参数:
            eeg_data: EEG数据，形状为[通道数, 时间点数]或[批量大小, 通道数, 时间点数]
            
        返回:
            freq_maps: 空间-频率特征图，形状为[高度, 宽度, 频带数]或[批量大小, 高度, 宽度, 频带数]
        """
        # 转换为numpy数组
        if isinstance(eeg_data, torch.Tensor):
            eeg_np = eeg_data.cpu().numpy()
        else:
            eeg_np = eeg_data
            
        # 处理批量输入
        if eeg_np.ndim == 3:
            batch_size = eeg_np.shape[0]
            results = []
            for i in range(batch_size):
                results.append(self.convert(eeg_np[i]))
            return torch.stack(results)
        
        # 计算功率谱
        freqs, psd = self._compute_psd(eeg_np)
        
        # 为每个频段生成地形图
        topo_maps = []
        for band in self.freq_bands:
            try:
                # 提取频段功率
                band_power = self._extract_band_power(freqs, psd, band)
                
                # 创建地形图
                topo_map = self._create_topo_map(band_power)
                topo_maps.append(topo_map)
            except Exception as e:
                logger.error(f"处理频段 {band} 时出错: {e}")
                # 创建零填充的地形图
                topo_map = np.zeros((self.output_shape[0], self.output_shape[1]))
                topo_maps.append(topo_map)
        
        # 堆叠所有频段的地形图
        freq_maps = np.stack(topo_maps, axis=-1)
        
        # 确保值域合理且无NaN值
        freq_maps = np.nan_to_num(freq_maps, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 转换为PyTorch张量
        return torch.tensor(freq_maps, dtype=torch.float32)
    
    def set_mapping_method(self, method: str):
        """设置插值方法"""
        if method in ['cubic', 'linear', 'nearest']:
            self.mapping_method = method
        else:
            logger.warning(f"不支持的插值方法: {method}，使用默认的'cubic'")
            self.mapping_method = 'cubic'
            
    def set_freq_bands(self, freq_bands: List[Tuple[float, float]]):
        """设置频段"""
        self.freq_bands = freq_bands
        logger.info(f"已更新频段: {len(self.freq_bands)}个")


def test_eeg_spectral_converter():
    """测试EEG空间-频率转换模块的功能"""
    logger.info("开始测试EEG空间-频率转换模块...")
    
    # 设置MNE日志级别
    mne.set_log_level("WARNING")
    
    # 创建随机EEG数据
    n_channels = 64
    n_times = 320
    
    # 测试单个样本
    logger.info("测试单个样本转换...")
    eeg_single = np.random.randn(n_channels, n_times)
    
    # 初始化转换器
    converter = EEGSpectralConverter(fs=160.0)
    
    # 计时
    start_time = time.time()
    
    # 转换数据
    freq_map_single = converter.convert(eeg_single)
    
    # 打印结果
    logger.info(f"单个样本转换完成，耗时: {time.time() - start_time:.4f}秒")
    logger.info(f"输入形状: {eeg_single.shape}")
    logger.info(f"输出形状: {freq_map_single.shape}")
    
    # 测试批处理
    logger.info("\n测试批处理转换...")
    batch_size = 4
    eeg_batch = np.random.randn(batch_size, n_channels, n_times)
    
    # 计时
    start_time = time.time()
    
    # 转换批数据
    freq_map_batch = converter.convert(eeg_batch)
    
    # 打印结果
    logger.info(f"批处理转换完成，耗时: {time.time() - start_time:.4f}秒")
    logger.info(f"输入形状: {eeg_batch.shape}")
    logger.info(f"输出形状: {freq_map_batch.shape}")
    
    # 测试不同的频段配置
    logger.info("\n测试不同频段配置...")
    custom_bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
    converter.set_freq_bands(custom_bands)
    
    # 转换数据
    freq_map_custom = converter.convert(eeg_single)
    
    # 打印结果
    logger.info(f"自定义频段转换完成")
    logger.info(f"频段数量: {len(custom_bands)}")
    logger.info(f"输出形状: {freq_map_custom.shape}")
    
    # 测试不同的映射方法
    logger.info("\n测试不同映射方法...")
    
    methods = ['linear', 'nearest', 'cubic']
    for method in methods:
        converter.set_mapping_method(method)
        start_time = time.time()
        freq_map = converter.convert(eeg_single)
        logger.info(f"映射方法 '{method}' 转换完成，耗时: {time.time() - start_time:.4f}秒")
    
    logger.info("\n测试异常处理...")
    # 测试极端情况 - 只有一个采样点
    try:
        eeg_short = np.random.randn(n_channels, 1)
        freq_map_short = converter.convert(eeg_short)
        logger.info(f"极短信号处理成功，输出形状: {freq_map_short.shape}")
    except Exception as e:
        logger.error(f"极短信号处理失败: {e}")
    
    logger.info("\nEEG空间-频率转换模块测试完成!")
    
    return converter


def test_with_torch_tensors():
    """使用PyTorch张量测试转换功能"""
    logger.info("\n测试PyTorch张量输入...")
    
    # 创建随机EEG数据
    n_channels = 64
    n_times = 320
    batch_size = 4
    
    # 创建PyTorch张量
    eeg_tensor_single = torch.randn(n_channels, n_times)
    eeg_tensor_batch = torch.randn(batch_size, n_channels, n_times)
    
    # 初始化转换器
    converter = EEGSpectralConverter(fs=160.0)
    
    # 测试单个样本
    start_time = time.time()
    freq_map_single = converter.convert(eeg_tensor_single)
    logger.info(f"PyTorch单样本转换完成，耗时: {time.time() - start_time:.4f}秒")
    logger.info(f"输入形状: {eeg_tensor_single.shape}")
    logger.info(f"输出形状: {freq_map_single.shape}")
    
    # 测试批处理
    start_time = time.time()
    freq_map_batch = converter.convert(eeg_tensor_batch)
    logger.info(f"PyTorch批处理转换完成，耗时: {time.time() - start_time:.4f}秒")
    logger.info(f"输入形状: {eeg_tensor_batch.shape}")
    logger.info(f"输出形状: {freq_map_batch.shape}")
    
    return freq_map_batch


def test_gpu_support():
    """测试GPU支持"""
    if torch.cuda.is_available():
        logger.info("\n测试GPU支持...")
        device = torch.device("cuda")
        
        # 创建随机EEG数据
        n_channels = 64
        n_times = 320
        
        # 创建PyTorch张量并移至GPU
        eeg_tensor = torch.randn(n_channels, n_times).to(device)
        
        # 初始化转换器
        converter = EEGSpectralConverter(fs=160.0)
        
        # 计时
        start_time = time.time()
        
        # 转换数据
        freq_map = converter.convert(eeg_tensor)
        
        # 检查输出是否在CPU上
        logger.info(f"输出设备: {freq_map.device}")
        
        # 打印结果
        logger.info(f"GPU数据转换完成，耗时: {time.time() - start_time:.4f}秒")
        logger.info(f"输入形状: {eeg_tensor.shape}")
        logger.info(f"输出形状: {freq_map.shape}")
    else:
        logger.info("无可用GPU，跳过GPU支持测试")


def print_system_info():
    """打印系统信息"""
    logger.info("\n系统信息:")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"NumPy版本: {np.__version__}")
    logger.info(f"MNE版本: {mne.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU设备: {torch.cuda.get_device_name(0)}")
        logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
        logger.info(f"当前GPU内存: {torch.cuda.memory_allocated(0) / 1024 / 1024:.2f} MB")
    else:
        logger.info("CUDA不可用")


if __name__ == "__main__":
    # 打印系统信息
    print_system_info()
    
    # 运行测试
    converter = test_eeg_spectral_converter()
    
    # 测试PyTorch张量
    freq_maps = test_with_torch_tensors()
    
    # 测试GPU支持
    test_gpu_support()
    
    # 打印最终内存使用
    if torch.cuda.is_available():
        logger.info(f"\nGPU内存使用: {torch.cuda.memory_allocated(0) / 1024 / 1024:.2f} MB")
        # 清理GPU内存
        torch.cuda.empty_cache()
        logger.info(f"清理后GPU内存使用: {torch.cuda.memory_allocated(0) / 1024 / 1024:.2f} MB")