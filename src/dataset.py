import os
import numpy as np
import mne
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import yaml
import random
import logging
from EEGSpectralConverter import EEGSpectralConverter
from EEGHDFCache import EEGHDFCache  # 添加正确的导入
import hashlib
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BrainAuth')

# 设置日志级别为警告或错误，以减少输出
mne.set_log_level("WARNING")  # 或者使用 "ERROR" 以获得更少的输出

class BrainAuthDataset(Dataset):
    """脑电图身份认证数据集
    
    针对验证(verification)场景的数据集，每个样本包含一对EEG数据，
    标签表示它们是否来自同一受试者。
    """

    def __init__(
        self, 
        data_dir: str,
        subject_ids: Optional[List[int]] = None,
        condition: str = 'eyes_open',
        preprocess_params: Dict = None,
        pos_ratio: float = 0.5,
        cache: bool = False,
        seed: int = 42,
        mode: str = 'train',  # 数据集模式(train/val/test)
        use_hdf5_cache: bool = True,  # 是否使用HDF5缓存
        hdf5_cache_dir: str = './data/hdf_cache',  # HDF5缓存目录
        reset_cache: bool = False  # 是否重置缓存
    ):
        """初始化数据集"""
        self.data_dir = data_dir
        self.condition = condition
        self.mode = mode
        self.pos_ratio = pos_ratio
        
        # 缓存相关设置
        self.memory_cache = cache
        self.cached_data = {}
        self.use_hdf5_cache = use_hdf5_cache
        
        # 缓存命中统计
        self.memory_hits = 0
        self.hdf5_hits = 0
        self.misses = 0
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)

        # 设置默认预处理参数
        default_params = {
            'l_freq': 1.0,        # 高通滤波频率
            'h_freq': None,       # 低通滤波频率(默认不使用)
            'segment_duration': 2.0,  # 每个样本的持续时间（秒）
            'overlap': 0.5,       # 样本间的重叠比例
            'normalize': True,    # 是否进行标准化
            'channels': None,     # 要使用的通道列表(默认全部使用)
            'resample': None,     # 降采样频率(Hz)，默认不降采样
        }

        # 更新预处理参数
        if preprocess_params is None:
            self.preprocess_params = default_params
        else:
            self.preprocess_params = {**default_params, **preprocess_params}
        
        # 获取所有受试者
        processed_dir = os.path.join(data_dir, 'processed', condition)
        all_subjects = [int(d.split('-')[1]) for d in os.listdir(processed_dir) 
                       if os.path.isdir(os.path.join(processed_dir, d)) and d.startswith('sub-')]
        
        # 过滤受试者
        if subject_ids is not None:
            self.subjects = [s for s in subject_ids if s in all_subjects]
        else:
            self.subjects = all_subjects

        self.subjects.sort()  # 确保顺序一致
        
        # 创建样本列表（每个受试者的EEG段）
        self.segments = []
        for subject_id in self.subjects:
            subject_dir = os.path.join(processed_dir, f'sub-{subject_id}')
            eeg_file = os.path.join(subject_dir, f'sub-{subject_id}_eeg.edf')
            
            # 检查文件是否存在
            if not os.path.exists(eeg_file):
                print(f"警告：无法找到{eeg_file}")
                continue

            # 读取数据以确定段数
            raw = mne.io.read_raw_edf(eeg_file, preload=False)
            duration = raw.times[-1]
            segment_duration = self.preprocess_params['segment_duration']
            overlap = self.preprocess_params['overlap']
            
            # 计算可以提取的段数
            step = segment_duration * (1 - overlap)
            num_segments = int((duration - segment_duration) / step) + 1
            
            # 为每个段添加样本
            for seg_idx in range(num_segments):
                start_time = seg_idx * step
                self.segments.append({
                    'subject_id': subject_id,
                    'eeg_file': eeg_file,
                    'start_time': start_time,
                    'end_time': start_time + segment_duration
                })
        
        # 生成样本对（正样本和负样本）
        self.pairs = self._generate_pairs()
        
        # 初始化HDF5缓存
        if self.use_hdf5_cache:
            self.hdf5_cache = EEGHDFCache(
                cache_dir=hdf5_cache_dir,
                mode=mode,
                condition=condition,
                overwrite=reset_cache
            )
            logger.info(f"HDF5缓存已初始化: {hdf5_cache_dir}")
            
            # 如果启用了缓存预处理，则先处理部分数据
            if preprocess_params and preprocess_params.get('preprocess_cache', False):
                self._prepare_hdf5_cache()

    def _prepare_hdf5_cache(self):
        """为HDF5缓存预处理样本数据"""
        if not self.use_hdf5_cache:
            return
            
        # 统计需要缓存的段
        unique_segments = set()
        segments_to_cache = []
        
        # 收集随机的段
        # 优先缓存前2000个pair中的段
        pairs_to_check = self.pairs[:min(2000, len(self.pairs))]
        
        for pair in pairs_to_check:
            for segment in [pair['segment1'], pair['segment2']]:
                # 创建段的唯一标识
                seg_id = f"{segment['subject_id']}_{segment['start_time']}"
                
                if seg_id not in unique_segments:
                    unique_segments.add(seg_id)
                    
                    # 检查缓存是否存在
                    if not self.hdf5_cache.contains(segment, 'spectral'):
                        segments_to_cache.append(segment)
        
        if not segments_to_cache:
            logger.info(f"所有初始段已缓存到HDF5")
            return
            
        # 生成缓存
        logger.info(f"预处理并缓存{len(segments_to_cache)}个段到HDF5...")
        
        # 批量处理EEG数据
        eeg_data_list = []
        for segment in tqdm(segments_to_cache, desc="加载EEG数据"):
            # 加载EEG数据
            eeg_data = self._load_and_preprocess(
                segment['eeg_file'], segment['start_time'], segment['end_time']
            )
            eeg_data_list.append(eeg_data)
            
        # 批量转换为特征图
        spectral_data_list = []
        for eeg_data in tqdm(eeg_data_list, desc="转换为特征图"):
            spectral_data = self._convert_to_spectral_features(eeg_data)
            spectral_data_list.append(spectral_data)
        
        # 批量存储到HDF5
        success_count = self.hdf5_cache.batch_put(segments_to_cache, spectral_data_list, 'spectral')
        logger.info(f"成功缓存 {success_count}/{len(segments_to_cache)} 个段到HDF5")

    def _generate_pairs(self) -> List[Dict]:
        """生成正负样本对
        
        正样本：来自同一受试者的两个不同EEG段
        负样本：来自不同受试者的两个EEG段
        
        返回:
            包含样本对信息的字典列表
        """

        # 按受试者ID组织段
        segments_by_subject = {}
        for segment in self.segments:
            subject_id = segment['subject_id']
            if subject_id not in segments_by_subject:
                segments_by_subject[subject_id] = []
            segments_by_subject[subject_id].append(segment)
        
        pairs = []

        # 计算正负样本数量
        num_segments = len(self.segments)
        # 总样本对数量 = C(n,2) = n*(n-1)/2，仅使用一部分
        total_pairs = min(num_segments * 5, num_segments * (num_segments - 1) // 2)
        num_pos_pairs = int(total_pairs * self.pos_ratio)
        num_neg_pairs = total_pairs - num_pos_pairs

        # 生成正样本（同一受试者的不同段）
        for subject_id, subject_segments in segments_by_subject.items():
            if len(subject_segments) >= 2:  # 需要至少两个段才能组成对
                # 计算当前受试者可以生成的最大对数
                max_pairs_for_subject = len(subject_segments) * (len(subject_segments) - 1) // 2
                # 按比例分配给每个受试者的正样本数量
                pairs_for_subject = min(
                    max_pairs_for_subject,
                    max(1, int(num_pos_pairs * (len(subject_segments) / num_segments)))
                )
                
                # 随机选择段对
                pairs_indices = []
                for _ in range(pairs_for_subject):
                    while True:
                        i, j = random.sample(range(len(subject_segments)), 2)
                        if (i, j) not in pairs_indices and (j, i) not in pairs_indices:
                            pairs_indices.append((i, j))
                            break
                
                for i, j in pairs_indices:
                    pairs.append({
                        'segment1': subject_segments[i],
                        'segment2': subject_segments[j],
                        'label': 1  # 正样本，同一受试者
                    })

        # 生成负样本（不同受试者的段）
        neg_pairs_count = 0
        subject_ids = list(segments_by_subject.keys())
        
        while neg_pairs_count < num_neg_pairs and len(subject_ids) >= 2:
            # 随机选择两个不同的受试者
            subject1, subject2 = random.sample(subject_ids, 2)
            
            # 从每个受试者中随机选择一个段
            segment1 = random.choice(segments_by_subject[subject1])
            segment2 = random.choice(segments_by_subject[subject2])
            
            # 检查这对样本是否已存在
            pair_exists = False
            for pair in pairs:
                if ((pair['segment1'] == segment1 and pair['segment2'] == segment2) or
                    (pair['segment1'] == segment2 and pair['segment2'] == segment1)):
                    pair_exists = True
                    break
            if not pair_exists:
                pairs.append({
                    'segment1': segment1,
                    'segment2': segment2,
                    'label': 0  # 负样本，不同受试者
                })
                neg_pairs_count += 1
    
        # 打乱样本对顺序
        random.shuffle(pairs)
        
        return pairs

    def __len__(self) -> int:
        """返回数据集中的样本对数量"""
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """获取单个样本对"""
        pair = self.pairs[index]
        segment1 = pair['segment1']
        segment2 = pair['segment2']
        label = pair['label']

        # 加载第一个段的特征图
        spec_data1 = self._load_spectral_data(segment1)
        
        # 加载第二个段的特征图
        spec_data2 = self._load_spectral_data(segment2)
        
        return spec_data1, spec_data2, torch.tensor(label, dtype=torch.long)

    def _load_spectral_data(self, segment):
        """加载段的特征图，使用三层缓存策略：内存 -> HDF5 -> 计算"""
        # 内存缓存键
        cache_key = f"{segment['subject_id']}_{segment['start_time']}"
        
        # 1. 检查内存缓存
        if self.memory_cache and cache_key in self.cached_data:
            self.memory_hits += 1
            return self.cached_data[cache_key]
        
        # 2. 检查HDF5缓存
        if self.use_hdf5_cache:
            spec_data = self.hdf5_cache.get(segment, 'spectral')
            if spec_data is not None:
                # 加入内存缓存
                if self.memory_cache:
                    self.cached_data[cache_key] = spec_data
                
                self.hdf5_hits += 1
                return spec_data
        
        # 3. 计算特征图
        # 加载EEG数据
        eeg_data = self._load_and_preprocess(
            segment['eeg_file'], segment['start_time'], segment['end_time']
        )
        
        # 转换为特征图
        spec_data = self._convert_to_spectral_features(eeg_data)
        
        # 保存到HDF5缓存
        if self.use_hdf5_cache:
            self.hdf5_cache.put(segment, spec_data, 'spectral')
        
        # 保存到内存缓存
        if self.memory_cache:
            self.cached_data[cache_key] = spec_data
        
        self.misses += 1
        return spec_data

    def _convert_to_spectral_features(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """将EEG数据转换为空间-频率特征图，加强错误处理"""
        # 检查输入数据
        if eeg_data is None or eeg_data.numel() == 0:
            logger.error("输入 EEG 数据为空或 None，返回零张量")
            return torch.zeros((110, 100, 10), dtype=torch.float32)
        
        # 检查是否有 NaN 或 Inf
        if torch.isnan(eeg_data).any() or torch.isinf(eeg_data).any():
            logger.warning("输入 EEG 数据含有 NaN 或 Inf 值，将被替换为零")
            eeg_data = torch.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 初始化转换器（如果尚未初始化）
        if not hasattr(self, 'spectral_converter'):
            # 从配置参数中获取频段信息
            freq_bands = None
            if 'freq_bands' in self.preprocess_params:
                freq_bands = self.preprocess_params['freq_bands']
            
            # 确定采样率
            fs = self.preprocess_params.get('resample', 160.0)
            
            # 创建转换器
            try:
                self.spectral_converter = EEGSpectralConverter(
                    fs=fs,
                    freq_bands=freq_bands,
                    output_shape=(110, 100, 10),  # 输出形状
                    mapping_method=self.preprocess_params.get('mapping_method', 'cubic'),
                    use_log=self.preprocess_params.get('use_log', True)
                )
                # logger.info(f"初始化EEG空间-频率转换器，频段数: 10")
            except Exception as e:
                logger.error(f"初始化转换器失败: {e}")
                return torch.zeros((110, 100, 10), dtype=torch.float32)

        try:
            # 转换EEG数据
            # 确保数据格式正确
            if isinstance(eeg_data, np.ndarray):
                eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
            
            # 确保形状正确 - 应该是 [channels, time_points]
            if len(eeg_data.shape) != 2:
                logger.warning(f"EEG 数据形状有误: {eeg_data.shape}，应为 [channels, time_points]")
            
            spectral_features = self.spectral_converter.convert(eeg_data)
            
            # 验证输出
            if spectral_features is None:
                logger.error("转换结果为 None，返回零张量")
                return torch.zeros((110, 100, 10), dtype=torch.float32)
            
            # 检查输出形状
            if spectral_features.shape != (110, 100, 10):
                logger.warning(f"输出特征图形状不正确: {spectral_features.shape}，应为 (110, 100, 10)")
                # 尝试调整形状
                if isinstance(spectral_features, torch.Tensor) and spectral_features.numel() > 0:
                    if spectral_features.numel() == 110 * 100 * 10:
                        spectral_features = spectral_features.reshape(110, 100, 10)
                    else:
                        # 无法恢复，返回零张量
                        logger.error("无法调整形状，返回零张量")
                        return torch.zeros((110, 100, 10), dtype=torch.float32)
            
            return spectral_features
        
        except Exception as e:
            logger.error(f"转换EEG数据到空间-频率特征图时出错: {e}")
            # 打印更详细的错误信息和堆栈跟踪
            import traceback
            logger.error(traceback.format_exc())
            
            # 返回零张量作为后备
            return torch.zeros((110, 100, 10), dtype=torch.float32)

    def _load_and_preprocess(self, eeg_file: str, start_time: float, end_time: float) -> torch.Tensor:
        """加载并预处理EEG数据段，增强错误处理"""
        try:
            # 检查文件是否存在
            if not os.path.exists(eeg_file):
                logger.error(f"EEG文件不存在: {eeg_file}")
                return torch.zeros((64, 256), dtype=torch.float32)
            
            # 加载数据
            raw = mne.io.read_raw_edf(eeg_file, preload=True)
            
            # 检查通道数
            if len(raw.ch_names) == 0:
                logger.error(f"EEG文件没有通道: {eeg_file}")
                return torch.zeros((64, 256), dtype=torch.float32)
            
            # 检查数据时长
            if raw.times[-1] < end_time:
                logger.warning(f"请求的结束时间 {end_time} 超出了数据时长 {raw.times[-1]}，将被截断")
                end_time = raw.times[-1]
            
            if start_time >= end_time:
                logger.error(f"无效的时间范围: {start_time}-{end_time}")
                return torch.zeros((64, 256), dtype=torch.float32)
            
            # 选择特定通道（如果指定）
            if self.preprocess_params['channels'] is not None:
                available_channels = raw.ch_names
                channels_to_use = [ch for ch in self.preprocess_params['channels'] if ch in available_channels]
                if len(channels_to_use) == 0:
                    logger.warning("指定的通道均不可用，使用所有通道")
                else:
                    raw.pick_channels(channels_to_use)
            
            # 降采样（如果指定）
            if self.preprocess_params['resample'] is not None:
                raw.resample(self.preprocess_params['resample'])

            # 高通滤波（默认总是应用）
            raw.filter(
                l_freq=self.preprocess_params['l_freq'], 
                h_freq=self.preprocess_params['h_freq']
            )
            
            # 提取时间段
            time_idx = raw.time_as_index([start_time, end_time])
            start_idx, end_idx = int(time_idx[0]), int(time_idx[1])
            
            # 确保索引有效
            if start_idx >= end_idx or start_idx < 0:
                logger.error(f"无效的时间索引: {start_idx}-{end_idx}")
                return torch.zeros((64, 256), dtype=torch.float32)
            
            data, times = raw[:, start_idx:end_idx]
            
            # 检查数据是否有效
            if data.size == 0:
                logger.error(f"提取的数据为空: {start_time}-{end_time}")
                return torch.zeros((64, 256), dtype=torch.float32)
            
            # 检查是否有NaN或Inf
            if np.isnan(data).any() or np.isinf(data).any():
                logger.warning("数据包含NaN或Inf值，将被替换")
                data = np.nan_to_num(data)
            
            # 标准化（可选）
            if self.preprocess_params['normalize']:
                # 避免除以零
                std = np.std(data, axis=1, keepdims=True)
                std[std < 1e-10] = 1.0  # 避免除以接近零的值
                data = (data - np.mean(data, axis=1, keepdims=True)) / std
            
            # 转换为PyTorch张量
            data_tensor = torch.tensor(data, dtype=torch.float32)
            
            # 验证输出形状
            expected_channels = raw.info['nchan']
            expected_samples = end_idx - start_idx
            
            if data_tensor.shape[0] != expected_channels or data_tensor.shape[1] != expected_samples:
                logger.warning(f"数据形状异常: {data_tensor.shape}，预期: [{expected_channels}, {expected_samples}]")
            
            return data_tensor
            
        except Exception as e:
            logger.error(f"加载并预处理EEG数据时出错: {e}")
            # 打印堆栈跟踪
            import traceback
            logger.error(traceback.format_exc())
            
            # 返回零张量
            return torch.zeros((64, 256), dtype=torch.float32)

    def get_cache_stats(self):
        """获取缓存统计信息"""
        total = self.memory_hits + self.hdf5_hits + self.misses
        if total == 0:
            return "无数据加载记录"
        
        memory_rate = self.memory_hits / total * 100 if total > 0 else 0
        hdf5_rate = self.hdf5_hits / total * 100 if total > 0 else 0
        miss_rate = self.misses / total * 100 if total > 0 else 0
        
        stats = f"\n===== 缓存统计 =====\n"
        stats += f"总加载次数: {total}\n"
        stats += f"内存缓存命中: {self.memory_hits} ({memory_rate:.2f}%)\n"
        stats += f"HDF5缓存命中: {self.hdf5_hits} ({hdf5_rate:.2f}%)\n"
        stats += f"缓存未命中: {self.misses} ({miss_rate:.2f}%)\n"
        
        # 添加HDF5缓存详情
        if self.use_hdf5_cache:
            stats += "\n" + self.hdf5_cache.get_stats()
        
        stats += f"\n======================"
        
        return stats


def get_dataloaders(
    config_path: str,
    train_subjects: Optional[List[int]] = None,
    val_subjects: Optional[List[int]] = None,
    test_subjects: Optional[List[int]] = None
) -> Dict[str, DataLoader]:
    """创建训练、验证和测试数据加载器，支持HDF5缓存"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    data_dir = data_config['data_dir']
    condition = data_config['condition']
    batch_size = data_config['batch_size']
    
    # 缓存相关配置
    memory_cache = data_config.get('cache_data', False)
    use_hdf5_cache = data_config.get('use_hdf5_cache', True)
    hdf5_cache_dir = data_config.get('hdf5_cache_dir', './data/hdf_cache')
    reset_cache = data_config.get('reset_cache', False)
    
    if use_hdf5_cache:
        logger.info(f"启用HDF5缓存，目录: {hdf5_cache_dir}")

    # 如果未指定受试者分组，则自动划分
    if train_subjects is None and val_subjects is None and test_subjects is None:
        # 获取所有受试者
        processed_dir = os.path.join(data_dir, 'processed', condition)
        all_subjects = [int(d.split('-')[1]) for d in os.listdir(processed_dir) 
                      if os.path.isdir(os.path.join(processed_dir, d)) and d.startswith('sub-')]
        all_subjects.sort()
        
        # 随机打乱
        np.random.seed(config.get('seed', 42))
        np.random.shuffle(all_subjects)

        # 按70%/15%/15%分割
        n = len(all_subjects)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        
        train_subjects = all_subjects[:train_size]
        val_subjects = all_subjects[train_size:train_size + val_size]
        test_subjects = all_subjects[train_size + val_size:]

    # 创建数据集
    train_dataset = BrainAuthDataset(
        data_dir=data_dir,
        subject_ids=train_subjects,
        condition=condition,
        preprocess_params=data_config.get('preprocess_params', None),
        cache=memory_cache,
        pos_ratio=data_config.get('pos_ratio', 0.5),
        mode='train',
        use_hdf5_cache=use_hdf5_cache,
        hdf5_cache_dir=hdf5_cache_dir,
        reset_cache=reset_cache
    )
    
    val_dataset = BrainAuthDataset(
        data_dir=data_dir,
        subject_ids=val_subjects,
        condition=condition,
        preprocess_params=data_config.get('preprocess_params', None),
        cache=memory_cache,
        pos_ratio=data_config.get('pos_ratio', 0.5),
        mode='val',
        use_hdf5_cache=use_hdf5_cache,
        hdf5_cache_dir=hdf5_cache_dir,
        reset_cache=reset_cache
    )
    
    test_dataset = BrainAuthDataset(
        data_dir=data_dir,
        subject_ids=test_subjects,
        condition=condition,
        preprocess_params=data_config.get('preprocess_params', None),
        cache=memory_cache,
        pos_ratio=data_config.get('pos_ratio', 0.5),
        mode='test',
        use_hdf5_cache=use_hdf5_cache,
        hdf5_cache_dir=hdf5_cache_dir,
        reset_cache=reset_cache
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.get('num_workers', 2),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.get('num_workers', 2),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.get('num_workers', 2),
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

def process_segment_worker(args):
    """处理单个EEG段的工作函数
    
    参数:
        args: 包含segment和其他必要参数的元组
    
    返回:
        (spec_data, segment)元组或(None, segment)
    """
    segment, data_dir, condition, preprocess_params = args
    
    try:
        # 创建临时数据集实例用于处理
        temp_dataset = BrainAuthDataset(
            data_dir=data_dir,
            subject_ids=[segment['subject_id']],
            condition=condition,
            preprocess_params=preprocess_params,
            cache=False,
            use_hdf5_cache=False
        )
        
        # 加载EEG
        eeg_data = temp_dataset._load_and_preprocess(
            segment['eeg_file'], segment['start_time'], segment['end_time']
        )
        
        # 转换为特征图
        spec_data = temp_dataset._convert_to_spectral_features(eeg_data)
        
        return spec_data, segment
    except Exception as e:
        logger.error(f"处理段时出错: {e}")
        return None, segment

def preprocess_and_cache_all_data(config_path):
    """预处理并缓存所有数据，用于训练前的离线处理"""
    import multiprocessing as mp
    from datetime import datetime
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    data_dir = data_config['data_dir']
    condition = data_config['condition']
    
    # 获取所有受试者
    processed_dir = os.path.join(data_dir, 'processed', condition)
    all_subjects = [int(d.split('-')[1]) for d in os.listdir(processed_dir) 
                  if os.path.isdir(os.path.join(processed_dir, d)) and d.startswith('sub-')]
    all_subjects.sort()
    
    # 划分训练、验证和测试集
    np.random.seed(config.get('seed', 42))
    np.random.shuffle(all_subjects)
    
    n = len(all_subjects)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    train_subjects = all_subjects[:train_size]
    val_subjects = all_subjects[train_size:train_size + val_size]
    test_subjects = all_subjects[train_size + val_size:]
    
    # 创建空数据集以获取所有段
    empty_dataset = BrainAuthDataset(
        data_dir=data_dir,
        subject_ids=all_subjects,
        condition=condition,
        preprocess_params=data_config.get('preprocess_params', None),
        cache=False,
        pos_ratio=0.5,
        mode='all',
        use_hdf5_cache=False
    )
    
    # 收集所有唯一段
    segments = []
    segment_ids = set()
    
    for pair in empty_dataset.pairs:
        for segment in [pair['segment1'], pair['segment2']]:
            seg_id = f"{segment['subject_id']}_{segment['start_time']}"
            if seg_id not in segment_ids:
                segment_ids.add(seg_id)
                segments.append(segment)
    
    logger.info(f"共有 {len(segments)} 个唯一EEG段需要预处理")
    
    # 初始化HDF5缓存
    hdf5_cache_dir = data_config.get('hdf5_cache_dir', './data/hdf_cache')
    hdf5_cache = EEGHDFCache(
        cache_dir=hdf5_cache_dir,
        mode='all',
        condition=condition,
        overwrite=data_config.get('reset_cache', False)
    )
    
    # 准备工作参数
    preprocess_params = data_config.get('preprocess_params', None)
    worker_args = [(seg, data_dir, condition, preprocess_params) for seg in segments]
    
    # 多进程处理
    num_workers = min(mp.cpu_count(), 8)  # 限制最大工作进程数
    logger.info(f"使用 {num_workers} 个进程并行预处理数据")
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_segment_worker, worker_args),
            total=len(segments),
            desc="并行预处理EEG段"
        ))
    
    # 收集有效结果
    valid_results = [(spec, seg) for spec, seg in results if spec is not None]
    spec_data_list = [spec for spec, _ in valid_results]
    valid_segments = [seg for _, seg in valid_results]
    
    logger.info(f"成功预处理 {len(valid_results)}/{len(segments)} 个段")
    
    # 批量存储到HDF5
    if valid_results:
        success_count = hdf5_cache.batch_put(valid_segments, spec_data_list, 'spectral')
        logger.info(f"成功缓存 {success_count}/{len(valid_results)} 个段到HDF5")
    
    logger.info("数据预处理和缓存完成！")


if __name__ == "__main__":
    import argparse
    import time
    import sys
    
    mne.set_log_level("WARNING")  # 或者使用 "ERROR" 以获得更少的输出
    
    # 命令行参数
    parser = argparse.ArgumentParser(description='验证BrainAuth数据集功能')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据根目录')
    parser.add_argument('--condition', type=str, default='eyes_open', help='实验条件: eyes_open或eyes_closed')
    parser.add_argument('--batch_size', type=int, default=4, help='批处理大小')
    parser.add_argument('--n_samples', type=int, default=5, help='要显示的样本数量')
    parser.add_argument('--test_cache', action='store_true', help='测试缓存功能')
    parser.add_argument('--cache_dir', type=str, default='./data/hdf_cache', help='缓存目录')
    parser.add_argument('--reset_cache', action='store_true', help='重置缓存')
    parser.add_argument('--preprocess', action='store_true', help='预处理并缓存所有数据')
    args = parser.parse_args()
    
    print(f"开始验证BrainAuth数据集功能 - 条件: {args.condition}")

    # 创建一个临时配置文件
    import tempfile
    import yaml
    
    config = {
        'data': {
            'data_dir': args.data_dir,
            'condition': args.condition,
            'batch_size': args.batch_size,
            'num_workers': 0,  # 调试时使用单线程
            'cache_data': True,  # 启用内存缓存
            'use_hdf5_cache': args.test_cache,  # 是否测试缓存
            'hdf5_cache_dir': args.cache_dir,
            'reset_cache': args.reset_cache,
            'pos_ratio': 0.5,
            'preprocess_params': {
                'l_freq': 1.0,
                'h_freq': None,
                'segment_duration': 2.0,
                'overlap': 0.5,
                'normalize': True,
                'channels': None,  # 使用所有通道
                'resample': 160.0,  # 设置采样率
                'preprocess_cache': True,  # 预处理缓存
            }
        },
        'seed': 42
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        # 如果选择预处理，则执行预处理
        if args.preprocess:
            print("\n===== 执行预处理 =====")
            preprocess_and_cache_all_data(config_path)
            
        # 测试缓存功能
        elif args.test_cache:
            print("\n===== 测试缓存功能 =====")
            
            # 创建缓存目录
            os.makedirs(args.cache_dir, exist_ok=True)
            
            # 第一次加载数据（此时应该没有缓存）
            start_time = time.time()
            print("第一次加载数据（生成缓存）...")
            dataloaders = get_dataloaders(config_path)
            train_loader = dataloaders['train']
            first_load_time = time.time() - start_time
            print(f"第一次加载耗时: {first_load_time:.2f}秒")
            
            # 获取缓存统计
            if hasattr(train_loader.dataset, 'get_cache_stats'):
                print(train_loader.dataset.get_cache_stats())
            
            # 获取一些样本以触发缓存生成
            print("加载几个批次以生成缓存...")
            for i, batch in enumerate(train_loader):
                if i >= 2:  # 只加载两个批次
                    break
            
            # 第二次加载数据（应该使用缓存）
            config['data']['reset_cache'] = False  # 确保不重置缓存
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f)
                config_path2 = f.name
            
            start_time = time.time()
            print("\n第二次加载数据（应使用缓存）...")
            dataloaders = get_dataloaders(config_path2)
            train_loader = dataloaders['train']
            second_load_time = time.time() - start_time
            print(f"第二次加载耗时: {second_load_time:.2f}秒")
            
            # 再次获取缓存统计
            if hasattr(train_loader.dataset, 'get_cache_stats'):
                print(train_loader.dataset.get_cache_stats())
            
            # 加载一些批次，查看缓存命中率
            print("\n检查缓存命中率...")
            start_time = time.time()
            for i, batch in enumerate(train_loader):
                if i >= 5:  # 只加载5个批次
                    break
            
            batch_time = time.time() - start_time
            print(f"加载5个批次耗时: {batch_time:.2f}秒")
            
            # 最终缓存统计
            if hasattr(train_loader.dataset, 'get_cache_stats'):
                print(train_loader.dataset.get_cache_stats())
            
            # 对比加载时间
            if first_load_time > 0 and second_load_time > 0:
                speedup = first_load_time / second_load_time
                print(f"\n缓存加速比: {speedup:.2f}x")
                if speedup > 1.2:
                    print("缓存功能工作正常! ✓")
                else:
                    print("缓存可能未正常工作，或者加速效果不明显 ✗")
        
        else:
            # 标准数据加载测试
            start_time = time.time()
            print("正在加载数据...")
            dataloaders = get_dataloaders(config_path)
            train_loader = dataloaders['train']
            val_loader = dataloaders['val']
            test_loader = dataloaders['test']
            print(f"数据加载完成，耗时: {time.time() - start_time:.2f}秒")
            
            # 数据集大小信息
            print(f"训练集大小: {len(train_loader.dataset)}个样本")
            print(f"验证集大小: {len(val_loader.dataset)}个样本")
            print(f"测试集大小: {len(test_loader.dataset)}个样本")

            # 查看几个样本
            print(f"\n查看{args.n_samples}个训练样本:")
            for i, batch in enumerate(train_loader):
                eeg1, eeg2, labels = batch
                print(f"批次 {i+1}:")
                print(f"  批次大小: {eeg1.shape[0]}")
                print(f"  EEG1形状: {eeg1.shape}")
                print(f"  EEG2形状: {eeg2.shape}")
                print(f"  标签形状: {labels.shape}")
                
                # 显示批次中的每个样本
                for j in range(min(eeg1.shape[0], 2)):  # 仅显示批次中的前两个样本
                    print(f"    样本 {j+1}:")
                    print(f"      标签: {labels[j].item()}")
                    print(f"      标签含义: {'同一受试者' if labels[j].item() == 1 else '不同受试者'}")
                
                if i >= args.n_samples - 1:
                    break
                    
            print("\n验证完成! 数据集功能工作正常。")
    
    except Exception as e:
        import traceback
        print(f"验证过程中出错: {e}")
        print(traceback.format_exc())
        sys.exit(1)
    
    finally:
        # 删除临时配置文件
        import os
        if os.path.exists(config_path):
            os.remove(config_path)
        if args.test_cache and 'config_path2' in locals() and os.path.exists(config_path2):
            os.remove(config_path2)