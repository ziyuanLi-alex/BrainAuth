import os
import numpy as np
import mne
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import yaml
import random


class BrainAuthDataset(Dataset):
    """脑电图身份认证数据集
    
    针对验证(verification)场景的数据集，每个样本包含一对EEG数据，
    标签表示它们是否来自同一受试者。
    
    属性:
        data_dir: 数据根目录
        subjects: 受试者ID列表
        condition: 实验条件 ('eyes_open' 或 'eyes_closed')
        preprocess_params: 预处理参数
        pairs: 正负样本对列表
        pos_ratio: 正样本(同一受试者)的比例
        cache: 是否缓存数据
        cached_data: 缓存的数据字典
    """

    def __init__(
        self, 
        data_dir: str,
        subject_ids: Optional[List[int]] = None,
        condition: str = 'eyes_open',
        preprocess_params: Dict = None,
        pos_ratio: float = 0.5,
        cache: bool = False,
        seed: int = 42
    ):

        """初始化数据集
    
    参数:
        data_dir: 数据根目录
        subject_ids: 要包含的受试者ID列表，如果为None则包含所有受试者
        condition: 'eyes_open' 或 'eyes_closed'
        preprocess_params: 预处理参数字典
        pos_ratio: 正样本(同一受试者)的比例
        cache: 是否将数据加载到内存中
        seed: 随机种子，用于生成样本对
    """

        self.data_dir = data_dir
        self.condition = condition
        self.cache = cache
        self.cached_data = {}
        self.pos_ratio = pos_ratio
        
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

    def _generate_pairs(self) -> List[Dict]:
        """生成正负样本对
        
        正样本：来自同一受试者的两个不同EEG段
        负样本：来自不同受试者的两个EEG段
        
        返回:
            包含样本对信息的字典列表
        """

        # 按受试者ID组织段，这里会获取所有受试者的所有segment
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
        """获取单个样本对
        
        参数:
            index: 样本索引
            
        返回:
            (eeg_data1, eeg_data2, label): 一对EEG数据张量和对应的标签
        """
        pair = self.pairs[index]
        segment1 = pair['segment1']
        segment2 = pair['segment2']
        label = pair['label']

        # 加载第一个段
        cache_key1 = f"{segment1['eeg_file']}_{segment1['start_time']}_{segment1['end_time']}"
        if self.cache and cache_key1 in self.cached_data:
            eeg_data1 = self.cached_data[cache_key1]
        else:
            eeg_data1 = self._load_and_preprocess(
                segment1['eeg_file'],
                segment1['start_time'],
                segment1['end_time']
            )
            if self.cache:
                self.cached_data[cache_key1] = eeg_data1
        
        # 加载第二个段
        cache_key2 = f"{segment2['eeg_file']}_{segment2['start_time']}_{segment2['end_time']}"
        if self.cache and cache_key2 in self.cached_data:
            eeg_data2 = self.cached_data[cache_key2]
        else:
            eeg_data2 = self._load_and_preprocess(
                segment2['eeg_file'],
                segment2['start_time'],
                segment2['end_time']
            )
            if self.cache:
                self.cached_data[cache_key2] = eeg_data2
        
        return eeg_data1, eeg_data2, torch.tensor(label, dtype=torch.long)


    def _load_and_preprocess(self, eeg_file: str, start_time: float, end_time: float) -> torch.Tensor:
        """加载并预处理EEG数据段
        
        参数:
            eeg_file: EDF文件路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            
        返回:
            preprocessed_data: 预处理后的EEG数据张量
        """
        # 加载数据
        raw = mne.io.read_raw_edf(eeg_file, preload=True)
        
        # 选择特定通道（如果指定）
        if self.preprocess_params['channels'] is not None:
            available_channels = raw.ch_names
            channels_to_use = [ch for ch in self.preprocess_params['channels'] if ch in available_channels]
            if len(channels_to_use) == 0:
                raise ValueError("指定的通道均不可用")
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
        start_idx, end_idx = int(time_idx[0]), int(time_idx[1])  # Convert to Python scalars
        data, times = raw[:, start_idx:end_idx]
        
        # 标准化（可选）
        if self.preprocess_params['normalize']:
            data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-10)
        
        # 转换为PyTorch张量
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        return data_tensor


def get_dataloaders(
    config_path: str,
    train_subjects: Optional[List[int]] = None,
    val_subjects: Optional[List[int]] = None,
    test_subjects: Optional[List[int]] = None
) -> Dict[str, DataLoader]:

    """创建训练、验证和测试数据加载器
    
    参数:
        config_path: 配置文件路径
        train_subjects: 训练集受试者
        val_subjects: 验证集受试者
        test_subjects: 测试集受试者
        
    返回:
        包含'train', 'val', 'test'的DataLoader字典
    """

    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    data_dir = data_config['data_dir']
    condition = data_config['condition']
    batch_size = data_config['batch_size']
    
    # 如果未指定受试者分组，则自动划分
    if train_subjects is None and val_subjects is None and test_subjects is None:
        # 获取所有受试者
        processed_dir = os.path.join(data_dir, 'processed', condition)
        all_subjects = [int(d.split('-')[1]) for d in os.listdir(processed_dir) 
                      if os.path.isdir(os.path.join(processed_dir, d)) and d.startswith('sub-')]
        all_subjects.sort()
        
        # 随机打乱
        np.random.seed(config.get('random_seed', 42))
        np.random.shuffle(all_subjects)

        # 按70%/15%/15%分割
        n = len(all_subjects)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        
        train_subjects = all_subjects[:train_size]
        val_subjects = all_subjects[train_size:train_size + val_size]
        test_subjects = all_subjects[train_size + val_size:]

    # 设置数据集参数
    dataset_class = BrainAuthDataset
    extra_params = {'pos_ratio': data_config.get('pos_ratio', 0.5)}
    

    # 创建数据集
    train_dataset = dataset_class(
        data_dir=data_dir,
        subject_ids=train_subjects,
        condition=condition,
        preprocess_params=data_config.get('preprocess_params', None),
        cache=data_config.get('cache_data', False),
        **extra_params
    )
    
    val_dataset = dataset_class(
        data_dir=data_dir,
        subject_ids=val_subjects,
        condition=condition,
        preprocess_params=data_config.get('preprocess_params', None),
        cache=data_config.get('cache_data', False),
        **extra_params
    )
    
    test_dataset = dataset_class(
        data_dir=data_dir,
        subject_ids=test_subjects,
        condition=condition,
        preprocess_params=data_config.get('preprocess_params', None),
        cache=data_config.get('cache_data', False),
        **extra_params
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


if __name__ == "__main__":
    import argparse
    import time
    
    mne.set_log_level("WARNING")  # 或者使用 "ERROR" 以获得更少的输出
    
    # 命令行参数
    parser = argparse.ArgumentParser(description='验证BrainAuth数据集功能')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据根目录')
    parser.add_argument('--condition', type=str, default='eyes_open', help='实验条件: eyes_open或eyes_closed')

    parser.add_argument('--batch_size', type=int, default=4, help='批处理大小')
    parser.add_argument('--n_samples', type=int, default=5, help='要显示的样本数量')
    args = parser.parse_args()
    
    print(f"开始验证BrainAuth数据集功能 - 条件: {args.condition}")

    #  创建一个临时配置文件
    import tempfile
    import yaml
    
    config = {
        'data': {
            'data_dir': args.data_dir,
            'condition': args.condition,
            'batch_size': args.batch_size,
            'num_workers': 4,  # 调试时使用单线程
            'cache_data': False,
            'pos_ratio': 0.5,
            'preprocess_params': {
                'l_freq': 1.0,
                'h_freq': None,
                'segment_duration': 2.0,
                'overlap': 0.5,
                'normalize': True,
                'channels': None,  # 使用所有通道
                'resample': None,  # 不进行降采样
            }
        },
        'random_seed': 42
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        # 获取数据加载器
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
            
            # 查看通道数和时间点数
            channels, time_points = eeg1.shape[1], eeg1.shape[2]
                
            print(f"  通道数: {channels}")
            print(f"  时间点数: {time_points}")
            print(f"  采样频率估计: {time_points / config['data']['preprocess_params']['segment_duration']:.1f} Hz")
            print("")
            
            if i >= args.n_samples - 1:
                break

        # # 使用DataLoader批处理方式统计
        # if args.task == 'verification':
        #     # 创建专用于统计的DataLoader，使用较大的批次大小加速处理
        #     stats_loader = DataLoader(
        #         train_loader.dataset,
        #         batch_size=128,  # 使用更大的批次
        #         shuffle=False,
        #         num_workers=0
        #     )
            
        #     print("正在统计正负样本比例...")
        #     pos_count = 0
        #     total_count = 0
            
        #     # 使用批处理方式统计
        #     for _, _, labels in stats_loader:
        #         pos_count += (labels == 1).sum().item()
        #         total_count += labels.size(0)
        #         print(pos_count, total_count)
            
        #     neg_count = total_count - pos_count
        #     print(f"\n训练集正负样本统计:")
        #     print(f"  正样本(同一受试者): {pos_count}个 ({pos_count/total_count*100:.1f}%)")
        #     print(f"  负样本(不同受试者): {neg_count}个 ({neg_count/total_count*100:.1f}%)")
        
        print("\n验证完成! 数据集功能工作正常。")
    
    finally:
        # 删除临时配置文件
        import os
        if os.path.exists(config_path):
            os.remove(config_path)


