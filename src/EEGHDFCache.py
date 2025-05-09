import os
import h5py
import numpy as np
import torch
import hashlib
from tqdm import tqdm
from pathlib import Path
import logging
from datetime import datetime  # 添加缺失的导入

class EEGHDFCache:
    """基于HDF5的脑电图数据缓存系统"""
    
    def __init__(
        self,
        cache_dir: str = './data/hdf_cache',
        mode: str = 'train',
        condition: str = 'eyes_open',
        overwrite: bool = False
    ):
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.condition = condition
        self.overwrite = overwrite
        self.logger = logging.getLogger('BrainAuth')
        
        # 缓存命中统计
        self.hits = 0
        self.misses = 0
        
        # 确保缓存目录存在
        self.cache_path = self.cache_dir / f"{self.mode}_{self.condition}.h5"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 初始化或打开HDF5文件
        self._init_hdf5()
    
    def _init_hdf5(self):
        """初始化HDF5文件"""
        if self.overwrite and self.cache_path.exists():
            self.logger.info(f"覆盖现有HDF5缓存: {self.cache_path}")
            # 如果文件存在且需要覆盖，先删除
            self.cache_path.unlink()
        
        # 如果文件不存在，创建新文件
        if not self.cache_path.exists():
            self.logger.info(f"创建新的HDF5缓存: {self.cache_path}")
            with h5py.File(self.cache_path, 'w') as f:
                # 创建基本组结构
                f.create_group('raw_eeg')  # 存储原始EEG数据
                f.create_group('spectral_features')  # 存储频谱特征
                # 创建元数据组
                metadata = f.create_group('metadata')
                # 添加一些元数据
                metadata.attrs['created_time'] = np.string_(str(datetime.now()))
                metadata.attrs['mode'] = np.string_(self.mode)
                metadata.attrs['condition'] = np.string_(self.condition)
        
        self.logger.info(f"HDF5缓存已初始化: {self.cache_path}")
    
    def _generate_key(self, segment):
        """为段生成唯一标识符"""
        # 创建缓存键
        cache_key = f"{segment['subject_id']}_{os.path.basename(segment['eeg_file'])}_{segment['start_time']}_{segment['end_time']}"
        # 生成哈希，避免文件名过长
        hash_key = hashlib.md5(cache_key.encode()).hexdigest()
        return hash_key
    
    def contains(self, segment, data_type='spectral'):
        """检查数据是否在缓存中"""
        key = self._generate_key(segment)
        group_name = 'spectral_features' if data_type == 'spectral' else 'raw_eeg'
        
        try:
            with h5py.File(self.cache_path, 'r') as f:
                return key in f[group_name]
        except Exception as e:
            self.logger.error(f"检查缓存时出错: {e}")
            return False
    
    def get(self, segment, data_type='spectral'):
        """从缓存获取数据"""
        key = self._generate_key(segment)
        group_name = 'spectral_features' if data_type == 'spectral' else 'raw_eeg'
        
        try:
            with h5py.File(self.cache_path, 'r') as f:
                if key in f[group_name]:
                    data = f[group_name][key][:]
                    self.hits += 1
                    return torch.from_numpy(data).float()
                else:
                    self.misses += 1
                    return None
        except Exception as e:
            self.logger.error(f"读取缓存时出错: {e}")
            self.misses += 1
            return None
    
    def put(self, segment, data, data_type='spectral'):
        """将数据存入缓存"""
        key = self._generate_key(segment)
        group_name = 'spectral_features' if data_type == 'spectral' else 'raw_eeg'
        
        # 确保数据是numpy数组
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        try:
            with h5py.File(self.cache_path, 'a') as f:
                # 如果键已存在且需要覆盖，则删除
                if key in f[group_name] and self.overwrite:
                    del f[group_name][key]
                
                # 如果键不存在或需要覆盖，则创建新数据集
                if key not in f[group_name] or self.overwrite:
                    # 创建数据集，启用压缩
                    f[group_name].create_dataset(
                        key, 
                        data=data, 
                        compression="gzip", 
                        compression_opts=9  # 最高压缩级别
                    )
                    
                    # 添加元数据
                    f[group_name][key].attrs['subject_id'] = segment['subject_id']
                    f[group_name][key].attrs['start_time'] = segment['start_time']
                    f[group_name][key].attrs['end_time'] = segment['end_time']
                    
                    return True
            
            return False
        except Exception as e:
            self.logger.error(f"写入缓存时出错: {e}")
            return False
    
    def batch_put(self, segments, data_list, data_type='spectral'):
        """批量存储多个段的数据"""
        success_count = 0
        try:
            with h5py.File(self.cache_path, 'a') as f:
                group_name = 'spectral_features' if data_type == 'spectral' else 'raw_eeg'
                
                for segment, data in tqdm(zip(segments, data_list), desc=f"批量缓存{data_type}数据", total=len(segments)):
                    key = self._generate_key(segment)
                    
                    # 确保数据是numpy数组
                    if isinstance(data, torch.Tensor):
                        data = data.cpu().numpy()
                    
                    # 如果键已存在且需要覆盖，则删除
                    if key in f[group_name] and self.overwrite:
                        del f[group_name][key]
                    
                    # 如果键不存在或需要覆盖，则创建新数据集
                    if key not in f[group_name] or self.overwrite:
                        # 创建数据集，启用压缩
                        f[group_name].create_dataset(
                            key, 
                            data=data, 
                            compression="gzip", 
                            compression_opts=4  # 平衡压缩率和性能
                        )
                        
                        # 添加元数据
                        f[group_name][key].attrs['subject_id'] = segment['subject_id']
                        f[group_name][key].attrs['start_time'] = segment['start_time']
                        f[group_name][key].attrs['end_time'] = segment['end_time']
                        
                        success_count += 1
            
            return success_count
        except Exception as e:
            self.logger.error(f"批量写入缓存时出错: {e}")
            return success_count
    
    def get_stats(self):
        """获取缓存使用统计"""
        total = self.hits + self.misses
        if total == 0:
            return "无数据加载记录"
        
        hit_rate = self.hits / total * 100 if total > 0 else 0
        
        try:
            with h5py.File(self.cache_path, 'r') as f:
                raw_count = len(f['raw_eeg'])
                spectral_count = len(f['spectral_features'])
                
                # 估算文件大小
                raw_size = sum(f['raw_eeg'][key].nbytes for key in f['raw_eeg'])
                spectral_size = sum(f['spectral_features'][key].nbytes for key in f['spectral_features'])
                
                stats = f"\n===== HDF5缓存统计 =====\n"
                stats += f"总加载请求: {total}\n"
                stats += f"缓存命中: {self.hits} ({hit_rate:.2f}%)\n"
                stats += f"缓存未命中: {self.misses} ({100-hit_rate:.2f}%)\n"
                stats += f"缓存文件: {self.cache_path}\n"
                stats += f"原始EEG缓存数量: {raw_count}\n"
                stats += f"频谱特征缓存数量: {spectral_count}\n"
                stats += f"原始EEG占用内存: {raw_size/1024/1024:.2f} MB\n"
                stats += f"频谱特征占用内存: {spectral_size/1024/1024:.2f} MB\n"
                stats += f"总占用内存: {(raw_size+spectral_size)/1024/1024:.2f} MB\n"
                stats += f"=========================="
                
                return stats
        except Exception as e:
            return f"获取缓存统计时出错: {e}"