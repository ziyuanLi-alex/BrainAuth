import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path

def explore_eeg_file(file_path: str) -> Dict:
    """探索EEG文件，获取基本信息
    
    参数:
        file_path: EEG文件路径
        
    返回:
        包含文件信息的字典
    """
    raw = mne.io.read_raw_edf(file_path, preload=False)
    
    info = {
        'channels': raw.ch_names,
        'n_channels': len(raw.ch_names),
        'sampling_freq': raw.info['sfreq'],
        'duration': raw.times[-1],
        'n_samples': len(raw.times),
    }
    
    return info


def plot_eeg_channels(file_path: str, 
                      start_time: float = 0, 
                      duration: float = 5,
                      channels: Optional[List[str]] = None,
                      filter_params: Optional[Dict] = None,
                      save_path: Optional[str] = None):
    """绘制EEG通道数据
    
    参数:
        file_path: EEG文件路径
        start_time: 起始时间（秒）
        duration: 持续时间（秒）
        channels: 要绘制的通道列表，None表示全部通道
        filter_params: 滤波参数，如 {'l_freq': 1.0, 'h_freq': 40.0}
        save_path: 保存路径，如果不为None则保存图片
    """
    raw = mne.io.read_raw_edf(file_path, preload=True)
    
    # 应用滤波器（如果指定）
    if filter_params:
        raw.filter(
            l_freq=filter_params.get('l_freq'),
            h_freq=filter_params.get('h_freq')
        )

    # 选择通道（如果指定）
    if channels:
        available_channels = raw.ch_names
        channels_to_plot = [ch for ch in channels if ch in available_channels]
        if not channels_to_plot:
            raise ValueError("指定的通道均不可用")
        raw.pick_channels(channels_to_plot)
    
    # 计算结束时间
    end_time = start_time + duration
    
    # 绘制数据
    plt.figure(figsize=(15, 10))
    raw.plot(start=start_time, duration=duration, scalings='auto', title='EEG Data')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

if __name__ == "__main__":
    # Get project root directory (assuming preprocess.py is in the src directory)
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()

    # Define relative paths
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    EYES_OPEN_DIR = PROCESSED_DIR / "eyes_open"
    EYES_CLOSED_DIR = PROCESSED_DIR / "eyes_closed"


    # 示例用法
    file_path = str(EYES_OPEN_DIR / "sub-1/sub-1_eeg.edf")
    info = explore_eeg_file(file_path)
    print(info)
    
    # plot_eeg_channels(file_path, start_time=0, duration=10)