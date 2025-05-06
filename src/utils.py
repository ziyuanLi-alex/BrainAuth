import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import random
import torch
import yaml
from mne.viz import plot_topomap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)

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


def set_seed(seed):
    """设置随机种子以确保实验可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_device():
    """获取设备配置"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device


def create_dirs(dirs):
    """创建目录(如果不存在)"""
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"创建目录: {dir_path}")


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    """保存模型检查点"""
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_path)
        print(f"保存最佳模型到 {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """加载模型检查点"""
    if not os.path.exists(checkpoint_path):
        print(f"检查点不存在: {checkpoint_path}")
        return 0, 0
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"从epoch {epoch}加载检查点, 损失值: {loss:.4f}")
    
    return epoch, loss

def calculate_metrics(y_true, y_pred, y_score=None):
    """计算模型性能评估指标"""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary')
    metrics['recall'] = recall_score(y_true, y_pred, average='binary')
    metrics['f1'] = f1_score(y_true, y_pred, average='binary')
    
    if y_score is not None:
        # ROC曲线和AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        metrics['auc'] = auc(fpr, tpr)
        
        # EER (Equal Error Rate)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
        EER = fpr[np.nanargmin(np.absolute(fnr - fpr))]
        metrics['eer'] = EER
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    return metrics


def plot_metrics(metrics_history, save_path=None):
    """绘制指标历史图"""
    plt.figure(figsize=(15, 10))
    
    # 训练和验证损失
    plt.subplot(2, 3, 1)
    plt.plot(metrics_history['train_loss'], label='训练损失')
    plt.plot(metrics_history['val_loss'], label='验证损失')
    plt.title('损失值')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率
    plt.subplot(2, 3, 2)
    plt.plot(metrics_history['train_accuracy'], label='训练准确率')
    plt.plot(metrics_history['val_accuracy'], label='验证准确率')
    plt.title('准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # F1分数
    plt.subplot(2, 3, 3)
    plt.plot(metrics_history['train_f1'], label='训练F1')
    plt.plot(metrics_history['val_f1'], label='验证F1')
    plt.title('F1分数')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # 精确率和召回率
    plt.subplot(2, 3, 4)
    plt.plot(metrics_history['train_precision'], label='训练精确率')
    plt.plot(metrics_history['val_precision'], label='验证精确率')
    plt.plot(metrics_history['train_recall'], label='训练召回率')
    plt.plot(metrics_history['val_recall'], label='验证召回率')
    plt.title('精确率和召回率')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    # AUC和EER (如果有)
    if 'val_auc' in metrics_history:
        plt.subplot(2, 3, 5)
        plt.plot(metrics_history['val_auc'], label='验证AUC')
        plt.title('AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
    
    if 'val_eer' in metrics_history:
        plt.subplot(2, 3, 6)
        plt.plot(metrics_history['val_eer'], label='验证EER')
        plt.title('等错误率(EER)')
        plt.xlabel('Epoch')
        plt.ylabel('EER')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"指标图保存到 {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm, classes=['同一人', '不同人'], save_path=None):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    if save_path:
        plt.savefig(save_path)
        print(f"混淆矩阵图保存到 {save_path}")
    
    plt.show()

def plot_roc_curve(y_true, y_score, save_path=None):
    """绘制ROC曲线"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率')
    plt.ylabel('真正例率')
    plt.title('接收者操作特征曲线')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        print(f"ROC曲线图保存到 {save_path}")
    
    plt.show()


def compute_eeg_frequency_features(raw_eeg, freq_bands=[(10, 15)], window_duration=2.0, overlap=0.5):
    """
    计算EEG的频带功率谱特征
    
    参数:
        raw_eeg: MNE Raw对象
        freq_bands: 待分析的频带列表，例如[(8, 12), (13, 30)]
        window_duration: 窗口持续时间(秒)
        overlap: 窗口重叠比例
        
    返回:
        channel_names: 通道名称列表
        frequencies: 频率点列表
        psd_data: 功率谱密度数据，形状为(n_channels, n_freqs)
    """
    # 计算窗口参数
    sfreq = raw_eeg.info['sfreq']
    n_per_seg = int(window_duration * sfreq)
    n_overlap = int(n_per_seg * overlap)
    
    # 计算功率谱密度
    psd, freqs = mne.time_frequency.psd_welch(
        raw_eeg,
        n_fft=n_per_seg,
        n_overlap=n_overlap,
        picks='all',
        fmin=freq_bands[0][0],
        fmax=freq_bands[-1][1],
        verbose=False
    )
    
    # 获取通道名称
    channel_names = raw_eeg.ch_names
    
    return channel_names, freqs, psd

def create_topo_map_from_values(raw_eeg, channel_values, title=None, show=True):
    """
    根据通道值创建脑电地形图
    
    参数:
        raw_eeg: MNE Raw对象(用于获取通道位置)
        channel_values: 每个通道的值
        title: 图像标题
        show: 是否显示图像
        
    返回:
        fig: 图像对象
    """
    # 获取通道位置
    pos = mne.channels.layout._find_topomap_coords(raw_eeg.info, picks='all')
    
    # 绘制地形图
    fig, ax = plt.subplots(figsize=(6, 6))
    im, _ = plot_topomap(channel_values, pos, axes=ax, show=False)
    
    if title:
        plt.title(title)
    
    plt.colorbar(im, ax=ax)
    
    if show:
        plt.show()
    
    return fig

def create_3d_spectral_image(raw_eeg, freq_range=(10, 15), freq_step=0.5, resolution=(110, 100)):
    """
    创建3D空-频特征图
    
    参数:
        raw_eeg: MNE Raw对象
        freq_range: 频率范围 (最小值, 最大值)
        freq_step: 频率步长
        resolution: 输出图像分辨率 (高度, 宽度)
        
    返回:
        spectral_image: 3D空-频特征图，形状为(n_freqs, height, width)
    """
    # 计算频带数量
    fmin, fmax = freq_range
    n_bands = int((fmax - fmin) / freq_step)
    
    # 创建输出图像
    height, width = resolution
    spectral_image = np.zeros((n_bands, height, width))
    
    # 获取通道位置
    pos = mne.channels.layout._find_topomap_coords(raw_eeg.info, picks='all')
    
    # 计算整个频率范围的PSD
    _, freqs, psd = compute_eeg_frequency_features(
        raw_eeg, 
        freq_bands=[(fmin, fmax)], 
        window_duration=2.0, 
        overlap=0.5
    )
    
    # 对每个子频带创建地形图
    for i in range(n_bands):
        current_fmin = fmin + i * freq_step
        current_fmax = current_fmin + freq_step
        
        # 找到当前频带范围内的频率索引
        freq_indices = np.where((freqs >= current_fmin) & (freqs < current_fmax))[0]
        
        if len(freq_indices) == 0:
            continue
        
        # 计算当前频带的平均功率
        band_psd = np.mean(psd[:, freq_indices], axis=1)
        
        # 创建密集网格进行插值
        x_grid = np.linspace(-0.5, 0.5, width)
        y_grid = np.linspace(-0.5, 0.5, height)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        
        # 使用MNE的插值函数创建地形图
        image_data = mne.viz.topomap._grid_data(
            pos, 
            band_psd, 
            grid_x, 
            grid_y, 
            'cubic'
        )
        
        # 存储到3D图像中
        spectral_image[i] = image_data
    
    return spectral_image




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


