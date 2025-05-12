#!/usr/bin/env python3
"""
EEG Data Preprocessor

This script preprocesses EEG data from the organized directory and saves it to HDF5 format
for easier training. It implements filtering, resampling, channel selection, normalization,
and windowing with configurable parameters from a YAML config file.

Based on:
- Fan et al. (2021) - CNN-based personal identification system using resting state EEG
- The provided data organization structure
"""

import os
import argparse
import numpy as np
import h5py
import mne
import yaml
from tqdm import tqdm
from scipy import signal
from pathlib import Path
import logging  # <-- 新增

# 配置 logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess EEG data and save to HDF5 format")
    
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--override', nargs='*', default=[],
                        help='Override config values, e.g., preprocessing.normalize=false')
    
    return parser.parse_args()


def load_config(config_path, overrides=None):
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML config file
        overrides: List of key=value pairs to override in config
        
    Returns:
        Configuration dictionary
    """
    # Load config from file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides if provided
    if overrides:
        for override in overrides:
            if '=' in override:
                key_path, value = override.split('=', 1)
                
                # Convert string value to appropriate type
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
                    value = float(value)
                
                # Navigate to the correct nested level
                keys = key_path.split('.')
                curr = config
                for k in keys[:-1]:
                    if k not in curr:
                        curr[k] = {}
                    curr = curr[k]
                
                # Set the value
                curr[keys[-1]] = value
    
    return config


def get_channel_names(channel_set):
    """
    Get channel names based on selected channel set.
    Based on the electrode configurations in Fan et al. paper.
    
    Args:
        channel_set: 'epoc_x' (14 channels) or 'all' (64 channels)
    
    Returns:
        list of channel names or None for all channels
    """
    if channel_set == 'epoc_x':
        # EMOTIV EPOC X 14 Channel configuration
        return ['Af3.', 'F7..', 'F3..', 'Fc5.', 'T7..', 'P7..', 'O1..', 'O2..', 'P8..', 'T8..', 'Fc6.', 'F4..', 'F8..', 'Af4.']
    else:  # 'all' - all 64 channels
        return None  # None means use all available channels


def apply_bandpass_filter(data, sfreq, lowcut, highcut):
    """
    Apply bandpass filter to EEG data.
    
    Args:
        data: EEG data array (channels x samples)
        sfreq: Sampling frequency in Hz
        lowcut: Lower cutoff frequency in Hz
        highcut: Upper cutoff frequency in Hz
    
    Returns:
        Filtered EEG data
    """
    nyquist = 0.5 * sfreq
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design filter
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Apply filter along time dimension for each channel
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i, :] = signal.filtfilt(b, a, data[i, :])
    
    return filtered_data


def resample_data(data, orig_freq, target_freq):
    """
    Resample EEG data to a new sampling frequency.
    
    Args:
        data: EEG data array (channels x samples)
        orig_freq: Original sampling frequency in Hz
        target_freq: Target sampling frequency in Hz
    
    Returns:
        Resampled EEG data
    """
    # Calculate the number of samples in the resampled data
    orig_samples = data.shape[1]
    target_samples = int(orig_samples * (target_freq / orig_freq))
    
    # Resample each channel
    resampled_data = np.zeros((data.shape[0], target_samples))
    for i in range(data.shape[0]):
        resampled_data[i] = signal.resample(data[i], target_samples)
    
    return resampled_data


def apply_z_score_normalization(data):
    """
    Apply z-score normalization to EEG data.
    For each channel, normalize by subtracting mean and dividing by std.
    
    Args:
        data: EEG data array (channels x samples)
    
    Returns:
        Normalized EEG data
    """
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        channel_data = data[i, :]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std > 0:  # Avoid division by zero
            normalized_data[i, :] = (channel_data - mean) / std
        else:
            normalized_data[i, :] = channel_data - mean
    
    return normalized_data


def segment_data(data, sfreq, window_length, window_stride):
    """
    Segment EEG data into windows with possible overlap.
    
    Args:
        data: EEG data array (channels x samples)
        sfreq: Sampling frequency in Hz
        window_length: Window length in seconds
        window_stride: Window stride in seconds
    
    Returns:
        List of segmented windows (each window: channels x samples)
    """
    # Convert from seconds to samples
    window_samples = int(window_length * sfreq)
    stride_samples = int(window_stride * sfreq)
    
    # Calculate number of windows
    n_samples = data.shape[1]
    n_windows = max(0, 1 + (n_samples - window_samples) // stride_samples)
    
    windows = []
    for i in range(n_windows):
        start = i * stride_samples
        end = start + window_samples
        
        if end <= n_samples:  # Ensure we don't go beyond data
            window = data[:, start:end]
            windows.append(window)
    
    return windows


def process_edf_file(file_path, config, orig_sfreq=160):
    """
    Process a single EDF file.
    
    Args:
        file_path: Path to EDF file
        config: Configuration dictionary
        orig_sfreq: Original sampling frequency (default 160 Hz for PhysioNet dataset)
    
    Returns:
        List of processed data windows and effective sampling frequency
    """
    # Read EDF file using MNE
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return [], orig_sfreq

    # Select channels if needed
    if config['preprocessing']['channels']['select']:
        channel_set = config['preprocessing']['channels']['set']
        channels = get_channel_names(channel_set)
        if channels:
            available_channels = raw.ch_names
            # Find intersection of requested and available channels
            use_channels = [ch for ch in channels if ch in available_channels]
            if len(use_channels) < len(channels):
                logger.warning(f"Only {len(use_channels)}/{len(channels)} channels available in {file_path}")
            if not use_channels:
                logger.error(f"No requested channels found in {file_path}")
                return [], orig_sfreq
            raw.pick(use_channels)
    
    # Get data as numpy array (channels x samples)
    data = raw.get_data()
    
    # Get effective sampling frequency
    sfreq = orig_sfreq
    
    # Apply bandpass filter if enabled
    if config['preprocessing']['filter']['apply']:
        lowcut = config['preprocessing']['filter']['lowcut']
        highcut = config['preprocessing']['filter']['highcut']
        data = apply_bandpass_filter(data, sfreq, lowcut, highcut)
    
    # Apply resampling if enabled
    if config['preprocessing']['resample']['apply']:
        target_freq = config['preprocessing']['resample']['freq']
        data = resample_data(data, sfreq, target_freq)
        sfreq = target_freq  # Update sampling frequency
    
    # Apply z-score normalization if enabled
    if config['preprocessing']['normalize']:
        data = apply_z_score_normalization(data)
    
    # Segment data into windows
    window_length = config['windowing']['window_length']
    window_stride = config['windowing']['window_stride']
    windows = segment_data(data, sfreq, window_length, window_stride)
    
    return windows, sfreq


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, args.override)
    
    # Create output directory if it doesn't exist
    output_dir = Path(config['data']['processed_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / config['data']['output_filename']
    
    # Conditions to process
    conditions = []
    if config['experiment']['include_eyes_open']:
        conditions.append('eyes_open')
    if config['experiment']['include_eyes_closed']:
        conditions.append('eyes_closed')
    
    # Original sampling frequency for PhysioNet dataset
    orig_sfreq = 160
    
    # Get effective sampling frequency after potential resampling
    effective_sfreq = orig_sfreq
    if config['preprocessing']['resample']['apply']:
        effective_sfreq = config['preprocessing']['resample']['freq']
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as hf:
        # Store metadata
        metadata = hf.create_group('metadata')
        metadata.attrs['window_length'] = config['windowing']['window_length']
        metadata.attrs['window_stride'] = config['windowing']['window_stride']
        metadata.attrs['original_sampling_frequency'] = orig_sfreq
        metadata.attrs['effective_sampling_frequency'] = effective_sfreq
        
        # Store filtering info
        metadata.attrs['filter_applied'] = config['preprocessing']['filter']['apply']
        if config['preprocessing']['filter']['apply']:
            metadata.attrs['lowcut'] = config['preprocessing']['filter']['lowcut']
            metadata.attrs['highcut'] = config['preprocessing']['filter']['highcut']
        
        # Store resampling info
        metadata.attrs['resampling_applied'] = config['preprocessing']['resample']['apply']
        if config['preprocessing']['resample']['apply']:
            metadata.attrs['target_frequency'] = config['preprocessing']['resample']['freq']
        
        # Store normalization info
        metadata.attrs['normalization_applied'] = config['preprocessing']['normalize']
        
        # Store channel selection info
        metadata.attrs['channel_selection'] = config['preprocessing']['channels']['select']
        if config['preprocessing']['channels']['select']:
            metadata.attrs['channel_set'] = config['preprocessing']['channels']['set']
            if config['preprocessing']['channels']['set'] == 'epoc_x':
                metadata.attrs['num_channels'] = 14
            else:
                metadata.attrs['num_channels'] = 64
        
        # Store full config as string
        metadata.attrs['config_yaml'] = yaml.dump(config)
        
        # Process each condition and subject
        data_group = hf.create_group('data')
        
        for condition in conditions:
            condition_path = Path(config['data']['raw_dir']) / condition
            if not condition_path.exists():
                logger.warning(f"Condition directory {condition_path} does not exist")
                continue
            
            # Create group for this condition
            condition_group = data_group.create_group(condition)
            
            # Get subject directories
            subject_dirs = [d for d in condition_path.iterdir() if d.is_dir()]
            
            # Process each subject
            for subject_dir in tqdm(subject_dirs, desc=f"Processing {condition}"):
                subject_id = subject_dir.name
                edf_file = subject_dir / f"{subject_id}_eeg.edf"
                
                if not edf_file.exists():
                    logger.warning(f"EDF file {edf_file} does not exist")
                    continue
                
                # Process EDF file
                windows, sfreq = process_edf_file(str(edf_file), config, orig_sfreq)
                
                if not windows:
                    logger.warning(f"No data windows extracted from {edf_file}")
                    continue
                
                # Create dataset for this subject
                # Convert windows list to a single numpy array (windows x channels x samples)
                windows_array = np.array(windows)
                subject_group = condition_group.create_group(subject_id)
                subject_group.create_dataset('windows', data=windows_array)
                subject_group.attrs['num_windows'] = len(windows)
                subject_group.attrs['sampling_frequency'] = sfreq
    
    logger.info(f"Processing complete. Output saved to {output_path}")


if __name__ == "__main__":
    main()