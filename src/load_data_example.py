
"""
Example script showing how to load and use the preprocessed EEG data.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Load and visualize preprocessed EEG data")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Path to the preprocessed HDF5 data file (overrides config)')
    parser.add_argument('--subject', type=str, default='sub-1',
                        help='Subject ID to visualize')
    parser.add_argument('--condition', type=str, default='eyes_open',
                        choices=['eyes_open', 'eyes_closed'],
                        help='Condition to visualize')
    parser.add_argument('--window', type=int, default=0,
                        help='Window index to visualize')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_metadata(h5file):
    """Print metadata from the HDF5 file."""
    metadata = h5file['metadata']
    print("\n=== Metadata ===")
    for key, value in metadata.attrs.items():
        if key != 'config_yaml':  # Skip the full config dump to keep output clean
            print(f"{key}: {value}")


def list_available_subjects(h5file, condition):
    """List available subjects for a given condition."""
    try:
        subjects = list(h5file['data'][condition].keys())
        return subjects
    except KeyError:
        return []


def load_subject_data(h5file, condition, subject):
    """Load all windows for a specific subject and condition."""
    try:
        return h5file['data'][condition][subject]['windows'][:]
    except KeyError:
        return None


def visualize_window(data, window_idx=0, sampling_rate=160):
    """
    Visualize a single EEG data window.
    
    Args:
        data: Windows data array (windows x channels x samples)
        window_idx: Index of the window to visualize
        sampling_rate: Sampling rate in Hz
    """
    if window_idx >= data.shape[0]:
        print(f"Error: Window index {window_idx} out of range (0-{data.shape[0]-1})")
        return
    
    window = data[window_idx]
    num_channels, num_samples = window.shape
    
    # Create time axis in seconds
    time = np.arange(num_samples) / sampling_rate
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Calculate appropriate spacing between channels
    spacing = max([np.abs(window).max() * 2, 1])
    
    # Plot each channel
    for i in range(num_channels):
        plt.plot(time, window[i] + i * spacing, label=f'CH {i+1}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Channel')
    plt.title(f'EEG Data Window {window_idx} - {num_channels} channels @ {sampling_rate} Hz')
    plt.yticks(np.arange(0, num_channels * spacing, spacing),
               [f'CH {i+1}' for i in range(num_channels)])
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine data file path
    if args.data_file:
        data_path = Path(args.data_file)
    else:
        data_path = Path(config['data']['processed_dir']) / config['data']['output_filename']
    
    if not data_path.exists():
        print(f"Error: Data file {data_path} does not exist")
        return
    
    with h5py.File(data_path, 'r') as hf:
        # Print metadata
        print_metadata(hf)
        
        # List available subjects
        subjects = list_available_subjects(hf, args.condition)
        if not subjects:
            print(f"No subjects found for condition: {args.condition}")
            return
        
        print(f"\n=== Available subjects for {args.condition} ===")
        print(subjects)
        
        # Check if requested subject exists
        if args.subject not in subjects:
            print(f"Subject {args.subject} not found. Available subjects: {subjects}")
            return
        
        # Load data for the subject
        windows_data = load_subject_data(hf, args.condition, args.subject)
        if windows_data is None:
            print(f"Error loading data for subject {args.subject}")
            return
        
        print(f"\n=== Data for {args.subject} ===")
        print(f"Shape: {windows_data.shape} (windows x channels x samples)")
        
        # Get sampling rate from subject metadata if available, otherwise use global metadata
        try:
            sampling_rate = hf['data'][args.condition][args.subject].attrs.get('sampling_frequency')
        except:
            sampling_rate = hf['metadata'].attrs.get('effective_sampling_frequency', 160)
        
        # Visualize a window
        visualize_window(windows_data, args.window, sampling_rate)


if __name__ == "__main__":
    main()