import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
import argparse
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('TkAgg')  # 显式设置后端

def load_edf_file(file_path: str, channels: Optional[List[str]] = None) -> Tuple[np.ndarray, dict]:
    """
    Load EDF file and return data and metadata
    
    Args:
        file_path: EDF file path
        channels: List of channels to load, None means load all channels
        
    Returns:
        Tuple of (data array, metadata dictionary)
    """

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    if channels is not None:
        available_channels = [ch for ch in channels if ch in raw.ch_names]
        if not available_channels:
            raise ValueError(f"Specified channels do not exist in file: {file_path}")
        raw.pick_channels(available_channels)

    # Get data and time points
    data, times = raw[:, :]
    
    # Create metadata dictionary
    info = {
        'sfreq': raw.info['sfreq'],
        'ch_names': raw.ch_names,
        'n_channels': len(raw.ch_names),
        'n_times': len(times),
        'duration': len(times) / raw.info['sfreq'],
        'times': times
    }
    
    return data, info

def visualize_eeg(data: np.ndarray, info: dict, title: str = "EEG Data Visualization",
                 start_time: float = 0, duration: float = 10, n_rows: int = None, page: int = 0):
    """
    Visualize EEG data
    
    Args:
        data: EEG data array
        info: Metadata dictionary
        title: Chart title
        start_time: Start time in seconds
        duration: Duration in seconds
        n_rows: Number of channels to display per page (None = all)
        page: Page number (0-based) when using n_rows
    """
    
    s_freq = info['sfreq']
    start_sample = int(start_time * s_freq)
    end_sample = min(int((start_time + duration) * s_freq), data.shape[1])
    times = np.arange(start_sample, end_sample) / s_freq

    # Handle pagination
    n_channels = data.shape[0]
    if n_rows is None:
        n_rows = n_channels
        channel_start = 0
    else:
        n_rows = min(n_rows, n_channels)
        channel_start = page * n_rows
        if channel_start >= n_channels:
            raise ValueError(f"Page {page} exceeds available channels")
    
    channel_end = min(channel_start + n_rows, n_channels)
    channels_to_plot = slice(channel_start, channel_end)

    # Create chart
    fig, axes = plt.subplots(channel_end - channel_start, 1, 
                            figsize=(15, n_rows * 0.5 + 2), 
                            sharex=True)
    
    # Handle single channel case
    if (channel_end - channel_start) == 1:
        axes = [axes]
    
    # Plot each channel
    for i, ax in enumerate(axes):
        ax.plot(times, data[i + channel_start, start_sample:end_sample], linewidth=0.5)
        ax.set_ylabel(info['ch_names'][i + channel_start])
        ax.grid(True)
    
    # Set x-axis label and title
    axes[-1].set_xlabel('Time (seconds)')
    page_info = f" (Channels {channel_start+1}-{channel_end}/{n_channels})" if n_rows < n_channels else ""
    fig.suptitle(f"{title}{page_info}\n{start_time} - {start_time + duration} seconds")
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Return chart for saving
    return fig


def main():

    parser = argparse.ArgumentParser(description="Visualize EEG data")
    parser.add_argument("file_path", type=str, default="/home/lizy/dev/projects/BrainAuth/data/processed/eyes_open/sub-1/sub-1_eeg.edf", help="EDF file path")
    parser.add_argument("--start_time", type=float, default=0, help="Start time (seconds)")
    parser.add_argument("--duration", type=float, default=10, help="Duration (seconds)")
    parser.add_argument("--channels", nargs='+', default=None, help="List of channels to load")
    parser.add_argument("--n_rows", type=int, default=None, help="Number of channels to display per page")
    parser.add_argument("--page", type=int, default=0, help="Page number when using n_rows")
    args = parser.parse_args()

    try: 
        data, info = load_edf_file(args.file_path, args.channels)
        print(f"File loaded: {args.file_path}")
        print(f"Number of channels: {info['n_channels']}")
        print(f"Sampling rate: {info['sfreq']} Hz")
        print(f"Total duration: {info['duration']:.2f} seconds")


    
        # Load EDF file
        data, info = load_edf_file(args.file_path, args.channels)

        # Visualize EEG data
        visualize_eeg(data, info, 
                     start_time=args.start_time, 
                     duration=args.duration,
                     n_rows=args.n_rows,
                     page=args.page)

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


