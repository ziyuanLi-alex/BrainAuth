#!/usr/bin/env python3
"""
EEG DataLoader

Flexible DataLoader for EEG data supporting two modes:
1. Identity mode: (eeg, label) pairs for standard classification (original paper)
2. Siamese mode: (eeg1, eeg2, diff_label) triplets for similarity learning

Settings are loaded from the YAML configuration file.
"""

import os
import numpy as np
import h5py
import random
import yaml
import torch
import hashlib
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class EEGDataset(Dataset):
    """
    Dataset for EEG data that supports both identity classification and siamese modes.
    
    Args:
        data_file: Path to HDF5 data file
        config: Configuration dictionary
        train: If True, use training set; if False, use test set
        conditions: List of conditions to include ('eyes_open', 'eyes_closed', or both)
    """
    def __init__(
        self, 
        data_file, 
        config,
        train=True, 
        conditions=None
    ):
        self.data_file = data_file
        self.config = config
        self.train = train
        
        # Extract relevant configuration
        self.mode = config['dataloader']['mode']
        self.test_size = config['dataloader']['test_size']
        self.seed = config['dataloader']['seed']
        
        # Siamese-specific settings
        if self.mode == 'siamese':
            self.pos_ratio = config['dataloader']['siamese']['pos_ratio']
            self.same_session = config['dataloader']['siamese']['same_session']
        
        # Set conditions
        if conditions is None:
            if config['experiment']['include_eyes_open'] and config['experiment']['include_eyes_closed']:
                self.conditions = ['eyes_open', 'eyes_closed']
            elif config['experiment']['include_eyes_open']:
                self.conditions = ['eyes_open']
            elif config['experiment']['include_eyes_closed']:
                self.conditions = ['eyes_closed']
            else:
                raise ValueError("No conditions specified in config")
        else:
            self.conditions = conditions
            
        # Set random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Load data and prepare indices
        self._load_data()
        
    def _load_data(self):
        """Load data from HDF5 file and prepare indices."""
        with h5py.File(self.data_file, 'r') as hf:
            # Check if conditions exist in the data file
            available_conditions = list(hf['data'].keys())
            self.conditions = [c for c in self.conditions if c in available_conditions]
            
            if not self.conditions:
                raise ValueError(f"None of the specified conditions exist in {self.data_file}")
            
            # Store metadata
            self.sampling_rate = hf['metadata'].attrs.get('effective_sampling_frequency', 160)
            self.window_length = hf['metadata'].attrs.get('window_length', 0.5)
            
            # Load subject IDs for each condition
            self.subjects = {}
            for condition in self.conditions:
                self.subjects[condition] = list(hf['data'][condition].keys())
            
            # Load all windows and labels
            self.all_windows = []
            self.all_labels = []
            self.subject_to_idx = {}  # Map subject ID to class index
            self.condition_indices = []  # Keep track of condition for each window
            
            class_idx = 0
            window_idx = 0
            
            # First pass: create subject mapping
            for subject in set().union(*[self.subjects[c] for c in self.conditions]):
                self.subject_to_idx[subject] = class_idx
                class_idx += 1
            
            # Second pass: load all windows
            for condition_idx, condition in enumerate(self.conditions):
                for subject in self.subjects[condition]:
                    # Get subject index (label)
                    subject_idx = self.subject_to_idx[subject]
                    
                    # Load windows for this subject
                    windows = hf['data'][condition][subject]['windows'][:]
                    num_windows = windows.shape[0]
                    
                    # Add windows and labels
                    self.all_windows.append(windows)
                    self.all_labels.extend([subject_idx] * num_windows)
                    
                    # Keep track of condition and index range
                    self.condition_indices.extend([(condition_idx, window_idx + i) for i in range(num_windows)])
                    window_idx += num_windows
        
        # Combine all windows and convert to numpy array
        self.all_windows = np.vstack([windows for windows in self.all_windows])
        self.all_labels = np.array(self.all_labels)
        self.condition_indices = np.array(self.condition_indices)
        
        # Split data into train and test sets
        self.num_classes = len(self.subject_to_idx)
        self._split_train_test()
        
        # For siamese mode, prepare pairs
        if self.mode == 'siamese':
            self._prepare_pairs()
            
    def _split_train_test(self):
        """Split data into training and testing sets."""
        indices = np.arange(len(self.all_labels))
        
        # Stratified split to maintain class distribution
        train_indices, test_indices = train_test_split(
            indices, 
            test_size=self.test_size, 
            random_state=self.seed,
            stratify=self.all_labels
        )
        
        if self.train:
            self.indices = train_indices
        else:
            self.indices = test_indices
            
    def _prepare_pairs(self):
        """Prepare positive and negative pairs for siamese mode."""
        # Check if cache is enabled
        cache_enabled = self.config['dataloader']['siamese'].get('cache_pairs', False)
        
        if cache_enabled:
            # Try to load from cache first
            pairs, labels = self._load_pairs_from_cache()
            if pairs is not None and labels is not None:
                self.pairs = pairs
                self.pair_labels = labels
                print(f"Loaded {len(self.pairs)} pairs from cache")
                return
        
        self.pairs = []
        self.pair_labels = []
        
        # Get valid indices for current split
        valid_indices = self.indices
        valid_labels = self.all_labels[valid_indices]
        valid_conditions = self.condition_indices[valid_indices]
        
        # Calculate number of pairs
        num_samples = len(valid_indices)
        num_pairs = num_samples * 2  # Generate twice as many pairs as samples
        
        # Calculate number of positive pairs based on pos_ratio
        num_pos_pairs = int(num_pairs * self.pos_ratio)
        num_neg_pairs = num_pairs - num_pos_pairs
        
        # Generate positive pairs (same subject)
        pos_pairs_generated = 0
        pos_attempts = 0
        max_attempts = num_pos_pairs * 10  # Limit attempts to avoid infinite loop

        with tqdm(total=num_pos_pairs, desc="Generating positive pairs") as pbar_pos:
            while pos_pairs_generated < num_pos_pairs and pos_attempts < max_attempts:
                pos_attempts += 1
                
                # Randomly select a subject with at least 2 samples
                subj_indices = random.choice(np.unique(valid_labels))
                indices_for_subj = np.where(valid_labels == subj_indices)[0]
                
                if len(indices_for_subj) < 2:
                    continue
                
                # Randomly select two different indices for the same subject
                idx1, idx2 = np.random.choice(indices_for_subj, 2, replace=False)
                
                # Get the actual indices in the original dataset
                idx1_orig = valid_indices[idx1]
                idx2_orig = valid_indices[idx2]
                
                # If same_session is True, check if they're from the same condition
                if self.same_session:
                    cond1 = valid_conditions[idx1][0]
                    cond2 = valid_conditions[idx2][0]
                    if cond1 != cond2:
                        # Different conditions, skip
                        continue
                
                self.pairs.append((idx1_orig, idx2_orig))
                self.pair_labels.append(1)  # 1 for same subject
                
                pos_pairs_generated += 1
                pbar_pos.update(1)
            
            if pos_pairs_generated < num_pos_pairs:
                print(f"Warning: Could only generate {pos_pairs_generated}/{num_pos_pairs} positive pairs")
        
        # Generate negative pairs (different subjects)
        neg_pairs_generated = 0
        neg_attempts = 0
        max_attempts = num_neg_pairs * 10  # Limit attempts to avoid infinite loop

        with tqdm(total=num_neg_pairs, desc="Generating negative pairs") as pbar_neg:
            while neg_pairs_generated < num_neg_pairs and neg_attempts < max_attempts:
                neg_attempts += 1
                
                # Randomly select two different subjects
                unique_subjects = np.unique(valid_labels)
                if len(unique_subjects) < 2:
                    break
                    
                subj1, subj2 = np.random.choice(unique_subjects, 2, replace=False)
                
                # Find indices for each subject
                indices_for_subj1 = np.where(valid_labels == subj1)[0]
                indices_for_subj2 = np.where(valid_labels == subj2)[0]
                
                # Randomly select an index for each subject
                idx1 = np.random.choice(indices_for_subj1)
                idx2 = np.random.choice(indices_for_subj2)
                
                # Get the actual indices in the original dataset
                idx1_orig = valid_indices[idx1]
                idx2_orig = valid_indices[idx2]
                
                # If same_session is True, check if they're from the same condition
                if self.same_session:
                    cond1 = valid_conditions[idx1][0]
                    cond2 = valid_conditions[idx2][0]
                    if cond1 != cond2:
                        # Different conditions, skip
                        continue
                
                self.pairs.append((idx1_orig, idx2_orig))
                self.pair_labels.append(0)  # 0 for different subjects
                neg_pairs_generated += 1
                pbar_neg.update(1)
            
            if neg_pairs_generated < num_neg_pairs:
                print(f"Warning: Could only generate {neg_pairs_generated}/{num_neg_pairs} negative pairs")
        
        # Convert to numpy arrays
        self.pairs = np.array(self.pairs)
        self.pair_labels = np.array(self.pair_labels)
        
        # Cache the pairs if enabled
        if cache_enabled:
            self._save_pairs_to_cache()
    
    def _generate_cache_path(self):
        """Generate a unique cache path based on dataset parameters."""
        # Create a string that represents the key parameters that affect pair generation
        config_str = (
            f"data_file={os.path.basename(self.data_file)}_"
            f"mode={self.mode}_"
            f"train={self.train}_"
            f"test_size={self.test_size}_"
            f"seed={self.seed}_"
            f"conditions={'_'.join(self.conditions)}_"
            f"pos_ratio={self.pos_ratio}_"
            f"same_session={self.same_session}"
        )
        
        # Generate a hash of the config string
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        # Create cache directory if it doesn't exist
        cache_dir = Path(self.config['dataloader']['siamese'].get('pairs_cache_dir', 'data/cache'))
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Return the full cache path
        return cache_dir / f"pairs_cache_{config_hash}.h5"
    
    def _save_pairs_to_cache(self):
        """Save generated pairs to cache file."""
        cache_path = self._generate_cache_path()
        
        try:
            with h5py.File(cache_path, 'w') as f:
                f.create_dataset('pairs', data=self.pairs)
                f.create_dataset('labels', data=self.pair_labels)
                
                # Store metadata for validation
                metadata = f.create_group('metadata')
                metadata.attrs['data_file'] = os.path.basename(self.data_file)
                metadata.attrs['mode'] = self.mode
                metadata.attrs['train'] = self.train
                metadata.attrs['test_size'] = self.test_size
                metadata.attrs['seed'] = self.seed
                metadata.attrs['conditions'] = ','.join(self.conditions)
                metadata.attrs['pos_ratio'] = self.pos_ratio
                metadata.attrs['same_session'] = self.same_session
                metadata.attrs['num_pairs'] = len(self.pairs)
            
            print(f"Saved {len(self.pairs)} pairs to cache: {cache_path}")
            return True
        except Exception as e:
            print(f"Error saving pairs to cache: {e}")
            return False
    
    def _load_pairs_from_cache(self):
        """Load pairs from cache file if it exists and is valid."""
        cache_path = self._generate_cache_path()
        
        if not cache_path.exists():
            return None, None
        
        try:
            with h5py.File(cache_path, 'r') as f:
                # Validate metadata
                metadata = f['metadata']
                
                if (
                    metadata.attrs['data_file'] != os.path.basename(self.data_file) or
                    metadata.attrs['mode'] != self.mode or
                    metadata.attrs['train'] != self.train or
                    metadata.attrs['test_size'] != self.test_size or
                    metadata.attrs['seed'] != self.seed or
                    metadata.attrs['conditions'] != ','.join(self.conditions) or
                    metadata.attrs['pos_ratio'] != self.pos_ratio or
                    metadata.attrs['same_session'] != self.same_session
                ):
                    print("Cache metadata mismatch, regenerating pairs")
                    return None, None
                
                # Load pairs and labels
                pairs = f['pairs'][:]
                labels = f['labels'][:]
                
                return pairs, labels
        except Exception as e:
            print(f"Error loading pairs from cache: {e}")
            return None, None
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        if self.mode == 'identity':
            return len(self.indices)
        elif self.mode == 'siamese':
            return len(self.pairs)
            
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        if self.mode == 'identity':
            # Get the window and label for the index
            index = self.indices[idx]
            window = self.all_windows[index]
            label = self.all_labels[index]
            
            # Convert to torch tensor
            window = torch.tensor(window, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.long)
            
            return window, label
            
        elif self.mode == 'siamese':
            # Get the pair and label for the index
            pair = self.pairs[idx]
            label = self.pair_labels[idx]
            
            # Get the windows for the pair
            window1 = self.all_windows[pair[0]]
            window2 = self.all_windows[pair[1]]
            
            # Convert to torch tensor
            window1 = torch.tensor(window1, dtype=torch.float32)
            window2 = torch.tensor(window2, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)
            
            return window1, window2, label


def get_dataloader(config_path, data_file=None, train=True, conditions=None):
    """
    Create a DataLoader for EEG data based on configuration.
    
    Args:
        config_path: Path to configuration YAML file
        data_file: Path to HDF5 data file (overrides config)
        train: If True, use training set; if False, use test set
        conditions: List of conditions to include (overrides config)
        
    Returns:
        DataLoader object
    """
    # Load configuration
    config = load_config(config_path)
    
    # Determine data file path
    if data_file is None:
        data_file = os.path.join(
            config['data']['processed_dir'], 
            config['data']['output_filename']
        )
    
    # Create dataset
    dataset = EEGDataset(
        data_file=data_file,
        config=config,
        train=train,
        conditions=conditions
    )
    
    # Create dataloader
    return DataLoader(
        dataset=dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=train,  # Shuffle only if training
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )


def get_dataloaders(config_path, data_file=None, conditions=None):
    """
    Create both training and testing DataLoaders.
    
    Args:
        config_path: Path to configuration YAML file
        data_file: Path to HDF5 data file (overrides config)
        conditions: List of conditions to include (overrides config)
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_loader = get_dataloader(
        config_path=config_path,
        data_file=data_file,
        train=True,
        conditions=conditions
    )
    
    test_loader = get_dataloader(
        config_path=config_path,
        data_file=data_file,
        train=False,
        conditions=conditions
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    """Example usage of the DataLoader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test EEG DataLoader")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Path to the preprocessed HDF5 data file (overrides config)')
    parser.add_argument('--mode', type=str, default=None, choices=['identity', 'siamese'],
                        help='Override DataLoader mode in config: identity or siamese')
    parser.add_argument('--clear_cache', action='store_true',
                        help='Clear pairs cache before running')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override mode if specified
    if args.mode:
        config['dataloader']['mode'] = args.mode
    
    # Clear cache if requested
    if args.clear_cache:
        cache_dir = Path(config['dataloader']['siamese'].get('pairs_cache_dir', 'data/cache'))
        if cache_dir.exists():
            for cache_file in cache_dir.glob('pairs_cache_*.h5'):
                print(f"Removing cache file: {cache_file}")
                cache_file.unlink()
    
    # Create DataLoader
    train_loader, test_loader = get_dataloaders(
        config_path=args.config,
        data_file=args.data_file
    )
    
    # Print dataset info
    dataset = train_loader.dataset
    print(f"\n=== DataLoader Info ===")
    print(f"Mode: {dataset.mode}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of samples (train): {len(train_loader.dataset)}")
    print(f"Number of samples (test): {len(test_loader.dataset)}")
    print(f"Sampling rate: {dataset.sampling_rate} Hz")
    print(f"Window length: {dataset.window_length} seconds")
    
    # Print cache info if in siamese mode
    if dataset.mode == 'siamese':
        cache_enabled = config['dataloader']['siamese'].get('cache_pairs', False)
        cache_dir = config['dataloader']['siamese'].get('pairs_cache_dir', 'data/cache')
        print(f"Pairs caching: {'Enabled' if cache_enabled else 'Disabled'}")
        if cache_enabled:
            print(f"Cache directory: {cache_dir}")
    
    # Get a batch of data
    for batch in train_loader:
        if dataset.mode == 'identity':
            windows, labels = batch
            print(f"\n=== Batch Info (Identity Mode) ===")
            print(f"Windows shape: {windows.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Unique labels in batch: {torch.unique(labels).numpy()}")
        else:  # siamese
            windows1, windows2, labels = batch
            print(f"\n=== Batch Info (Siamese Mode) ===")
            print(f"Windows1 shape: {windows1.shape}")
            print(f"Windows2 shape: {windows2.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Positive pairs: {torch.sum(labels).item()}")
            print(f"Negative pairs: {len(labels) - torch.sum(labels).item()}")
            
        # Just show the first batch
        break