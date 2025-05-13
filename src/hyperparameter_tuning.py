#!/usr/bin/env python3
"""
Hyperparameter Tuning for EEG-based Personal Identification Models

This script uses Optuna to automatically search for optimal hyperparameters
for the SiameseICAConvNet model. It creates custom configurations for each trial
and implements its own training loop to avoid interfering with the main training.

Usage:
    python src/hyperparameter_tuning.py
"""

import os
import yaml
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from pathlib import Path
import logging
from datetime import datetime
import sys
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

# Import from existing modules
from dataset import get_dataloaders, load_config
from model import ICAConvNet, SiameseICAConvNet, ContrastiveSiameseNet, contrastive_loss


def setup_logging(output_dir):
    """
    Set up logging configuration.
    
    Args:
        output_dir: Directory to save log file
    
    Returns:
        Logger object
    """
    # Create logger
    logger = logging.getLogger('hyperparameter_tuning')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    log_file = output_dir / 'tuning.log'
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def validate_dimensions(trial_config):
    """
    Validate that the dimensions in the configuration are compatible.
    This is critical to avoid shape mismatch errors in the model.
    
    Args:
        trial_config: Configuration dictionary
        
    Returns:
        Boolean indicating whether dimensions are valid
    """
    # Get key dimensions
    ica_components = trial_config['model']['ica_components']
    fc_dim = trial_config['model']['fc_dim']
    
    # The default ICAConvNet architecture from the original code has these constraints:
    # 1. ICA components must be divisible by 2 (for the pooling operations)
    # 2. Embedding dimensions must be consistent
    
    # Ensure ICA components is divisible by 2
    if ica_components % 2 != 0:
        return False
    
    # For siamese models, ensure embedding dimensions match
    if trial_config['dataloader']['mode'] == 'siamese':
        embedding_dim = trial_config['model']['siamese']['embedding_dim']
        if embedding_dim != fc_dim:
            # Update to make consistent
            trial_config['model']['siamese']['embedding_dim'] = fc_dim
    
    return True


def create_trial_config(config, trial):
    """
    Create a configuration for a trial by updating the base config with 
    hyperparameters suggested by Optuna.
    """
    # Create a deep copy to avoid modifying the original
    trial_config = copy.deepcopy(config)
    
    # Model hyperparameters
    trial_config['model']['ica_components'] = trial.suggest_categorical('ica_components', [32, 64, 128])
    num_filters = trial.suggest_categorical('num_filters', [16, 32, 64])
    trial_config['model']['conv_filters'] = [num_filters, num_filters, num_filters]
    kernel_size_x = trial.suggest_categorical('kernel_size_x', [3, 5, 7])
    kernel_size_y = trial.suggest_categorical('kernel_size_y', [3, 5])
    trial_config['model']['kernel_sizes'] = [
        [kernel_size_x, kernel_size_y], 
        [3, 3], 
        [3, 3]
    ]
    fc_dim = trial.suggest_categorical('fc_dim', [128, 256, 512])
    trial_config['model']['fc_dim'] = fc_dim
    trial_config['model']['dropout_rate'] = trial.suggest_float('dropout_rate', 0.2, 0.6)

    # ==== 支持改进模型的超参数 ====
    # improved_enabled = trial.suggest_categorical('improved_enabled', [True, False])
    # trial_config['model']['improved']['enabled'] = improved_enabled
    improved_enabled = trial_config['model']['improved']['enabled']
    if improved_enabled:
        trial_config['model']['improved']['batch_norm'] = trial.suggest_categorical('improved_batch_norm', [True, False])
        trial_config['model']['improved']['residual'] = trial.suggest_categorical('improved_residual', [True, False])
        trial_config['model']['improved']['skip_stride'] = [
            trial.suggest_categorical('improved_skip_stride_0', [2, 4]),
            trial.suggest_categorical('improved_skip_stride_1', [1, 2])
        ]

    # Siamese specific parameters
    if trial_config['dataloader']['mode'] == 'siamese':
        trial_config['model']['siamese']['embedding_dim'] = fc_dim
        trial_config['model']['siamese']['hidden_dim'] = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        if trial_config['dataloader']['siamese']['use_contrastive']:
            trial_config['model']['siamese']['margin'] = trial.suggest_float('margin', 0.5, 2.0)
        if improved_enabled:
            # 支持改进版相似度网络
            trial_config['model']['siamese']['similarity_network']['use_batch_norm'] = trial.suggest_categorical('simnet_batch_norm', [True, False])
            trial_config['model']['siamese']['similarity_network']['additional_layer'] = trial.suggest_categorical('simnet_additional_layer', [True, False])
            trial_config['model']['siamese']['similarity_network']['dropout_reduction'] = trial.suggest_float('simnet_dropout_reduction', 0.2, 0.8)
            # 支持注意力机制
            trial_config['model']['siamese']['attention']['enabled'] = trial.suggest_categorical('attention_enabled', [True, False])
            trial_config['model']['siamese']['attention']['reduction_ratio'] = trial.suggest_categorical('attention_reduction_ratio', [2, 4, 8])

    # Training hyperparameters
    trial_config['training']['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    trial_config['training']['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    trial_config['training']['optimizer'] = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    trial_config['training']['epochs'] = min(trial_config['training']['epochs'], 30)
    window_length = trial.suggest_categorical('window_length', [0.25, 0.5, 1.0])
    trial_config['windowing']['window_length'] = window_length
    window_stride = trial.suggest_categorical('window_stride', [0.125, 0.25, 0.5])
    if window_stride > window_length:
        window_stride = window_length
    trial_config['windowing']['window_stride'] = window_stride

    # Preprocessing hyperparameters
    if trial.suggest_categorical('apply_filter', [True, False]):
        trial_config['preprocessing']['filter']['apply'] = True
        trial_config['preprocessing']['filter']['lowcut'] = trial.suggest_float('lowcut', 0.5, 5.0)
        trial_config['preprocessing']['filter']['highcut'] = trial.suggest_float('highcut', 30.0, 50.0)
    else:
        trial_config['preprocessing']['filter']['apply'] = False

    trial_config['preprocessing']['channels']['select'] = True
    trial_config['preprocessing']['channels']['set'] = 'epoc_x'

    validate_dimensions(trial_config)
    return trial_config


def save_trial_config(config, config_path):
    """
    Save trial configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Returns:
        None
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def clear_cache_for_trial(trial_config, logger):
    """
    Clear cache directories for a specific trial to force reprocessing.
    
    Args:
        trial_config: Configuration dictionary for the trial
        logger: Logger object
    """
    # Clear dataset pair cache if in siamese mode
    if trial_config['dataloader']['mode'] == 'siamese' and trial_config['dataloader']['siamese'].get('cache_pairs', False):
        cache_dir = Path(trial_config['dataloader']['siamese'].get('pairs_cache_dir', 'data/cache'))
        if cache_dir.exists():
            logger.info(f"Clearing siamese pair cache: {cache_dir}")
            for cache_file in cache_dir.glob('pairs_cache_*.h5'):
                try:
                    cache_file.unlink()
                    logger.info(f"Removed cache file: {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
    
    # Create a unique preprocessed data file for this trial
    trial_data_filename = f"eeg_data_trial_{hash(str(trial_config['windowing']))}.h5"
    trial_config['data']['output_filename'] = trial_data_filename
    
    return trial_config

def preprocess_data_for_trial(trial_config, logger):
    """
    Preprocess data specifically for a trial with given hyperparameters.
    
    Args:
        trial_config: Configuration dictionary for the trial
        logger: Logger object
        
    Returns:
        Path to the preprocessed data file
    """
    from preprocess import process_edf_file, segment_data
    import h5py
    from pathlib import Path
    
    # Update output filename to be unique for this trial
    window_config_str = f"{trial_config['windowing']['window_length']}_{trial_config['windowing']['window_stride']}"
    filter_config_str = ""
    if trial_config['preprocessing']['filter']['apply']:
        filter_config_str = f"_filt_{trial_config['preprocessing']['filter']['lowcut']}_{trial_config['preprocessing']['filter']['highcut']}"
    
    trial_data_filename = f"eeg_data_trial_{window_config_str}{filter_config_str}.h5"
    trial_config['data']['output_filename'] = trial_data_filename
    
    output_dir = Path(trial_config['data']['processed_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / trial_data_filename
    
    # Skip preprocessing if file already exists
    if output_path.exists():
        logger.info(f"Using existing preprocessed data file: {output_path}")
        return str(output_path)
    
    # Determine conditions to process
    conditions = []
    if trial_config['experiment']['include_eyes_open']:
        conditions.append('eyes_open')
    if trial_config['experiment']['include_eyes_closed']:
        conditions.append('eyes_closed')
    
    # Original sampling frequency for PhysioNet dataset
    orig_sfreq = 160
    
    # Get effective sampling frequency after potential resampling
    effective_sfreq = orig_sfreq
    if trial_config['preprocessing']['resample']['apply']:
        effective_sfreq = trial_config['preprocessing']['resample']['freq']
    
    logger.info(f"Preprocessing data with window length: {trial_config['windowing']['window_length']}, "
                f"stride: {trial_config['windowing']['window_stride']}")
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as hf:
        # Store metadata
        metadata = hf.create_group('metadata')
        metadata.attrs['window_length'] = trial_config['windowing']['window_length']
        metadata.attrs['window_stride'] = trial_config['windowing']['window_stride']
        metadata.attrs['original_sampling_frequency'] = orig_sfreq
        metadata.attrs['effective_sampling_frequency'] = effective_sfreq
        
        # Store filtering info
        metadata.attrs['filter_applied'] = trial_config['preprocessing']['filter']['apply']
        if trial_config['preprocessing']['filter']['apply']:
            metadata.attrs['lowcut'] = trial_config['preprocessing']['filter']['lowcut']
            metadata.attrs['highcut'] = trial_config['preprocessing']['filter']['highcut']
        
        # Store resampling info
        metadata.attrs['resampling_applied'] = trial_config['preprocessing']['resample']['apply']
        if trial_config['preprocessing']['resample']['apply']:
            metadata.attrs['target_frequency'] = trial_config['preprocessing']['resample']['freq']
        
        # Store normalization info
        metadata.attrs['normalization_applied'] = trial_config['preprocessing']['normalize']
        
        # Store channel selection info
        metadata.attrs['channel_selection'] = trial_config['preprocessing']['channels']['select']
        if trial_config['preprocessing']['channels']['select']:
            metadata.attrs['channel_set'] = trial_config['preprocessing']['channels']['set']
            if trial_config['preprocessing']['channels']['set'] == 'epoc_x':
                metadata.attrs['num_channels'] = 14
            else:
                metadata.attrs['num_channels'] = 64
        
        # Store full config as string
        metadata.attrs['config_yaml'] = yaml.dump(trial_config)
        
        # Process each condition and subject
        data_group = hf.create_group('data')
        
        for condition in conditions:
            condition_path = Path(trial_config['data']['raw_dir']) / condition
            if not condition_path.exists():
                logger.warning(f"Condition directory {condition_path} does not exist")
                continue
            
            # Create group for this condition
            condition_group = data_group.create_group(condition)
            
            # Get subject directories
            subject_dirs = [d for d in condition_path.iterdir() if d.is_dir()]
            
            # Process each subject
            from tqdm import tqdm
            for subject_dir in tqdm(subject_dirs, desc=f"Processing {condition}"):
                subject_id = subject_dir.name
                edf_file = subject_dir / f"{subject_id}_eeg.edf"
                
                if not edf_file.exists():
                    logger.warning(f"EDF file {edf_file} does not exist")
                    continue
                
                # Process EDF file
                windows, sfreq = process_edf_file(str(edf_file), trial_config, orig_sfreq)
                
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
    
    logger.info(f"Preprocessing complete. Output saved to {output_path}")
    return str(output_path)

def train_one_epoch(model, train_loader, criterion, optimizer, device, mode='identity', use_contrastive=False, show_progress=False):
    """
    Custom implementation of training for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        mode: 'identity' or 'siamese'
        use_contrastive: Whether to use contrastive loss (only for siamese mode)
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    progress_bar = tqdm(train_loader, desc="Training", leave=False, disable=not show_progress)
    
    for batch in progress_bar:
        if mode == 'identity':
            # Unpack batch
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        else:  # siamese
            # Unpack batch
            inputs1, inputs2, labels = batch
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            if use_contrastive:
                _, _, distance = model(inputs1, inputs2)
                loss = criterion(distance, labels)
                
                # For accuracy calculation (small distance = same subject)
                predictions = (distance < 1.0).float()
            else:
                outputs = model(inputs1, inputs2)
                loss = criterion(outputs.squeeze(), labels)
                
                # For accuracy calculation (output > 0.5 = same subject)
                predictions = (outputs.squeeze() > 0.5).float()
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate statistics
            running_loss += loss.item() * inputs1.size(0)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    
    # Calculate final metrics
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total if total > 0 else 0
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc
    }


def evaluate_model(model, val_loader, criterion, device, mode='identity', use_contrastive=False, show_progress=False):
    """
    Custom implementation of model evaluation.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on
        mode: 'identity' or 'siamese'
        use_contrastive: Whether to use contrastive loss (only for siamese mode)
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False, disable=not show_progress):
            if mode == 'identity':
                # Unpack batch
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Convert log probabilities to probabilities
                scores = torch.exp(outputs)
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                
                # Calculate statistics
                running_loss += loss.item() * inputs.size(0)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for later analysis
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.append(scores.cpu().numpy())
                
            else:  # siamese
                # Unpack batch
                inputs1, inputs2, labels = batch
                inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
                
                # Forward pass
                if use_contrastive:
                    _, _, distance = model(inputs1, inputs2)
                    loss = criterion(distance, labels)
                    
                    # Convert distance to similarity score (closer = more similar)
                    scores = 1 / (1 + distance)
                    predictions = (distance < 1.0).float()
                else:
                    outputs = model(inputs1, inputs2)
                    loss = criterion(outputs.squeeze(), labels)
                    
                    scores = outputs.squeeze()
                    predictions = (scores > 0.5).float()
                
                # Calculate statistics
                running_loss += loss.item() * inputs1.size(0)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
                
                # Store for later analysis
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
    
    # Calculate final metrics
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = 100 * correct / total if total > 0 else 0
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Return basic metrics dictionary
    metrics = {
        'loss': val_loss,
        'accuracy': val_acc,
        'predictions': all_preds,
        'labels': all_labels,
    }
    
    # For siamese mode, add siamese-specific metrics
    if mode == 'siamese':
        all_scores = np.array(all_scores)
        metrics['scores'] = all_scores
        
        # Calculate ROC curve and AUC if sklearn is available
        try:
            from sklearn.metrics import roc_curve, auc
            
            # Calculate ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
            roc_auc = auc(fpr, tpr)
            
            # Calculate EER (Equal Error Rate)
            fnr = 1 - tpr
            abs_diff = np.abs(fpr - fnr)
            eer_idx = np.argmin(abs_diff)
            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
            
            metrics.update({
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'roc_auc': roc_auc,
                'eer': eer
            })
        except ImportError:
            # If sklearn is not available, skip these metrics
            pass
    
    return metrics


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, output_dir):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses
        train_accs: List of training accuracies
        val_losses: List of validation losses
        val_accs: List of validation accuracies
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 5))
        
        # Plot loss curves
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curves')
        
        # Plot accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train')
        plt.plot(val_accs, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy Curves')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'))
        plt.close()
    except ImportError:
        # If matplotlib is not available, skip plotting
        pass


def train_trial_model(trial_config, trial_dir, logger):
    """
    Custom implementation of model training for a trial.
    This avoids modifying the global config.yaml or using the main train.py.
    
    Args:
        trial_config: Configuration dictionary for the trial
        trial_dir: Directory to save model and results
        logger: Logger object
        
    Returns:
        Dictionary of results including best metrics and model path
    """
    
        # 清除缓存并创建特定于该试验的配置
    trial_config = clear_cache_for_trial(trial_config, logger)
    
    # 预处理数据
    try:
        data_file = preprocess_data_for_trial(trial_config, logger)
        logger.info(f"Using preprocessed data file: {data_file}")
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise
    

    # Set random seed for reproducibility
    seed = trial_config['dataloader']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Set device
    use_gpu = trial_config['training'].get('use_gpu', True)
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Save the trial configuration to a temporary file in the trial directory
    temp_config_path = trial_dir / 'config.yaml'
    save_trial_config(trial_config, temp_config_path)
    
    # # ====== 强制重新预处理数据，避免复用旧的预处理结果 ======
    # # 假设 get_dataloaders 使用 config['preprocessing']['cache_dir'] 或类似字段作为缓存目录
    # # 你需要根据你的实际数据缓存实现调整下面的路径
    # preprocessing_cache_dir = None
    # if 'preprocessing' in trial_config and 'cache_dir' in trial_config['preprocessing']:
    #     preprocessing_cache_dir = trial_config['preprocessing']['cache_dir']
    # else:
    #     # 假设默认缓存目录为 data/preprocessed
    #     preprocessing_cache_dir = 'data/preprocessed'
    # # 为每个 trial 使用独立的缓存目录（推荐），或者直接删除缓存目录
    # import shutil
    # if os.path.exists(preprocessing_cache_dir):
    #     logger.info(f"Removing preprocessing cache directory: {preprocessing_cache_dir}")
    #     shutil.rmtree(preprocessing_cache_dir)
    # # ====== 结束强制重新预处理 ======

    # Create data loaders using the trial configuration
    logger.info("Creating DataLoaders...")
    train_loader, val_loader = get_dataloaders(config_path=str(temp_config_path))
    
    # Get dataset info
    train_dataset = train_loader.dataset
    mode = train_dataset.mode
    use_contrastive = trial_config['dataloader']['siamese'].get('use_contrastive', False) if mode == 'siamese' else False
    
    logger.info(f"Mode: {mode}")
    # logger.info(f"Original Number of classes: {train_dataset.num_classes}")
    if mode == 'identity':
        logger.info(f"Number of classes: {train_dataset.num_classes}")
    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(val_loader.dataset)}")
    
    # Get input dimensions
    if mode == 'identity':
        sample, _ = train_dataset[0]
    else:
        sample, _, _ = train_dataset[0]
    
    num_channels, num_samples = sample.shape
    
    # Create model
    logger.info("Creating Model...")
    if mode == 'identity':
        model = ICAConvNet(
            num_channels=num_channels, 
            num_samples=num_samples, 
            num_classes=train_dataset.num_classes, 
            config_path=str(temp_config_path)
        ).to(device)
        
        # Loss function
        criterion = nn.NLLLoss()
        
    elif use_contrastive:
        model = ContrastiveSiameseNet(
            num_channels=num_channels, 
            num_samples=num_samples, 
            config_path=str(temp_config_path)
        ).to(device)
        
        # Contrastive loss with configurable margin
        margin = trial_config['model']['siamese'].get('margin', 1.0)
        criterion = lambda distance, label: contrastive_loss(distance, label, margin=margin)
        
    else:  # standard siamese
        model = SiameseICAConvNet(
            num_channels=num_channels, 
            num_samples=num_samples, 
            config_path=str(temp_config_path)
        ).to(device)
        
        # Loss function
        criterion = nn.BCELoss()
    
    # Create optimizer
    learning_rate = trial_config['training']['learning_rate']
    weight_decay = trial_config['training']['weight_decay']
    optimizer_name = trial_config['training']['optimizer'].lower()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:  # sgd
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    
    # Training parameters
    num_epochs = trial_config['training']['epochs']
    early_stopping = trial_config['training'].get('early_stopping', False)
    patience = trial_config['training'].get('patience', 5)
    
    # Initialize variables for early stopping
    best_val_acc = 0.0
    best_val_eer = 1.0  # Lower is better for EER
    best_epoch = 0
    epochs_no_improve = 0
    best_metrics = None
    
    # Lists to store metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train one epoch
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            mode=mode,
            use_contrastive=use_contrastive
        )
        
        # Evaluate on validation set
        val_metrics = evaluate_model(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            mode=mode,
            use_contrastive=use_contrastive
        )
        
        # Log metrics
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
        
        if mode == 'siamese' and 'eer' in val_metrics:
            logger.info(f"Val ROC AUC: {val_metrics['roc_auc']:.4f}, Val EER: {val_metrics['eer']:.4f}")
        
        # Store metrics
        train_losses.append(train_metrics['loss'])
        train_accs.append(train_metrics['accuracy'])
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['accuracy'])
        
        # Plot training curves
        plot_training_curves(train_losses, train_accs, val_losses, val_accs, trial_dir)
        
        # Save current model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy'],
            'val_metrics': val_metrics
        }, trial_dir / 'last_model.pt')
        
        # Check if this is the best model
        improved = False
        
        if mode == 'identity' or not use_contrastive:
            # For identity or standard siamese, use accuracy
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_epoch = epoch
                best_metrics = val_metrics
                improved = True
        else:
            # For contrastive siamese, use EER (lower is better)
            if 'eer' in val_metrics and val_metrics['eer'] < best_val_eer:
                best_val_eer = val_metrics['eer']
                best_epoch = epoch
                best_metrics = val_metrics
                improved = True
        
        if improved:
            epochs_no_improve = 0
            logger.info(f"New best model at epoch {epoch+1}!")
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_acc': val_metrics['accuracy'],
                'val_metrics': val_metrics
            }, trial_dir / 'best_model.pt')
            
        else:
            epochs_no_improve += 1
            
        # Early stopping
        if early_stopping and epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs!")
            logger.info(f"Best model was at epoch {best_epoch+1}")
            break
    
    # Training finished
    logger.info("Training completed")
    if mode == 'identity' or not use_contrastive:
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch+1})")
    else:
        logger.info(f"Best validation EER: {best_val_eer:.4f} (epoch {best_epoch+1})")
    
    # Save final metrics
    np.savez(
        trial_dir / 'training_history.npz',
        train_losses=train_losses,
        train_accs=train_accs,
        val_losses=val_losses,
        val_accs=val_accs,
        best_epoch=best_epoch
    )
    
    # Return results
    return {
        'best_metrics': best_metrics,
        'best_epoch': best_epoch,
        'best_model_path': str(trial_dir / 'best_model.pt'),
        'final_model_path': str(trial_dir / 'last_model.pt')
    }


def objective(trial, study_dir, logger):
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        study_dir: Directory for study outputs
        logger: Logger object
        
    Returns:
        Performance metric to be minimized (EER) or maximized (accuracy)
    """
    # Load base configuration without modifying the original
    config_path = 'configs/config.yaml'
    base_config = load_config(config_path)
    
    # Create trial configuration
    trial_config = create_trial_config(base_config, trial)
    # logger.info(f"Trial {trial.number} configuration: {trial_config}")
    
    # Create trial directory within study directory
    trial_dir = study_dir / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trial configuration
    trial_config_path = trial_dir / 'config.yaml'
    save_trial_config(trial_config, trial_config_path)
    logger.info(f"Saved trial configuration to {trial_config_path}")
    
    # Train model with trial configuration
    try:
        # Train the model using our custom training implementation
        logger.info(f"Starting trial {trial.number} training")
        results = train_trial_model(trial_config, trial_dir, logger)
        
        # Extract metrics
        best_metrics = results.get('best_metrics', {})
        best_epoch = results.get('best_epoch', 0)
        
        if best_metrics:
            # For siamese with contrastive loss, use EER (lower is better)
            if trial_config['dataloader']['mode'] == 'siamese':
                if 'eer' in best_metrics:
                    metric = best_metrics['eer']
                    logger.info(f"Trial {trial.number} finished with EER: {metric:.4f} at epoch {best_epoch+1}")
                    return metric
                else:
                    logger.warning(f"No EER found in best metrics for trial {trial.number}")
                    return 1.0  # Worst EER
            else:
                # For identity or standard siamese, use accuracy (higher is better)
                metric = best_metrics.get('accuracy', 0.0)
                logger.info(f"Trial {trial.number} finished with Accuracy: {metric:.2f}% at epoch {best_epoch+1}")
                return 100.0 - metric  # Convert to minimization problem
        else:
            logger.warning(f"No best metrics found for trial {trial.number}")
            return 100.0 if trial_config['dataloader']['mode'] == 'siamese' and trial_config['dataloader']['siamese'].get('use_contrastive', False) else 100.0
        
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {e}")
        return 100.0 if trial_config['dataloader']['mode'] == 'siamese' and trial_config['dataloader']['siamese'].get('use_contrastive', False) else 100.0


def main():
    """
    Main function to run Optuna hyperparameter tuning.
    """
    # Determine model name from config
    config = load_config('configs/config.yaml')
    model_mode = config['dataloader']['mode']
    
    if (model_mode == 'siamese'):
        if config['dataloader']['siamese'].get('use_contrastive', False):
            model_name = "contrastive_siamese"
        else:
            model_name = "siamese"
    else:
        model_name = "identity"
    
    # Create study directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    study_dir = Path(f"outputs/results/optuna/{model_name}_{timestamp}")
    study_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(study_dir)
    logger.info(f"Created study directory: {study_dir}")
    logger.info("Starting hyperparameter tuning with Optuna")
    
    # Create storage file within the study directory
    storage_file = study_dir / f"{model_name}_study.db"
    storage_name = f"sqlite:///{storage_file}"
    
    logger.info(f"Study name: {model_name}_{timestamp}")
    logger.info(f"Storage: {storage_name}")
    
    # Create study and optimize
    study = optuna.create_study(
        study_name=f"{model_name}_{timestamp}",
        storage=storage_name,
        direction="minimize",  # Minimize EER or (100 - Accuracy)
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Number of trials
    n_trials = 30  # Adjust based on available computational resources
    logger.info(f"Running {n_trials} trials")
    
    # Run optimization with study_dir passed to objective
    study.optimize(lambda trial: objective(trial, study_dir, logger), n_trials=n_trials)
    
    # Get best trial
    best_trial = study.best_trial
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best value: {best_trial.value}")
    logger.info("Best hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    # Save best configuration to study directory
    best_config_path = study_dir / 'best_config.yaml'
    # Load base configuration
    base_config = load_config('configs/config.yaml')
    # Create best configuration
    best_config = create_trial_config(base_config, best_trial)
    # Save best configuration
    save_trial_config(best_config, best_config_path)
    logger.info(f"Saved best configuration to {best_config_path}")
    
    # Also save to configs directory for easy access
    configs_best_path = Path(f"configs/best_{model_name}_{timestamp}.yaml")
    save_trial_config(best_config, configs_best_path)
    logger.info(f"Saved best configuration to {configs_best_path}")
    
    # Print importance of hyperparameters
    logger.info("Hyperparameter importance:")
    try:
        importance = optuna.importance.get_param_importances(study)
        for key, value in importance.items():
            logger.info(f"  {key}: {value}")
    except Exception as e:
        logger.error(f"Error calculating hyperparameter importance: {e}")
    
    # Create visualizations if matplotlib and plotly are available
    try:
        import matplotlib.pyplot as plt
        import plotly
        
        # Create visualization directory within study directory
        viz_dir = study_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(str(viz_dir / "optimization_history.png"))
        
        # Plot parameter importances
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(str(viz_dir / "param_importances.png"))
        
        # Plot parameter relationships
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(str(viz_dir / "parallel_coordinate.png"))
        
        # Plot contour plots for top parameters
        try:
            most_important_params = list(importance.keys())[:3]  # Top 3 parameters
            if len(most_important_params) >= 2:
                fig = optuna.visualization.plot_contour(study, params=most_important_params[:2])
                fig.write_image(str(viz_dir / "contour_plot.png"))
        except Exception as e:
            logger.warning(f"Error creating contour plot: {e}")
        
        # Plot slice plots for key parameters
        for param in study.best_trial.params.keys():
            try:
                fig = optuna.visualization.plot_slice(study, params=[param])
                fig.write_image(str(viz_dir / f"slice_{param}.png"))
            except Exception as e:
                logger.warning(f"Error creating slice plot for {param}: {e}")
        
        logger.info(f"Saved visualizations to {viz_dir}")
    except ImportError:
        logger.warning("Matplotlib or plotly not available, skipping visualizations")
    
    # Save study data for later analysis
    study_data = {
        'study_name': f"{model_name}_{timestamp}",
        'best_trial': best_trial.number,
        'best_value': best_trial.value,
        'best_params': best_trial.params,
        'n_trials': n_trials,
        'trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': t.state.name,
                'datetime_start': t.datetime_start.isoformat() if t.datetime_start else None,
                'datetime_complete': t.datetime_complete.isoformat() if t.datetime_complete else None,
            }
            for t in study.trials
        ]
    }
    
    # Save as YAML
    with open(study_dir / 'study_data.yaml', 'w') as f:
        yaml.dump(study_data, f)
    
    logger.info(f"Hyperparameter tuning completed. All results saved to {study_dir}")
    
    # Return path to best configuration for potential further use
    return str(configs_best_path)

if __name__ == "__main__":
    main()