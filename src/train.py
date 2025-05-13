#!/usr/bin/env python3
"""
Training script for EEG-based personal identification models.

This script trains models (ICAConvNet, SiameseICAConvNet, or ContrastiveSiameseNet)
using parameters from the configs/config.yaml file.
"""

import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import logging
import sys

from dataset import get_dataloaders, load_config
from model import ICAConvNet, SiameseICAConvNet, ContrastiveSiameseNet, contrastive_loss, create_model


def setup_logging(output_dir=None, level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        output_dir: Directory to save log file (if None, only console logging is configured)
        level: Logging level
    
    Returns:
        Logger object
    """
    # Create logger
    logger = logging.getLogger('eeg_train')
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if output directory is provided
    if output_dir:
        file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def train_one_epoch(model, train_loader, criterion, optimizer, device, mode='identity', use_contrastive=False, show_progress=False):
    """
    Train model for one epoch.
    
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
    progress_bar = tqdm(train_loader, desc="Training", disable=not show_progress)
    
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
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total if total > 0 else 0
            })
            
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
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total if total > 0 else 0
            })
    
    # Calculate final metrics
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total if total > 0 else 0
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc
    }


def evaluate(model, val_loader, criterion, device, mode='identity', use_contrastive=False, show_progress=False):
    """
    Evaluate model on validation set.
    
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
        for batch in tqdm(val_loader, desc="Validating", disable=not show_progress):
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
    
    if mode == 'identity':
        all_scores = np.vstack(all_scores)
    else:
        all_scores = np.array(all_scores)
    
    # Additional metrics for siamese mode
    if mode == 'siamese':
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)
        
        # Calculate EER
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        
        return {
            'loss': val_loss,
            'accuracy': val_acc,
            'predictions': all_preds,
            'labels': all_labels,
            'scores': all_scores,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'roc_auc': roc_auc,
            'eer': eer
        }
    else:
        # For identity mode, calculate confusion matrix 
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        return {
            'loss': val_loss,
            'accuracy': val_acc,
            'predictions': all_preds,
            'labels': all_labels,
            'scores': all_scores,
            'confusion_matrix': conf_matrix
        }


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
    
    # Plot accuracy zoomed if we have enough epochs
    if len(train_accs) > 10:
        plt.figure(figsize=(8, 6))
        plt.plot(train_accs, label='Train')
        plt.plot(val_accs, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy Curves (Last 80%)')
        
        # Zoom in on the last 80% of training (after initial rapid changes)
        num_epochs = len(train_accs)
        start_idx = int(num_epochs * 0.2)
        
        # Calculate y-axis limits with some padding
        min_acc = min(min(train_accs[start_idx:]), min(val_accs[start_idx:]))
        max_acc = max(max(train_accs[start_idx:]), max(val_accs[start_idx:]))
        padding = (max_acc - min_acc) * 0.1
        
        plt.xlim(start_idx, num_epochs - 1)
        plt.ylim(max(0, min_acc - padding), min(100, max_acc + padding))
        
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_curves_zoomed.png'))
        plt.close()


def train(config_path='configs/config.yaml'):
    """
    Train model based on configuration.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Path to output directory
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    seed = config['dataloader']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create output directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"models/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir)
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Set device
    use_gpu = config['training'].get('use_gpu', True)
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating DataLoaders...")
    train_loader, val_loader = get_dataloaders(config_path=config_path)
    
    # Get dataset info
    train_dataset = train_loader.dataset
    mode = train_dataset.mode
    use_contrastive = config['dataloader']['siamese'].get('use_contrastive', False) if mode == 'siamese' else False
    
    logger.info("=== Dataset Info ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Number of classes: {train_dataset.num_classes}")
    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(val_loader.dataset)}")
    
    # Get input dimensions
    if mode == 'identity':
        sample, _ = train_dataset[0]
    else:
        sample, _, _ = train_dataset[0]
    num_channels, num_samples = sample.shape

    # Create model using factory (supports improved model)
    logger.info("=== Creating Model ===")
    if mode == 'identity':
        model = create_model(
            config_path=config_path,
            num_channels=num_channels,
            num_samples=num_samples,
            num_classes=train_dataset.num_classes
        ).to(device)
        logger.info(f"Created {type(model).__name__} with {train_dataset.num_classes} output classes")
        criterion = nn.NLLLoss()
    elif use_contrastive:
        model = create_model(
            config_path=config_path,
            num_channels=num_channels,
            num_samples=num_samples
        ).to(device)
        logger.info(f"Created {type(model).__name__}")
        margin = config['model']['siamese'].get('margin', 1.0)
        criterion = lambda distance, label: contrastive_loss(distance, label, margin=margin)
    else:
        model = create_model(
            config_path=config_path,
            num_channels=num_channels,
            num_samples=num_samples
        ).to(device)
        logger.info(f"Created {type(model).__name__}")
        criterion = nn.BCELoss()
    
    # Create optimizer
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    optimizer_name = config['training']['optimizer'].lower()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:  # sgd
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    
    # Training parameters
    num_epochs = config['training']['epochs']
    early_stopping = config['training'].get('early_stopping', False)
    patience = config['training'].get('patience', 5)
    
    # Initialize variables for early stopping
    best_val_acc = 0.0
    best_val_eer = 1.0  # Lower is better for EER
    best_epoch = 0
    epochs_no_improve = 0
    
    # Lists to store metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Training loop
    logger.info("=== Starting Training ===")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Early stopping: {'Enabled' if early_stopping else 'Disabled'}")
    if early_stopping:
        logger.info(f"Patience: {patience}")
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
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
        val_metrics = evaluate(
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
        
        if mode == 'siamese':
            logger.info(f"Val ROC AUC: {val_metrics['roc_auc']:.4f}, Val EER: {(val_metrics['eer'] * 100):.4f}")
        
        # Store metrics
        train_losses.append(train_metrics['loss'])
        train_accs.append(train_metrics['accuracy'])
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['accuracy'])
        
        # Plot training curves
        plot_training_curves(train_losses, train_accs, val_losses, val_accs, output_dir)
        
        # Save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy'],
            'val_metrics': val_metrics
        }, output_dir / 'last_model.pt')
        
        # Check if this is the best model
        improved = False
        
        if mode == 'identity':
            # For identity or standard siamese, use accuracy
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_epoch = epoch
                improved = True
        else:
            # For siamese, use EER (lower is better)
            if val_metrics['eer'] < best_val_eer:
                best_val_eer = val_metrics['eer']
                best_epoch = epoch
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
            }, output_dir / 'best_model.pt')
            
            # Save training progress
            np.savez(
                output_dir / 'training_progress.npz',
                train_losses=train_losses,
                train_accs=train_accs,
                val_losses=val_losses,
                val_accs=val_accs,
                best_epoch=best_epoch,
                best_val_acc=best_val_acc,
                best_val_eer=best_val_eer
            )
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve}/{patience} epochs")
            # logger.info(f"Best model was at epoch {best_epoch+1}")
            # logger.info(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch+1})")
            logger.info(f"Best validation EER: {best_val_eer:.4f} (epoch {best_epoch+1})")
            
        # Early stopping
        if early_stopping and epochs_no_improve >= patience:
            logger.info(f"\nEarly stopping triggered after {epoch+1} epochs!")
            logger.info(f"Best model was at epoch {best_epoch+1}")
            break
    
    # Training finished
    logger.info("\n=== Training Completed ===")
    if mode == 'identity':
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch+1})")
    else:
        logger.info(f"Best validation EER: {best_val_eer:.4f} (epoch {best_epoch+1})")
    
    # Final evaluation on validation set
    logger.info("\nFinal evaluation on validation set:")
    final_metrics = evaluate(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        mode=mode,
        use_contrastive=use_contrastive
    )
    
    # Save final metrics
    logger.info(f"Saving final model and metrics to {output_dir}")
    np.savez(
        output_dir / 'final_metrics.npz',
        **final_metrics
    )
    
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train EEG model")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration YAML file')
    args = parser.parse_args()

    try:
        # 不再在这里配置 logging，setup_logging 会在 train() 内部处理
        print(f"Starting training with config: {args.config}")
        output_dir = train(config_path=args.config)
        print(f"Training completed. Results saved to {output_dir}")
    except Exception as e:
        # 这里可以用 print 或简单 logging
        print(f"Training failed with error: {e}")
        raise