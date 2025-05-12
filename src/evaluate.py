"""
Evaluation script for EEG-based personal identification models.
Calculates metrics including accuracy per subject, confusion matrix, 
ROC curve and Equal Error Rate (EER).

Uses configuration from configs/config.yaml.
"""

import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm
from pathlib import Path

from dataset import get_dataloaders, load_config
from model import ICAConvNet, SiameseICAConvNet, ContrastiveSiameseNet, create_model


def calculate_eer(fpr, tpr, thresholds):
    """
    Calculate Equal Error Rate (EER).
    """
    # Calculate False Negative Rate (fnr)
    fnr = 1 - tpr
    
    # Find the threshold where fpr and fnr are closest
    abs_diff = np.abs(fpr - fnr)
    idx = np.argmin(abs_diff)
    
    # Get EER and threshold
    eer = (fpr[idx] + fnr[idx]) / 2
    threshold = thresholds[idx]
    
    return eer, threshold


def evaluate_identity_model(model, data_loader, device):
    """
    Evaluate identity model (standard classification).
    """
    model.eval()
    
    # Initialize arrays for predictions and labels
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            scores = torch.exp(outputs)  # Convert log probabilities to probabilities
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.vstack(all_scores)
    
    # Calculate accuracy
    accuracy = np.mean(all_preds == all_labels) * 100
    
    # Calculate accuracy per subject
    unique_labels = np.unique(all_labels)
    accs_per_subject = {}
    
    for label in unique_labels:
        mask = all_labels == label
        if np.sum(mask) > 0:
            accs_per_subject[int(label)] = np.mean(all_preds[mask] == all_labels[mask]) * 100
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Calculate ROC curves and EER for one-vs-rest classification
    n_classes = len(unique_labels)
    fpr = {}
    tpr = {}
    roc_auc = {}
    eer = {}
    eer_threshold = {}
    
    for i in range(n_classes):
        # Prepare binary labels
        # For each class, we're doing one-vs-rest
        binary_labels = (all_labels == i).astype(int)
        
        # Get scores for this class
        class_scores = all_scores[:, i]
        
        # Calculate ROC curve
        fpr[i], tpr[i], thresholds = roc_curve(binary_labels, class_scores)
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Calculate EER
        eer[i], eer_threshold[i] = calculate_eer(fpr[i], tpr[i], thresholds)
    
    # Calculate average EER
    avg_eer = np.mean(list(eer.values()))
    
    # Return results
    return {
        'accuracy': accuracy,
        'accs_per_subject': accs_per_subject,
        'confusion_matrix': conf_matrix,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'eer': eer,
        'avg_eer': avg_eer,
        'eer_threshold': eer_threshold,
        'predictions': all_preds,
        'labels': all_labels,
        'scores': all_scores
    }


def evaluate_siamese_model(model, data_loader, device, use_contrastive=False):
    """
    Evaluate siamese model (similarity comparison).
    """
    model.eval()
    
    # Initialize arrays for predictions and labels
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for inputs1, inputs2, labels in tqdm(data_loader, desc="Evaluating"):
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            
            # Forward pass
            if use_contrastive:
                _, _, distance = model(inputs1, inputs2)
                # Convert distance to similarity score (closer = more similar)
                scores = 1 / (1 + distance)
                # Threshold at 0.5 (distance of 1.0)
                preds = (distance < 1.0).float()
            else:
                outputs = model(inputs1, inputs2)
                scores = outputs.squeeze()
                preds = (scores > 0.5).float()
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Calculate accuracy
    accuracy = np.mean(all_preds == all_labels) * 100
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate EER
    eer, eer_threshold = calculate_eer(fpr, tpr, thresholds)
    
    # Calculate true positive rate and true negative rate at EER threshold
    preds_at_eer = (all_scores >= eer_threshold).astype(int)
    tp = np.sum((preds_at_eer == 1) & (all_labels == 1))
    tn = np.sum((preds_at_eer == 0) & (all_labels == 0))
    fp = np.sum((preds_at_eer == 1) & (all_labels == 0))
    fn = np.sum((preds_at_eer == 0) & (all_labels == 1))
    
    tpr_at_eer = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr_at_eer = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Return results
    return {
        'accuracy': accuracy,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'tpr_at_eer': tpr_at_eer,
        'tnr_at_eer': tnr_at_eer,
        'predictions': all_preds,
        'labels': all_labels,
        'scores': all_scores
    }


def plot_results(results, output_dir, mode='identity', use_contrastive=False):
    """
    Plot and save evaluation results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if mode == 'identity':
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(results['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Plot ROC curves for top 10 classes by EER
        plt.figure(figsize=(10, 8))
        
        # Sort classes by EER
        classes_by_eer = sorted(results['eer'].items(), key=lambda x: x[1])
        classes_to_plot = [class_idx for class_idx, _ in classes_by_eer[:10]]
        
        for i in classes_to_plot:
            plt.plot(
                results['fpr'][i], 
                results['tpr'][i], 
                lw=2, 
                label=f'Class {i} (AUC = {results["roc_auc"][i]:.2f}, EER = {results["eer"][i]:.2f})'
            )
        
        plt.plot([0, 1], [1, 0], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Top 10 Classes')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curves_top10.png'))
        plt.close()
        
        # Plot EER histogram
        plt.figure(figsize=(10, 6))
        plt.hist(list(results['eer'].values()), bins=20, alpha=0.7)
        plt.axvline(results['avg_eer'], color='r', linestyle='--', 
                    label=f'Average EER: {results["avg_eer"]:.4f}')
        plt.xlabel('Equal Error Rate (EER)')
        plt.ylabel('Number of Classes')
        plt.title('Distribution of EER across Classes')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eer_histogram.png'))
        plt.close()
        
        # Plot accuracy per subject
        plt.figure(figsize=(12, 6))
        subjects = list(results['accs_per_subject'].keys())
        accs = list(results['accs_per_subject'].values())
        
        # Sort by accuracy
        sorted_idx = np.argsort(accs)
        subjects = [subjects[i] for i in sorted_idx]
        accs = [accs[i] for i in sorted_idx]
        
        plt.bar(range(len(subjects)), accs)
        plt.xlabel('Subject Index')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy per Subject')
        plt.xticks([])  # Hide subject indices to reduce clutter
        plt.axhline(results['accuracy'], color='r', linestyle='--', 
                    label=f'Overall Accuracy: {results["accuracy"]:.2f}%')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_per_subject.png'))
        plt.close()
        
    else:  # siamese mode
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(
            results['fpr'], 
            results['tpr'], 
            lw=2, 
            label=f'ROC curve (AUC = {results["roc_auc"]:.2f})'
        )
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Plot EER point
        eer_idx = np.argmin(np.abs(results['fpr'] - (1 - results['tpr'])))
        plt.scatter(
            results['fpr'][eer_idx], 
            results['tpr'][eer_idx], 
            s=100, 
            c='r', 
            marker='o', 
            label=f'EER = {results["eer"]:.4f}'
        )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        
        # Plot score distributions
        plt.figure(figsize=(10, 6))
        
        pos_scores = results['scores'][results['labels'] == 1]
        neg_scores = results['scores'][results['labels'] == 0]
        
        plt.hist(pos_scores, bins=50, alpha=0.5, label='Same Subject (Positive)')
        plt.hist(neg_scores, bins=50, alpha=0.5, label='Different Subjects (Negative)')
        
        plt.axvline(results['eer_threshold'], color='r', linestyle='--', 
                    label=f'EER Threshold: {results["eer_threshold"]:.4f}')
        
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.title('Distribution of Similarity Scores')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'score_distributions.png'))
        plt.close()
        
        # Plot DET curve (Detection Error Tradeoff)
        plt.figure(figsize=(10, 8))
        
        # Calculate FRR (False Rejection Rate) = 1 - TPR
        frr = 1 - results['tpr']
        
        plt.plot(results['fpr'], frr, lw=2)
        plt.scatter(results['eer'], results['eer'], s=100, c='r', marker='o', 
                   label=f'EER = {results["eer"]:.4f}')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([0.001, 1.0])
        plt.ylim([0.001, 1.0])
        plt.xlabel('False Acceptance Rate (FAR)')
        plt.ylabel('False Rejection Rate (FRR)')
        plt.title('Detection Error Tradeoff (DET) Curve')
        plt.grid(True, which="both", ls="-")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'det_curve.png'))
        plt.close()
        
def main():
    """Main function for model evaluation using configuration from YAML file."""
    # Load configuration
    config_path = 'configs/config.yaml'
    config = load_config(config_path)
    
    # Extract evaluation settings
    eval_config = config['evaluation']
    model_path = eval_config['model_path']
    output_dir = eval_config['output_dir']
    plot_results_enabled = eval_config.get('plot_results', True)
    
    # Extract mode settings
    dataloader_config = config['dataloader']
    mode = dataloader_config['mode']
    use_contrastive = dataloader_config['siamese'].get('use_contrastive', False) if mode == 'siamese' else False
    
    # Set device
    use_gpu = config['training'].get('use_gpu', True)
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loader
    print("\nCreating DataLoader...")
    _, test_loader = get_dataloaders(config_path=config_path)
    
    # Get dataset info
    dataset = test_loader.dataset
    print(f"\n=== Dataset Info ===")
    print(f"Mode: {dataset.mode}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of test samples: {len(dataset)}")
    
    # Get input dimensions
    if mode == 'identity':
        sample, _ = dataset[0]
    else:
        sample, _, _ = dataset[0]
    
    num_channels, num_samples = sample.shape
    
    # Create and load model
    print(f"\n=== Loading Model ===")
    if mode == 'identity':
        model = ICAConvNet(
            num_channels=num_channels, 
            num_samples=num_samples, 
            num_classes=dataset.num_classes, 
            config_path=config_path
        ).to(device)
        print(f"Created ICAConvNet with {dataset.num_classes} output classes")
    elif use_contrastive:
        model = ContrastiveSiameseNet(
            num_channels=num_channels, 
            num_samples=num_samples, 
            config_path=config_path
        ).to(device)
        print(f"Created ContrastiveSiameseNet")
    else:  # standard siamese
        model = SiameseICAConvNet(
            num_channels=num_channels, 
            num_samples=num_samples, 
            config_path=config_path
        ).to(device)
        print(f"Created SiameseICAConvNet")
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model weights from {model_path}")
    print(f"Model was trained for {checkpoint['epoch']} epochs")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    
    # Create output directory
    model_type = "identity" if mode == 'identity' else "siamese"
    if use_contrastive:
        model_type = "contrastive"
    run_output_dir = os.path.join(output_dir, f"{model_type}_eval")
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Evaluate model
    print("\n=== Evaluating Model ===")
    if mode == 'identity':
        results = evaluate_identity_model(model, test_loader, device)
        print(f"\nResults:")
        print(f"  Overall Accuracy: {results['accuracy']:.2f}%")
        print(f"  Average EER: {results['avg_eer']:.4f}")
    else:  # siamese
        results = evaluate_siamese_model(model, test_loader, device, use_contrastive)
        print(f"\nResults:")
        print(f"  Accuracy: {results['accuracy']:.2f}%")
        print(f"  EER: {results['eer']:.4f}")
        print(f"  EER Threshold: {results['eer_threshold']:.4f}")
        print(f"  ROC AUC: {results['roc_auc']:.4f}")
    
    # Plot results if enabled
    if plot_results_enabled:
        print("\n=== Generating Plots ===")
        plot_results(results, run_output_dir, mode, use_contrastive)
    
    # Save numerical results
    np.savez(
        os.path.join(run_output_dir, 'evaluation_results.npz'),
        **results
    )
    
    print(f"\nEvaluation complete. Results saved to {run_output_dir}")


if __name__ == "__main__":
    main()