#!/usr/bin/env python3
"""
CNN models for EEG-based personal identification

Implementation of ICAConvNet and SiameseICAConvNet based on:
Fan et al. (2021) - CNN-based personal identification system using resting state electroencephalography

Models are configured via the configs/config.yaml file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ICAConvNet(nn.Module):
    """
    Implementation of the ICAConvNet architecture from Fan et al. paper.
    
    Inputs EEG data and classifies the subject ID.
    """
    def __init__(self, num_channels=14, num_samples=80, num_classes=109, config_path='configs/config.yaml'):
        """
        Initialize the model with parameters from config file.
        
        Args:
            num_channels: Number of EEG channels
            num_samples: Number of time samples
            num_classes: Number of output classes (subjects)
            config_path: Path to YAML config file
        """
        super(ICAConvNet, self).__init__()
        
        # Load configuration
        config = load_config(config_path)
        model_config = config['model']
        
        # Extract parameters
        ica_comp = model_config['ica_components']
        conv_filters = model_config['conv_filters']
        kernel_sizes = model_config['kernel_sizes']
        fc_dim = model_config['fc_dim']
        dropout_rate = model_config['dropout_rate']
        
        # ICA stage - learnable linear transformation
        self.ica = nn.Linear(num_channels, ica_comp, bias=False)
        
        # Convolutional stages
        self.conv1 = nn.Conv2d(1, conv_filters[0], kernel_size=kernel_sizes[0], 
                               stride=(2, 1), padding=(kernel_sizes[0][0]//2, kernel_sizes[0][1]//2))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        
        self.conv2 = nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=kernel_sizes[1], 
                               stride=(1, 1), padding=(kernel_sizes[1][0]//2, kernel_sizes[1][1]//2))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        
        self.conv3 = nn.Conv2d(conv_filters[1], conv_filters[2], kernel_size=kernel_sizes[2], 
                               stride=(1, 1), padding=(kernel_sizes[2][0]//2, kernel_sizes[2][1]//2))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        
        # Calculate the size of the flattened features
        # After all conv and pooling layers
        samples_after = num_samples // 8  # After all stride and pooling operations
        width_after = ica_comp // 2  # After all width-direction pooling
        feature_size = conv_filters[2] * samples_after * width_after
        
        # Fully connected layers
        self.fc1 = nn.Linear(feature_size, fc_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape [batch_size, channels, samples]
            
        Returns:
            Class scores of shape [batch_size, num_classes]
        """
        # x shape: [batch_size, channels, samples]
        batch_size, channels, samples = x.size()
        
        # Apply ICA transform - we need to permute the dimensions
        x = x.permute(0, 2, 1)  # [batch_size, samples, channels]
        x = self.ica(x)  # [batch_size, samples, ica_components]
        x = x.permute(0, 2, 1)  # [batch_size, ica_components, samples]
        
        # Add channel dimension for 2D convolution
        x = x.unsqueeze(1)  # [batch_size, 1, ica_components, samples]
        
        # Apply convolutional features with ELU activation
        x = F.elu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.elu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.elu(self.conv3(x))
        x = self.pool3(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Log softmax for numerical stability
        return F.log_softmax(x, dim=1)
    
    def get_embedding(self, x):
        """
        Get the embedding (features before the final classification layer).
        Useful for feature extraction or siamese networks.
        
        Args:
            x: Input tensor of shape [batch_size, channels, samples]
            
        Returns:
            Embedding of shape [batch_size, fc_dim]
        """
        # x shape: [batch_size, channels, samples]
        batch_size, channels, samples = x.size()
        
        # Apply ICA transform
        x = x.permute(0, 2, 1)  # [batch_size, samples, channels]
        x = self.ica(x)  # [batch_size, samples, ica_components]
        x = x.permute(0, 2, 1)  # [batch_size, ica_components, samples]
        
        # Add channel dimension for 2D convolution
        x = x.unsqueeze(1)  # [batch_size, 1, ica_components, samples]
        
        # Apply convolutional features
        x = F.elu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.elu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.elu(self.conv3(x))
        x = self.pool3(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Get embedding from fc1
        x = F.elu(self.fc1(x))
        
        return x


class SiameseICAConvNet(nn.Module):
    """
    Siamese network using the ICAConvNet architecture.
    
    Takes two EEG segments and determines if they're from the same person.
    """
    def __init__(self, num_channels=14, num_samples=80, config_path='configs/config.yaml'):
        """
        Initialize the siamese model with parameters from config file.
        
        Args:
            num_channels: Number of EEG channels
            num_samples: Number of time samples
            config_path: Path to YAML config file
        """
        super(SiameseICAConvNet, self).__init__()
        
        # Load configuration
        config = load_config(config_path)
        model_config = config['model']
        siamese_config = model_config['siamese']
        
        # Extract parameters
        embedding_dim = siamese_config.get('embedding_dim', model_config['fc_dim'])
        hidden_dim = siamese_config.get('hidden_dim', embedding_dim // 2)
        dropout_rate = model_config['dropout_rate']
        
        # Base network for feature extraction
        self.base_network = ICAConvNet(
            num_channels=num_channels, 
            num_samples=num_samples, 
            num_classes=embedding_dim,
            config_path=config_path
        )
        
        # Layers for similarity prediction
        self.fc_out = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward_once(self, x):
        """Get embedding for one input."""
        return self.base_network.get_embedding(x)
    
    def forward(self, x1, x2):
        """
        Forward pass for the siamese network.
        
        Args:
            x1: First EEG segment [batch_size, channels, samples]
            x2: Second EEG segment [batch_size, channels, samples]
            
        Returns:
            Similarity score between 0 and 1
        """
        # Get embeddings for both inputs
        embedding1 = self.forward_once(x1)
        embedding2 = self.forward_once(x2)
        
        # Compute absolute difference between embeddings
        diff = torch.abs(embedding1 - embedding2)
        
        # Compute similarity score
        similarity = self.fc_out(diff)
        
        return similarity


class ImprovedICAConvNet(ICAConvNet):
    """Enhanced ICAConvNet with residual connections and additional non-linearities"""
    
    def __init__(self, num_channels=14, num_samples=80, num_classes=109, config_path='configs/config.yaml'):
        super(ImprovedICAConvNet, self).__init__(num_channels, num_samples, num_classes, config_path)
        
        # Add batch normalization layers
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)
        
        # Add skip connection for residual learning
        self.skip_conv = nn.Conv2d(1, self.conv3.out_channels, kernel_size=1, stride=(4, 2))
        
    def forward(self, x):
        batch_size, channels, samples = x.size()
        
        # Apply ICA transform
        x = x.permute(0, 2, 1)  # [batch_size, samples, channels]
        x = self.ica(x)  # [batch_size, samples, ica_components]
        x = x.permute(0, 2, 1)  # [batch_size, ica_components, samples]
        
        # Add channel dimension for 2D convolution
        x = x.unsqueeze(1)  # [batch_size, 1, ica_components, samples]
        
        # Save input for skip connection
        skip = x
        
        # Apply convolutional features with BN and ELU activation
        x = F.elu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.elu(self.bn3(self.conv3(x)))
        
        # Apply skip connection
        skip = self.skip_conv(skip)
        x = x + skip
        
        x = self.pool3(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

class ImprovedSiameseNet(nn.Module):
    """Enhanced Siamese network with attention mechanism"""
    
    def __init__(self, num_channels=14, num_samples=80, config_path='configs/config.yaml'):
        super(ImprovedSiameseNet, self).__init__()
        
        config = load_config(config_path)
        model_config = config['model']
        siamese_config = model_config['siamese']
        
        embedding_dim = siamese_config.get('embedding_dim', model_config['fc_dim'])
        hidden_dim = siamese_config.get('hidden_dim', embedding_dim // 2)
        dropout_rate = model_config['dropout_rate']
        
        # Use improved base network
        self.base_network = ImprovedICAConvNet(
            num_channels=num_channels, 
            num_samples=num_samples, 
            num_classes=embedding_dim,
            config_path=config_path
        )
        
        # Self-attention for embedding refinement
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.Tanh(),
            nn.Linear(embedding_dim // 4, 1)
        )
        
        # Similarity network with more layers
        self.fc_out = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward_once(self, x):
        """Get embedding with attention"""
        embed = self.base_network.get_embedding(x)
        
        # Apply self-attention for feature refinement
        att_weights = self.attention(embed)
        att_weights = F.softmax(att_weights, dim=1)
        attended_embed = embed * att_weights
        
        return attended_embed
    
    def forward(self, x1, x2):
        embedding1 = self.forward_once(x1)
        embedding2 = self.forward_once(x2)
        
        # Compute absolute difference between embeddings
        diff = torch.abs(embedding1 - embedding2)
        
        # Compute similarity score
        similarity = self.fc_out(diff)
        
        return similarity



class ContrastiveSiameseNet(nn.Module):
    """
    Alternative Siamese network using contrastive loss.
    
    Instead of predicting similarity directly, this model learns
    a distance metric between EEG segments.
    """
    def __init__(self, num_channels=14, num_samples=80, config_path='configs/config.yaml'):
        """
        Initialize the contrastive siamese model with parameters from config file.
        
        Args:
            num_channels: Number of EEG channels
            num_samples: Number of time samples
            config_path: Path to YAML config file
        """
        super(ContrastiveSiameseNet, self).__init__()
        
        # Load configuration
        config = load_config(config_path)
        model_config = config['model']
        siamese_config = model_config['siamese']
        
        # Extract parameters
        embedding_dim = siamese_config.get('embedding_dim', model_config['fc_dim'])
        self.margin = siamese_config.get('margin', 1.0)
        
        # Base network for feature extraction
        self.base_network = ICAConvNet(
            num_channels=num_channels, 
            num_samples=num_samples, 
            num_classes=embedding_dim,
            config_path=config_path
        )
    
    def forward_once(self, x):
        """Get embedding for one input."""
        return self.base_network.get_embedding(x)
    
    def forward(self, x1, x2):
        """
        Forward pass for the contrastive siamese network.
        
        Args:
            x1: First EEG segment [batch_size, channels, samples]
            x2: Second EEG segment [batch_size, channels, samples]
            
        Returns:
            Embeddings and Euclidean distance between them
        """
        # Get embeddings for both inputs
        embedding1 = self.forward_once(x1)
        embedding2 = self.forward_once(x2)
        
        # Compute Euclidean distance
        distance = F.pairwise_distance(embedding1, embedding2)
        
        return embedding1, embedding2, distance



def contrastive_loss(distance, label, margin=1.0):
    """
    Contrastive loss function for siamese networks.
    
    Args:
        distance: Euclidean distance between embeddings
        label: 1 if same subject, 0 if different
        margin: Margin for negative pairs
        
    Returns:
        Loss value
    """
    # For similar pairs (label=1), minimize distance
    # For dissimilar pairs (label=0), ensure distance > margin
    loss = label * torch.pow(distance, 2) + \
           (1 - label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    
    return loss.mean()


def create_model(config_path='configs/config.yaml', num_channels=None, num_samples=None, num_classes=None):
    """
    Create a model based on configuration.
    """
    config = load_config(config_path)
    mode = config['dataloader']['mode']
    use_contrastive = config['dataloader']['siamese'].get('use_contrastive', False)
    improved = config['model'].get('improved', {}).get('enabled', False)

    if mode == 'identity':
        if num_classes is None:
            raise ValueError("num_classes must be specified for identity model")
        if improved:
            return ImprovedICAConvNet(num_channels, num_samples, num_classes, config_path)
        else:
            return ICAConvNet(num_channels, num_samples, num_classes, config_path)
    elif mode == 'siamese':
        if use_contrastive:
            return ContrastiveSiameseNet(num_channels, num_samples, config_path)
        elif improved:
            return ImprovedSiameseNet(num_channels, num_samples, config_path)
        else:
            return SiameseICAConvNet(num_channels, num_samples, config_path)
    else:
        raise ValueError(f"Unknown model type: {mode}")


if __name__ == "__main__":
    """Test the models with parameters from config."""
    import numpy as np
    
    # Load configuration
    config = load_config()
    
    # Parameters
    batch_size = config['dataloader']['batch_size']
    ica_components = config['model']['ica_components']
    
    # Assuming EPOC X (14 channels) and 0.5s at 160Hz (80 samples)
    num_channels = 14
    num_samples = 80
    num_classes = 109  # Number of subjects
    
    # Print configuration
    print(f"=== Model Configuration ===")
    print(f"Mode: {config['dataloader']['mode']}")
    print(f"ICA Components: {ica_components}")
    print(f"Conv Filters: {config['model']['conv_filters']}")
    print(f"Kernel Sizes: {config['model']['kernel_sizes']}")
    print(f"FC Dimension: {config['model']['fc_dim']}")
    print(f"Dropout Rate: {config['model']['dropout_rate']}")
    
    # Create random input data
    x = torch.randn(batch_size, num_channels, num_samples)
    
    # Test identity model
    print(f"\n=== Testing Identity Model ===")
    identity_model = ICAConvNet(num_channels, num_samples, num_classes)
    output = identity_model(x)
    print(f"Output shape: {output.shape}")
    
    # Test siamese model
    print(f"\n=== Testing Siamese Model ===")
    siamese_model = SiameseICAConvNet(num_channels, num_samples)
    x1 = torch.randn(batch_size, num_channels, num_samples)
    x2 = torch.randn(batch_size, num_channels, num_samples)
    similarity = siamese_model(x1, x2)
    print(f"Similarity output shape: {similarity.shape}")
    
    # Test contrastive model
    print(f"\n=== Testing Contrastive Model ===")
    contrastive_model = ContrastiveSiameseNet(num_channels, num_samples)
    emb1, emb2, distance = contrastive_model(x1, x2)
    print(f"Embedding shape: {emb1.shape}")
    print(f"Distance shape: {distance.shape}")
    
    # Test factory function
    print(f"\n=== Testing Model Factory ===")
    # Temporarily set mode to identity
    config['dataloader']['mode'] = 'identity'
    with open('configs/config.yaml', 'w') as f:
        yaml.dump(config, f)
    model = create_model(num_channels=num_channels, num_samples=num_samples, num_classes=num_classes)
    print(f"Created model type: {type(model).__name__}")
    
    # Temporarily set mode to siamese with contrastive
    config['dataloader']['mode'] = 'siamese'
    config['dataloader']['siamese']['use_contrastive'] = True
    with open('configs/config.yaml', 'w') as f:
        yaml.dump(config, f)
    model = create_model(num_channels=num_channels, num_samples=num_samples)
    print(f"Created model type: {type(model).__name__}")
    
    # Restore original mode
    config['dataloader']['mode'] = 'identity'
    config['dataloader']['siamese']['use_contrastive'] = False
    with open('configs/config.yaml', 'w') as f:
        yaml.dump(config, f)