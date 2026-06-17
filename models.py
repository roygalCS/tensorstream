"""
models.py - Model loading and management

Handles loading pretrained models and preparing them for inference.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


class ModelManager:
    """
    Manages model loading, device placement, and inference.
    """
    
    def __init__(self, model_name: str, device: str, num_classes: int = 1000):
        """
        Initialize model manager.
        
        Args:
            model_name: Name of model ("resnet18", "resnet50", etc.)
            device: Device to load model on ("cuda" or "cpu")
            num_classes: Number of output classes
        """
        self.model_name = model_name
        self.device = device
        self.num_classes = num_classes
        
        # Load the model
        self.model = self._load_model()
        
        # Move to device and set to eval mode
        self.model = self.model.to(device)
        self.model.eval()  # Disable dropout, batch norm updates
        
        print(f"✓ Model '{model_name}' loaded on {device}")
    
    def _load_model(self) -> nn.Module:
        """
        Load pretrained model from torchvision.
        
        Returns:
            PyTorch model ready for inference
        """
        if self.model_name == "resnet18":
            # Load ResNet18 pretrained on ImageNet
            model = models.resnet18(pretrained=True)
        elif self.model_name == "resnet50":
            model = models.resnet50(pretrained=True)
        elif self.model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=True)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        return model
    
    def predict(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on a batch of images.
        
        Args:
            batch: Tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            predictions: Class indices (batch_size,)
            confidences: Confidence scores (batch_size,)
        """
        with torch.no_grad():  # Don't track gradients (faster)
            # Forward pass through model
            logits = self.model(batch)  # Shape: (batch_size, num_classes)
            
            # Get probabilities
            probs = torch.softmax(logits, dim=1)  # Shape: (batch_size, num_classes)
            
            # Get top predictions and confidences
            confidences, predictions = torch.max(probs, dim=1)  # Both shape: (batch_size,)
        
        return predictions, confidences
    
    def get_class_name(self, class_idx: int) -> str:
        """
        Get human-readable class name from ImageNet labels.
        
        Args:
            class_idx: Class index (0-999 for ImageNet)
        
        Returns:
            Class name (e.g., "golden retriever")
        """
        # ImageNet class labels (you could load from a file)
        # For now, just return the index
        # In production, you'd load from imagenet_classes.txt
        return f"Class {class_idx}"


def create_model(config) -> ModelManager:
    """
    Factory function to create model manager.
    
    Args:
        config: PipelineConfig object
    
    Returns:
        Initialized ModelManager
    """
    return ModelManager(
        model_name=config.model_name,
        device=config.device,
        num_classes=config.num_classes
    )