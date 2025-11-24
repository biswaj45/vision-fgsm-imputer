"""
Perturbation generation utilities for inference.
"""

import torch
import numpy as np
from typing import Union


def generate_perturbation(
    model: torch.nn.Module,
    image: torch.Tensor,
    epsilon: float = 0.05,
    device: str = 'cpu',
    boost_strength: float = 1.5
) -> torch.Tensor:
    """
    Generate perturbation map using the trained model.
    
    Args:
        model: Trained U-Net or Autoencoder
        image: Input image tensor [B, C, H, W] or [C, H, W]
        epsilon: Perturbation scale
        device: Device to run on
        boost_strength: Multiplier for stronger protection (default 1.5)
    
    Returns:
        Perturbation tensor in range [-epsilon*boost_strength, epsilon*boost_strength]
    """
    model.eval()
    
    # Handle single image
    squeeze_output = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_output = True
    
    image = image.to(device)
    
    with torch.no_grad():
        # Generate raw perturbation (range: -1 to 1)
        perturbation = model(image)
        
        # Scale to epsilon range with boost
        perturbation = perturbation * epsilon * boost_strength
    
    if squeeze_output:
        perturbation = perturbation.squeeze(0)
    
    return perturbation


def apply_perturbation_tensor(
    image: torch.Tensor,
    perturbation: torch.Tensor
) -> torch.Tensor:
    """
    Apply perturbation to image tensor.
    
    Args:
        image: Original image tensor [B, C, H, W]
        perturbation: Perturbation tensor [B, C, H, W]
    
    Returns:
        Perturbed image tensor, clamped to [0, 1]
    """
    perturbed = image + perturbation
    perturbed = torch.clamp(perturbed, 0.0, 1.0)
    return perturbed


def adaptive_perturbation(
    model: torch.nn.Module,
    image: torch.Tensor,
    base_epsilon: float = 0.02,
    max_epsilon: float = 0.05,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate adaptive perturbation based on image content.
    Areas with more detail get stronger perturbations.
    
    Args:
        model: Trained model
        image: Input image tensor
        base_epsilon: Base perturbation magnitude
        max_epsilon: Maximum perturbation magnitude
        device: Device to run on
    
    Returns:
        Adaptive perturbation tensor
    """
    model.eval()
    
    with torch.no_grad():
        # Generate base perturbation
        perturbation = model(image.to(device))
        
        # Compute local variance as measure of detail
        # High variance = more detail = stronger perturbation allowed
        image_gray = image.mean(dim=1, keepdim=True)
        kernel_size = 5
        padding = kernel_size // 2
        
        # Simple local variance approximation
        mean_filter = torch.nn.AvgPool2d(kernel_size, stride=1, padding=padding)
        local_mean = mean_filter(image_gray)
        local_var = mean_filter(image_gray ** 2) - local_mean ** 2
        
        # Normalize variance to [0, 1]
        var_normalized = (local_var - local_var.min()) / (local_var.max() - local_var.min() + 1e-8)
        
        # Scale epsilon based on variance
        epsilon_map = base_epsilon + (max_epsilon - base_epsilon) * var_normalized
        
        # Apply adaptive scaling
        adaptive_pert = perturbation * epsilon_map
    
    return adaptive_pert


if __name__ == "__main__":
    # Test perturbation generation
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.unet_tiny import create_tiny_unet
    
    model = create_tiny_unet()
    model.eval()
    
    # Test image
    test_img = torch.randn(1, 3, 256, 256).clamp(0, 1)
    
    # Generate perturbation
    perturbation = generate_perturbation(model, test_img, epsilon=0.02)
    
    print(f"Input shape: {test_img.shape}")
    print(f"Perturbation shape: {perturbation.shape}")
    print(f"Perturbation range: [{perturbation.min():.6f}, {perturbation.max():.6f}]")
    
    # Apply perturbation
    perturbed_img = apply_perturbation_tensor(test_img, perturbation)
    print(f"Perturbed image range: [{perturbed_img.min():.6f}, {perturbed_img.max():.6f}]")
