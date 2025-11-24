"""
FGSM (Fast Gradient Sign Method) perturbation utilities.
Generate adversarial noise for training and inference.
"""

import torch
import torch.nn as nn
from typing import Callable, Optional


def fgsm(
    x: torch.Tensor,
    model: nn.Module,
    epsilon: float = 0.02,
    loss_fn: Optional[Callable] = None,
    target: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Generate FGSM adversarial perturbation.
    
    Args:
        x: Input image tensor [B, C, H, W] with values in [0, 1]
        model: Model to attack (typically a classifier or the U-Net itself)
        epsilon: Maximum perturbation magnitude
        loss_fn: Loss function to maximize (default: MSE with target)
        target: Target tensor for targeted attack (default: x itself)
    
    Returns:
        Perturbed image tensor [B, C, H, W]
    """
    # Set model to eval mode
    was_training = model.training
    model.eval()
    
    # For U-Net that outputs perturbations, we want adversarial examples
    # that make the model produce different perturbations
    if target is None:
        # Generate random target perturbations to maximize confusion
        target = torch.randn_like(x) * 0.1
    
    # Default loss function
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    
    # Ensure x requires grad
    x_adv = x.clone().detach().requires_grad_(True)
    
    # Forward pass - model outputs perturbation, not reconstruction
    with torch.enable_grad():
        output = model(x_adv)
        
        # Compute loss
        loss = loss_fn(output, target)
        
        # Backward pass to get gradients
        loss.backward()
    
    # Generate perturbation using sign of gradients
    with torch.no_grad():
        # Get gradient sign
        grad_sign = x_adv.grad.sign()
        
        # Apply FGSM perturbation
        x_perturbed = x_adv + epsilon * grad_sign
        
        # Clamp to valid range [0, 1]
        x_perturbed = torch.clamp(x_perturbed, 0.0, 1.0)
    
    # Restore model training state
    if was_training:
        model.train()
    
    return x_perturbed


def fgsm_targeted(
    x: torch.Tensor,
    model: nn.Module,
    target: torch.Tensor,
    epsilon: float = 0.02,
    alpha: float = 0.01,
    iterations: int = 5
) -> torch.Tensor:
    """
    Iterative FGSM for stronger perturbations.
    
    Args:
        x: Input image tensor [B, C, H, W]
        model: Model to attack
        target: Target output to achieve
        epsilon: Total maximum perturbation
        alpha: Step size per iteration
        iterations: Number of iterations
    
    Returns:
        Perturbed image tensor
    """
    was_training = model.training
    model.eval()
    
    x_adv = x.clone().detach()
    loss_fn = nn.MSELoss()
    
    for i in range(iterations):
        x_adv.requires_grad = True
        
        output = model(x_adv)
        loss = loss_fn(output, target)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + alpha * grad_sign
            
            # Project back to epsilon ball around original x
            perturbation = torch.clamp(x_adv - x, -epsilon, epsilon)
            x_adv = torch.clamp(x + perturbation, 0.0, 1.0)
        
        x_adv = x_adv.detach()
    
    if was_training:
        model.train()
    
    return x_adv


def generate_fgsm_training_pair(
    clean_img: torch.Tensor,
    model: nn.Module,
    epsilon: float = 0.02
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate (clean, fgsm_perturbed) pair for training.
    
    Args:
        clean_img: Clean input image [B, C, H, W]
        model: Model to use for generating perturbation
        epsilon: Perturbation magnitude
    
    Returns:
        Tuple of (clean_img, fgsm_target_img)
    """
    with torch.no_grad():
        fgsm_target = fgsm(clean_img, model, epsilon=epsilon)
    
    return clean_img, fgsm_target


if __name__ == "__main__":
    # Test FGSM
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.unet_tiny import create_tiny_unet
    
    model = create_tiny_unet()
    model.eval()
    
    # Test image
    x = torch.randn(2, 3, 256, 256).clamp(0, 1)
    
    # Generate FGSM perturbation
    x_perturbed = fgsm(x, model, epsilon=0.02)
    
    print(f"Original range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Perturbed range: [{x_perturbed.min():.3f}, {x_perturbed.max():.3f}]")
    print(f"Average perturbation: {(x_perturbed - x).abs().mean():.6f}")
