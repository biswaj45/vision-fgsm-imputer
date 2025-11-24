"""
Post-processing utilities for inference outputs.
"""

import torch
import numpy as np
import cv2
from typing import Tuple


def denormalize_image(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Denormalize tensor back to [0, 255] range.
    
    Args:
        tensor: Normalized tensor (C, H, W) or (B, C, H, W)
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Denormalized numpy array (H, W, C) in range [0, 255]
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Convert to numpy
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    # Denormalize
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    
    img = tensor * std + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    # CHW to HWC
    img = np.transpose(img, (1, 2, 0))
    
    return img


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy image.
    
    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W) in range [0, 1]
    
    Returns:
        Numpy array (H, W, C) in range [0, 255]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    # Scale to [0, 255]
    img = np.clip(tensor, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    # CHW to HWC
    img = np.transpose(img, (1, 2, 0))
    
    return img


def image_to_tensor(
    image: np.ndarray,
    normalize: bool = True
) -> torch.Tensor:
    """
    Convert numpy image to tensor.
    
    Args:
        image: Numpy array (H, W, C) in range [0, 255]
        normalize: Whether to normalize using ImageNet stats
    
    Returns:
        Tensor (1, C, H, W)
    """
    # Normalize to [0, 1]
    img = image.astype(np.float32) / 255.0
    
    # HWC to CHW
    img = np.transpose(img, (2, 0, 1))
    
    # Convert to tensor
    tensor = torch.from_numpy(img).unsqueeze(0)
    
    # Apply ImageNet normalization
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    
    return tensor


def blend_images(
    original: np.ndarray,
    perturbed: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Blend original and perturbed images for visualization.
    
    Args:
        original: Original image (H, W, C)
        perturbed: Perturbed image (H, W, C)
        alpha: Blending factor (0 = original, 1 = perturbed)
    
    Returns:
        Blended image (H, W, C)
    """
    blended = cv2.addWeighted(original, 1 - alpha, perturbed, alpha, 0)
    return blended


def create_comparison_grid(
    original: np.ndarray,
    perturbed: np.ndarray,
    difference: np.ndarray = None,
    scale_diff: float = 10.0
) -> np.ndarray:
    """
    Create side-by-side comparison grid.
    
    Args:
        original: Original image (H, W, C)
        perturbed: Perturbed image (H, W, C)
        difference: Difference map (H, W, C), if None will be computed
        scale_diff: Scale factor for difference visualization
    
    Returns:
        Comparison grid (H, W*2 or W*3, C)
    """
    if difference is None:
        # Compute difference
        diff = (perturbed.astype(np.float32) - original.astype(np.float32))
        diff = np.abs(diff) * scale_diff
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        difference = diff
    
    # Stack horizontally
    if difference is not None:
        grid = np.hstack([original, perturbed, difference])
    else:
        grid = np.hstack([original, perturbed])
    
    return grid


def resize_keep_aspect(
    image: np.ndarray,
    target_size: int = 256,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image keeping aspect ratio.
    
    Args:
        image: Input image (H, W, C)
        target_size: Target size for the smaller dimension
        interpolation: OpenCV interpolation method
    
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if h < w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized


def add_watermark(
    image: np.ndarray,
    text: str = "Protected",
    position: Tuple[int, int] = None,
    font_scale: float = 0.5,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1
) -> np.ndarray:
    """
    Add text watermark to image.
    
    Args:
        image: Input image (H, W, C)
        text: Watermark text
        position: Text position (x, y), if None uses bottom-right
        font_scale: Font scale
        color: Text color (B, G, R)
        thickness: Text thickness
    
    Returns:
        Image with watermark
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    if position is None:
        # Default to bottom-right
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        position = (w - text_size[0] - 10, h - 10)
    
    cv2.putText(
        img,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )
    
    return img


if __name__ == "__main__":
    # Test post-processing functions
    
    # Create test tensors
    test_tensor = torch.randn(1, 3, 256, 256).clamp(0, 1)
    
    # Test tensor to image
    img = tensor_to_image(test_tensor)
    print(f"Converted image shape: {img.shape}")
    print(f"Image range: [{img.min()}, {img.max()}]")
    
    # Test image to tensor
    tensor = image_to_tensor(img, normalize=False)
    print(f"Converted tensor shape: {tensor.shape}")
    print(f"Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")
