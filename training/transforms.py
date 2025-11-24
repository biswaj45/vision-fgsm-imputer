"""
Image transformations using Albumentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, Any
import cv2


def get_training_transforms(image_size: int = 256) -> A.Compose:
    """
    Get augmentation pipeline for training.
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations composition
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def get_validation_transforms(image_size: int = 256) -> A.Compose:
    """
    Get transformation pipeline for validation (no augmentation).
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations composition
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def get_inference_transforms(image_size: int = 256) -> A.Compose:
    """
    Get transformation pipeline for inference.
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations composition
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def denormalize(tensor: np.ndarray) -> np.ndarray:
    """
    Denormalize tensor back to [0, 255] range.
    
    Args:
        tensor: Normalized tensor (C, H, W)
    
    Returns:
        Denormalized array (H, W, C) in range [0, 255]
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    # Denormalize
    img = tensor * std + mean
    
    # Clip to valid range
    img = np.clip(img, 0, 1)
    
    # Convert to uint8
    img = (img * 255).astype(np.uint8)
    
    # Change from CHW to HWC
    img = np.transpose(img, (1, 2, 0))
    
    return img


def apply_perturbation(
    image: np.ndarray,
    perturbation: np.ndarray,
    epsilon: float = 0.02
) -> np.ndarray:
    """
    Apply perturbation to image.
    
    Args:
        image: Original image (H, W, C) in range [0, 255]
        perturbation: Perturbation map (H, W, C) in range [-1, 1]
        epsilon: Perturbation scale
    
    Returns:
        Perturbed image (H, W, C) in range [0, 255]
    """
    # Normalize image to [0, 1]
    img_normalized = image.astype(np.float32) / 255.0
    
    # Apply perturbation
    img_perturbed = img_normalized + epsilon * perturbation
    
    # Clip to valid range
    img_perturbed = np.clip(img_perturbed, 0, 1)
    
    # Convert back to uint8
    img_perturbed = (img_perturbed * 255).astype(np.uint8)
    
    return img_perturbed


if __name__ == "__main__":
    # Test transforms
    import cv2
    
    # Create dummy image
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Test training transforms
    train_transform = get_training_transforms()
    transformed = train_transform(image=dummy_img)
    print(f"Training transform output shape: {transformed['image'].shape}")
    
    # Test validation transforms
    val_transform = get_validation_transforms()
    transformed = val_transform(image=dummy_img)
    print(f"Validation transform output shape: {transformed['image'].shape}")
    
    # Test denormalization
    denorm_img = denormalize(transformed['image'].numpy())
    print(f"Denormalized image shape: {denorm_img.shape}")
    print(f"Denormalized image range: [{denorm_img.min()}, {denorm_img.max()}]")
