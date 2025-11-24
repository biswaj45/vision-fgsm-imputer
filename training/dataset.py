"""
Dataset loader for CelebA or VGGFace2.
Returns (clean_img, fgsm_target_img) pairs for training.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Optional, Callable, Tuple
from tqdm import tqdm


class FaceDataset(Dataset):
    """
    Face dataset for anti-deepfake training.
    Loads images from a folder structure and applies transformations.
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')
    ):
        """
        Args:
            root_dir: Root directory containing images
            transform: Albumentations transformation pipeline
            max_samples: Maximum number of samples to load (None = all)
            extensions: Valid image file extensions
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions
        
        # Find all image files
        self.image_paths = self._find_images(max_samples)
        
        print(f"Loaded {len(self.image_paths)} images from {root_dir}")
    
    def _find_images(self, max_samples: Optional[int]) -> list:
        """Find all image files in the directory."""
        image_paths = []
        
        for ext in self.extensions:
            image_paths.extend(self.root_dir.rglob(f"*{ext}"))
        
        # Sort for reproducibility
        image_paths = sorted(image_paths)
        
        # Limit samples if specified
        if max_samples:
            image_paths = image_paths[:max_samples]
        
        return image_paths
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and transform a single image.
        
        Returns:
            Transformed image tensor
        """
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Convert to tensor if no transform
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image


class FGSMDataset(Dataset):
    """
    Dataset that generates FGSM pairs on-the-fly during training.
    Returns (clean_img, fgsm_target_img) for each sample.
    """
    
    def __init__(
        self,
        root_dir: str,
        model: torch.nn.Module,
        transform: Optional[Callable] = None,
        epsilon: float = 0.02,
        max_samples: Optional[int] = None,
        device: str = 'cpu'
    ):
        """
        Args:
            root_dir: Root directory containing images
            model: Model to use for generating FGSM targets
            transform: Albumentations transformation pipeline
            epsilon: FGSM perturbation magnitude
            max_samples: Maximum number of samples
            device: Device for FGSM generation
        """
        self.base_dataset = FaceDataset(root_dir, transform, max_samples)
        self.model = model
        self.epsilon = epsilon
        self.device = device
        
        # Set model to eval for FGSM generation
        self.model.eval()
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get (clean_img, fgsm_target_img) pair.
        
        Returns:
            Tuple of (clean image tensor, FGSM target tensor)
        """
        clean_img = self.base_dataset[idx]
        
        # Generate FGSM target
        with torch.no_grad():
            clean_img_batch = clean_img.unsqueeze(0).to(self.device)
            
            # Simple FGSM: add random noise as target
            # (In real training, you might use a more sophisticated approach)
            noise = torch.randn_like(clean_img_batch) * self.epsilon
            fgsm_target = torch.clamp(clean_img_batch + noise, 0, 1)
            
            fgsm_target = fgsm_target.squeeze(0).cpu()
        
        return clean_img, fgsm_target


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        train_transform: Training transformations
        val_transform: Validation transformations
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_samples: Maximum samples per split
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = FaceDataset(
        root_dir=train_dir,
        transform=train_transform,
        max_samples=max_samples
    )
    
    val_dataset = FaceDataset(
        root_dir=val_dir,
        transform=val_transform,
        max_samples=max_samples
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    from transforms import get_training_transforms
    
    # Create dummy data directory for testing
    test_dir = "./test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Test transforms
    transform = get_training_transforms()
    
    # If test images exist, load them
    if os.path.exists(test_dir):
        dataset = FaceDataset(test_dir, transform=transform)
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample shape: {sample.shape}")
            print(f"Dataset size: {len(dataset)}")
