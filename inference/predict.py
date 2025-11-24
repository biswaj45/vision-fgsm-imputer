"""
Main prediction API for inference.
Simple interface: impute_noise("input.jpg", model_path="model.pth")
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional
import time

from models.unet_tiny import create_tiny_unet
from models.autoencoder import create_autoencoder
from perturb import generate_perturbation, apply_perturbation_tensor
from postprocess import image_to_tensor, tensor_to_image, denormalize_image


class NoiseImputer:
    """
    Fast CPU inference for anti-deepfake protection.
    Target: <300ms per image.
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'unet',
        epsilon: float = 0.02,
        device: str = 'auto'
    ):
        """
        Initialize noise imputer.
        
        Args:
            model_path: Path to trained model weights (.pth)
            model_type: Model type ('unet' or 'autoencoder')
            epsilon: Perturbation magnitude
            device: Device to run on ('auto', 'cuda', 'cpu'). 'auto' uses GPU if available.
        """
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.epsilon = epsilon
        self.model_type = model_type
        
        # Load model
        print(f"Loading model from {model_path}...")
        start_time = time.time()
        
        if model_type == 'unet':
            self.model = create_tiny_unet(pretrained_path=model_path)
        else:
            self.model = create_autoencoder(pretrained_path=model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Enable CUDA optimizations if available
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time*1000:.2f}ms on {self.device.upper()}")
        
        # Warmup
        self._warmup()
    
    def _warmup(self):
        """Warmup model with dummy input."""
        dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)
    
    def impute_from_array(
        self,
        image: np.ndarray,
        return_perturbation: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Generate perturbed image from numpy array.
        
        Args:
            image: Input image (H, W, C) in range [0, 255], RGB format
            return_perturbation: If True, also return perturbation map
        
        Returns:
            Perturbed image (H, W, C) in range [0, 255]
            If return_perturbation=True, returns (perturbed_img, perturbation_map)
        """
        original_size = image.shape[:2]
        
        # Resize to model input size
        image_resized = cv2.resize(image, (256, 256))
        
        # Convert to tensor
        image_tensor = image_to_tensor(image_resized, normalize=True)
        image_tensor = image_tensor.to(self.device)
        
        # Generate perturbation
        with torch.no_grad():
            perturbation = generate_perturbation(
                self.model,
                image_tensor,
                epsilon=self.epsilon,
                device=self.device
            )
            
            # Apply perturbation
            # Note: input is normalized, need to denormalize first
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            
            image_unnorm = image_tensor * std + mean
            perturbed_tensor = apply_perturbation_tensor(image_unnorm, perturbation)
        
        # Convert back to numpy
        perturbed_img = tensor_to_image(perturbed_tensor)
        
        # Resize back to original size
        if original_size != (256, 256):
            perturbed_img = cv2.resize(perturbed_img, (original_size[1], original_size[0]))
        
        if return_perturbation:
            pert_map = tensor_to_image(perturbation * 10)  # Scale for visualization
            if original_size != (256, 256):
                pert_map = cv2.resize(pert_map, (original_size[1], original_size[0]))
            return perturbed_img, pert_map
        
        return perturbed_img
    
    def impute_from_path(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        return_perturbation: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Generate perturbed image from file path.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)
            return_perturbation: If True, also return perturbation map
        
        Returns:
            Perturbed image (H, W, C) in range [0, 255]
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate perturbed image
        result = self.impute_from_array(image, return_perturbation=return_perturbation)
        
        # Save if output path provided
        if output_path:
            if return_perturbation:
                perturbed_img, _ = result
            else:
                perturbed_img = result
            
            # Convert RGB to BGR for saving
            perturbed_bgr = cv2.cvtColor(perturbed_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, perturbed_bgr)
            print(f"Saved perturbed image to {output_path}")
        
        return result
    
    def benchmark(self, num_iterations: int = 10) -> dict:
        """
        Benchmark inference speed.
        
        Args:
            num_iterations: Number of iterations for benchmarking
        
        Returns:
            Dictionary with timing statistics
        """
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = self.impute_from_array(dummy_image)
            elapsed = time.time() - start
            times.append(elapsed)
        
        stats = {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }
        
        return stats


def impute_noise(
    image_path: str,
    model_path: str = "outputs/checkpoints/best.pth",
    output_path: Optional[str] = None,
    epsilon: float = 0.02,
    model_type: str = 'unet',
    device: str = 'auto'
) -> np.ndarray:
    """
    Simple API for noise imputation.
    
    Args:
        image_path: Path to input image
        model_path: Path to trained model
        output_path: Path to save output (optional)
        epsilon: Perturbation magnitude
        model_type: Model type ('unet' or 'autoencoder')
        device: Device to use ('auto', 'cuda', 'cpu')
    
    Returns:
        Perturbed image as numpy array
    """
    imputer = NoiseImputer(
        model_path=model_path,
        model_type=model_type,
        epsilon=epsilon,
        device=device
    )
    
    result = imputer.impute_from_path(image_path, output_path)
    return result


if __name__ == "__main__":
    # Test inference
    import sys
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "perturbed_output.jpg"
        
        try:
            result = impute_noise(input_path, output_path=output_path)
            print(f"Successfully processed {input_path}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python predict.py <input_image> [output_image]")
        print("\nRunning benchmark with dummy model...")
        
        # Create dummy model for testing
        from models.unet_tiny import create_tiny_unet
        model = create_tiny_unet()
        torch.save(model.state_dict(), "dummy_model.pth")
        
        imputer = NoiseImputer("dummy_model.pth", model_type='unet')
        stats = imputer.benchmark(num_iterations=10)
        
        print("\nBenchmark Results:")
        print(f"Mean: {stats['mean_ms']:.2f}ms")
        print(f"Std: {stats['std_ms']:.2f}ms")
        print(f"Min: {stats['min_ms']:.2f}ms")
        print(f"Max: {stats['max_ms']:.2f}ms")
        print(f"FPS: {stats['fps']:.2f}")
