"""
Deepfake testing module to verify protection effectiveness.
Tests protected images against lightweight face manipulation models.
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple
import io
import base64

try:
    from diffusers import StableDiffusionImg2ImgPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class DeepfakeTester:
    """Test anti-deepfake protection by attempting face manipulation."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model = None
        self.model_loaded = False
        
    def load_model(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        """Load lightweight diffusion model for testing."""
        if not DIFFUSERS_AVAILABLE:
            return False, "diffusers library not available. Install with: pip install diffusers"
        
        try:
            print(f"Loading {model_name}...")
            self.model = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Enable memory optimizations
            if self.device == 'cpu':
                self.model.enable_attention_slicing()
            
            self.model_loaded = True
            return True, "Model loaded successfully"
        except Exception as e:
            return False, f"Failed to load model: {str(e)}"
    
    def test_manipulation(
        self,
        image: np.ndarray,
        prompt: str,
        strength: float = 0.75,
        guidance_scale: float = 7.5
    ) -> Tuple[Optional[np.ndarray], str, dict]:
        """
        Attempt to manipulate image with deepfake model.
        
        Args:
            image: Input image (protected or unprotected)
            prompt: Manipulation prompt
            strength: How much to transform (0-1)
            guidance_scale: Prompt adherence (1-20)
            
        Returns:
            (result_image, status_message, metrics)
        """
        if not self.model_loaded:
            return None, "Model not loaded. Click 'Load Model' first.", {}
        
        try:
            # Convert to PIL
            if image.dtype == np.uint8:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Resize to model input size
            pil_image = pil_image.resize((512, 512), Image.LANCZOS)
            
            # Generate
            print(f"Generating with prompt: {prompt}")
            result = self.model(
                prompt=prompt,
                image=pil_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=20  # Fast generation
            )
            
            # Convert back to numpy
            result_img = np.array(result.images[0])
            
            # Compute quality metrics
            metrics = self._compute_metrics(np.array(pil_image.resize((512, 512))), result_img)
            
            status = "✅ Generation successful - Protection may be insufficient"
            if metrics['mse'] > 5000:  # High error indicates failure
                status = "🛡️ Generation corrupted - Protection working!"
            
            return result_img, status, metrics
            
        except Exception as e:
            return None, f"❌ Generation failed: {str(e)}", {}
    
    def _compute_metrics(self, original: np.ndarray, generated: np.ndarray) -> dict:
        """Compute quality metrics to assess protection effectiveness."""
        # MSE
        mse = np.mean((original.astype(float) - generated.astype(float)) ** 2)
        
        # PSNR
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            psnr = 100
        
        # Structural similarity (simple version)
        ssim = self._simple_ssim(original, generated)
        
        return {
            'mse': float(mse),
            'psnr': float(psnr),
            'ssim': float(ssim),
            'corruption_score': float(mse / 1000)  # Higher = more corrupted
        }
    
    def _simple_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Simple SSIM approximation."""
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(float)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(float)
        
        mu1 = img1.mean()
        mu2 = img2.mean()
        
        sigma1 = img1.std()
        sigma2 = img2.std()
        
        covariance = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * covariance + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
        
        return float(np.clip(ssim, 0, 1))


def create_comparison_visualization(
    original: np.ndarray,
    protected: np.ndarray,
    deepfake_original: Optional[np.ndarray],
    deepfake_protected: Optional[np.ndarray]
) -> np.ndarray:
    """Create 2x2 comparison grid."""
    h, w = 512, 512
    
    # Resize all images
    orig_resized = cv2.resize(original, (w, h))
    prot_resized = cv2.resize(protected, (w, h))
    
    if deepfake_original is not None:
        df_orig = cv2.resize(deepfake_original, (w, h))
    else:
        df_orig = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(df_orig, "Not generated", (w//4, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if deepfake_protected is not None:
        df_prot = cv2.resize(deepfake_protected, (w, h))
    else:
        df_prot = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(df_prot, "Not generated", (w//4, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Create 2x2 grid
    top_row = np.hstack([orig_resized, prot_resized])
    bottom_row = np.hstack([df_orig, df_prot])
    grid = np.vstack([top_row, bottom_row])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(grid, "Original Input", (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(grid, "Protected Input", (w + 10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(grid, "Deepfake from Original", (10, h + 30), font, 0.8, (255, 100, 100), 2)
    cv2.putText(grid, "Deepfake from Protected", (w + 10, h + 30), font, 0.8, (100, 100, 255), 2)
    
    return grid


if __name__ == "__main__":
    # Test
    tester = DeepfakeTester(device='cpu')
    success, msg = tester.load_model()
    print(f"Model load: {success} - {msg}")
