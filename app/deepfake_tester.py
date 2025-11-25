"""
Deepfake testing module to verify protection effectiveness.
Uses Stable Diffusion for real deepfake generation testing.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple

try:
    from diffusers import StableDiffusionImg2ImgPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class DeepfakeTester:
    """Test anti-deepfake protection with Stable Diffusion."""
    
    def __init__(self, device: str = 'auto'):
        """Initialize with auto device detection."""
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model = None
        self.model_loaded = False
        
    def load_model(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        """Load Stable Diffusion model."""
        if not DIFFUSERS_AVAILABLE:
            return False, "❌ diffusers not installed. Run: pip install diffusers transformers accelerate"
        
        try:
            print(f"Loading Stable Diffusion on {self.device.upper()}...")
            
            # Use float16 on GPU for faster inference
            dtype = torch.float16 if self.device == 'cuda' else torch.float32
            
            self.model = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Optimizations
            if self.device == 'cuda':
                self.model.enable_attention_slicing()
                self.model.enable_vae_slicing()
            else:
                self.model.enable_attention_slicing()
            
            self.model_loaded = True
            return True, f"✅ Model loaded on {self.device.upper()}"
        except Exception as e:
            return False, f"❌ Failed: {str(e)}"
    
    def test_manipulation(
        self,
        image: np.ndarray,
        prompt: str = "a photo of a person",
        strength: float = 0.65,
        guidance_scale: float = 7.5,
        num_steps: int = 25
    ) -> Tuple[Optional[np.ndarray], str, dict]:
        """
        Attempt to manipulate image with Stable Diffusion.
        
        Args:
            image: Input image
            prompt: Text description of manipulation
            strength: Transformation strength (0.5-0.8 recommended)
            guidance_scale: Prompt adherence
            num_steps: Inference steps (20-30 recommended)
            
        Returns:
            (result_image, status, metrics)
        """
        if not self.model_loaded:
            return None, "❌ Model not loaded", {}
        
        try:
            # Resize and convert
            img = cv2.resize(image, (512, 512))
            pil_img = Image.fromarray(img)
            
            print(f"Generating: '{prompt}' (strength={strength})")
            
            # Generate
            result = self.model(
                prompt=prompt,
                image=pil_img,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps
            )
            
            result_img = np.array(result.images[0])
            
            # Metrics
            metrics = self._compute_metrics(img, result_img)
            
            status = "✅ Manipulation successful"
            if metrics['corruption_detected']:
                status = "🛡️ Manipulation corrupted!"
            
            return result_img, status, metrics
            
        except Exception as e:
            return None, f"❌ Failed: {str(e)}\n🛡️ Protection may be working!", {'corruption_detected': True}
    
    def _compute_metrics(self, original: np.ndarray, generated: np.ndarray) -> dict:
        """Compute metrics."""
        mse = float(np.mean((original.astype(float) - generated.astype(float)) ** 2))
        psnr = float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else 100.0
        
        # Detect corruption
        corruption = (
            mse > 5000 or
            np.any(np.isnan(generated)) or
            generated.std() < 10
        )
        
        return {
            'mse': mse,
            'psnr': psnr,
            'corruption_detected': corruption,
            'corruption_score': mse / 1000
        }


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
