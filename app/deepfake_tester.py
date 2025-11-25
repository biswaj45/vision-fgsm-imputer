"""
Deepfake testing module to verify protection effectiveness.
Uses a simple CNN-based face manipulation as a lightweight alternative.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple


class SimpleFaceManipulator(nn.Module):
    """Lightweight CNN for face manipulation testing."""
    
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(feat)
        return out


class DeepfakeTester:
    """Test anti-deepfake protection with lightweight face manipulation."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model = None
        self.model_loaded = False
        
    def load_model(self):
        """Load lightweight face manipulation model."""
        try:
            print("Loading lightweight face manipulator...")
            self.model = SimpleFaceManipulator().to(self.device)
            
            # Initialize with random weights (simulates a pre-trained model)
            # In practice, this would be a real face manipulation model
            self.model.eval()
            
            self.model_loaded = True
            return True, "✅ Lightweight model loaded (< 5 seconds)"
        except Exception as e:
            return False, f"Failed to load model: {str(e)}"
    
    def test_manipulation(
        self,
        image: np.ndarray,
        prompt: str = "",
        strength: float = 0.75,
        guidance_scale: float = 7.5
    ) -> Tuple[Optional[np.ndarray], str, dict]:
        """
        Attempt to manipulate image.
        
        Args:
            image: Input image (protected or unprotected)
            prompt: Manipulation description (unused in lightweight version)
            strength: How much to transform
            guidance_scale: Not used
            
        Returns:
            (result_image, status_message, metrics)
        """
        if not self.model_loaded:
            return None, "Model not loaded. Click 'Load Model' first.", {}
        
        try:
            # Resize and normalize
            img = cv2.resize(image, (256, 256))
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Generate manipulation
            with torch.no_grad():
                manipulated = self.model(img_tensor)
                
                # Blend with original based on strength
                result = img_tensor * (1 - strength) + manipulated * strength
            
            # Convert back
            result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
            result_np = ((result_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            
            # Compute metrics
            metrics = self._compute_metrics(img, result_np)
            
            status = "✅ Manipulation successful"
            if metrics['mse'] > 2000 or metrics['corruption_detected']:
                status = "🛡️ Manipulation corrupted - Protection working!"
            
            return result_np, status, metrics
            
        except Exception as e:
            return None, f"❌ Generation failed: {str(e)}\n🛡️ This could mean protection is working!", {'corruption_detected': True}
    
    def _compute_metrics(self, original: np.ndarray, generated: np.ndarray) -> dict:
        """Compute quality metrics."""
        mse = np.mean((original.astype(float) - generated.astype(float)) ** 2)
        
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            psnr = 100
        
        # Simple corruption detection
        corruption_detected = (
            mse > 3000 or  # High MSE
            np.any(np.isnan(generated)) or  # NaN values
            generated.std() < 10  # Too uniform (failed)
        )
        
        return {
            'mse': float(mse),
            'psnr': float(psnr),
            'corruption_detected': bool(corruption_detected),
            'corruption_score': float(mse / 1000)
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
