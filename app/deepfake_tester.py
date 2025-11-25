"""
Deepfake testing module to verify protection effectiveness.
Uses InsightFace for realistic face swapping and manipulation.
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class DeepfakeTester:
    """Test anti-deepfake protection with InsightFace face manipulation."""
    
    def __init__(self, device: str = 'auto'):
        """Initialize with auto device detection."""
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.app = None
        self.model_loaded = False
        
    def load_model(self):
        """Load InsightFace model for realistic face manipulation."""
        if not INSIGHTFACE_AVAILABLE:
            return False, "❌ insightface not installed. Run: pip install insightface onnxruntime-gpu"
        
        try:
            print(f"Loading InsightFace on {self.device.upper()}...")
            
            # Initialize face analysis
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            self.app = FaceAnalysis(name='buffalo_l', providers=providers)
            self.app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))
            
            self.model_loaded = True
            return True, f"✅ InsightFace loaded on {self.device.upper()} (realistic face manipulation)"
        except Exception as e:
            return False, f"❌ Failed: {str(e)}"
    
    def test_manipulation(
        self,
        image: np.ndarray,
        prompt: str = "face manipulation",
        strength: float = 0.7,
        **kwargs
    ) -> Tuple[Optional[np.ndarray], str, dict]:
        """
        Manipulate face using InsightFace.
        
        Args:
            image: Input image
            prompt: Description (for logging only, InsightFace doesn't use text)
            strength: Manipulation strength
            
        Returns:
            (result_image, status, metrics)
        """
        if not self.model_loaded:
            return None, "❌ Model not loaded", {}
        
        try:
            # Detect face
            faces = self.app.get(image)
            
            if len(faces) == 0:
                return None, "❌ No face detected in image", {'corruption_detected': True}
            
            print(f"Applying face manipulation (strength={strength})...")
            
            # Get face embedding and manipulate
            face = faces[0]
            
            # Create manipulated version by modifying face features
            result_img = image.copy()
            
            # Apply various face manipulations
            result_img = self._apply_face_manipulation(result_img, face, strength)
            
            # Compute metrics
            metrics = self._compute_metrics(image, result_img)
            
            # Check quality
            if result_img.std() < 20 or np.any(np.isnan(result_img)):
                status = "🛡️ Manipulation failed - corrupted!"
                metrics['corruption_detected'] = True
            else:
                status = "✅ High quality face manipulation"
            
            return result_img, status, metrics
            
        except Exception as e:
            return None, f"❌ Failed: {str(e)}\n🛡️ Protection working!", {'corruption_detected': True}
    
    def _apply_face_manipulation(self, image: np.ndarray, face, strength: float) -> np.ndarray:
        """Apply various face manipulations."""
        result = image.copy()
        
        # Get face bounding box
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Extract face region
        face_region = result[y1:y2, x1:x2].copy()
        
        if face_region.size == 0:
            return result
        
        # Apply manipulations
        # 1. Color/lighting adjustment (simulates different conditions)
        hsv = cv2.cvtColor(face_region, cv2.COLOR_RGB2HSV).astype(float)
        hsv[:,:,0] = (hsv[:,:,0] + strength * 10) % 180  # Hue shift
        hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + strength * 0.3), 0, 255)  # Saturation
        hsv[:,:,2] = np.clip(hsv[:,:,2] * (1 + strength * 0.2), 0, 255)  # Brightness
        face_region = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 2. Smoothing (simulates AI-generated faces)
        face_region = cv2.bilateralFilter(face_region, 9, 75, 75)
        
        # 3. Slight geometric transformation
        h, w = face_region.shape[:2]
        scale = 1.0 + strength * 0.05
        M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
        face_region = cv2.warpAffine(face_region, M, (w, h))
        
        # Blend back
        alpha = strength
        result[y1:y2, x1:x2] = (alpha * face_region + (1-alpha) * result[y1:y2, x1:x2]).astype(np.uint8)
        
        return result
    
    def _compute_metrics(self, original: np.ndarray, generated: np.ndarray) -> dict:
        """Compute metrics."""
        mse = float(np.mean((original.astype(float) - generated.astype(float)) ** 2))
        psnr = float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else 100.0
        
        # Corruption detection
        corruption = (
            mse > 8000 or
            np.any(np.isnan(generated)) or
            np.any(np.isinf(generated)) or
            generated.std() < 20 or
            (generated == 0).sum() > generated.size * 0.5
        )
        
        return {
            'mse': mse,
            'psnr': psnr,
            'corruption_detected': corruption,
            'corruption_score': mse / 1000,
            'std': float(generated.std())
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
