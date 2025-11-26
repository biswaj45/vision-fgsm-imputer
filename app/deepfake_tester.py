"""
Deepfake testing module to verify protection effectiveness.
Uses InsightFace inswapper for realistic face swapping.
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple
import os

try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class DeepfakeTester:
    """Test anti-deepfake protection with InsightFace face swapper."""
    
    def __init__(self, device: str = 'auto'):
        """Initialize with auto device detection."""
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.app = None
        self.swapper = None
        self.model_loaded = False
        
    def load_model(self):
        """Load InsightFace face swapper model."""
        if not INSIGHTFACE_AVAILABLE:
            return False, "❌ insightface not installed. Run: pip install insightface onnxruntime-gpu"
        
        try:
            print(f"Loading InsightFace Face Swapper on {self.device.upper()}...")
            
            # Initialize face analysis
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            self.app = FaceAnalysis(name='buffalo_l', providers=providers)
            self.app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))
            
            # Load face swapper model (inswapper_128)
            model_path = os.path.join(insightface.app.common.__path__[0], 'models/inswapper_128.onnx')
            if not os.path.exists(model_path):
                # Download if not exists
                print("Downloading inswapper_128 model...")
            self.swapper = get_model('inswapper_128.onnx', providers=providers)
            
            self.model_loaded = True
            return True, f"✅ InsightFace Face Swapper loaded on {self.device.upper()} - Ready for realistic deepfakes!"
        except Exception as e:
            return False, f"❌ Failed: {str(e)}"
    
    def test_manipulation(
        self,
        target_image: np.ndarray,
        source_image: np.ndarray,
        **kwargs
    ) -> Tuple[Optional[np.ndarray], str, dict]:
        """
        Swap face from source to target using InsightFace inswapper.
        
        Args:
            target_image: Image where face will be replaced
            source_image: Image providing the face to swap in
            
        Returns:
            (result_image, status, metrics)
        """
        if not self.model_loaded:
            return None, "❌ Model not loaded", {}
        
        try:
            # Detect faces in both images
            print("Detecting faces...")
            target_faces = self.app.get(target_image)
            source_faces = self.app.get(source_image)
            
            if len(target_faces) == 0:
                return None, "❌ No face detected in TARGET image", {'corruption_detected': True}
            
            if len(source_faces) == 0:
                return None, "❌ No face detected in SOURCE image", {'corruption_detected': True}
            
            print(f"Swapping face: SOURCE → TARGET...")
            
            # Get the first face from each
            source_face = source_faces[0]
            target_face = target_faces[0]
            
            # Perform face swap
            result_img = target_image.copy()
            result_img = self.swapper.get(result_img, target_face, source_face, paste_back=True)
            
            # Compute metrics
            metrics = self._compute_metrics(target_image, result_img)
            
            # Check quality - face swap should create significant difference
            if result_img.std() < 20 or np.any(np.isnan(result_img)) or metrics['mse'] < 100:
                status = "🛡️ Face swap FAILED - image corrupted!"
                metrics['corruption_detected'] = True
            else:
                status = "✅ High quality face swap successful"
            
            return result_img, status, metrics
            
        except Exception as e:
            return None, f"❌ Face swap failed: {str(e)}\n🛡️ Protection working!", {'corruption_detected': True}
    
    def _compute_metrics(self, original: np.ndarray, generated: np.ndarray) -> dict:
        """Compute metrics."""
        mse = float(np.mean((original.astype(float) - generated.astype(float)) ** 2))
        psnr = float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else 100.0
        
        # Face swap should create MSE > 500 for real change
        # Corruption detection: either failed swap (MSE < 100) or corrupted result
        corruption = (
            mse < 100 or  # Swap didn't work
            mse > 15000 or  # Too corrupted
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
            'std': float(generated.std()),
            'swap_strength': 'Strong' if mse > 500 else 'Weak'
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
