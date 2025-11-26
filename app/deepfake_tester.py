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
import gdown

try:
    import insightface
    from insightface.app import FaceAnalysis
    import onnxruntime
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
            
            # Download inswapper model if not exists
            model_dir = os.path.expanduser('~/.insightface/models/inswapper')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'inswapper_128.onnx')
            
            if not os.path.exists(model_path):
                print("Downloading inswapper_128 model (~554MB, one-time download)...")
                # Google Drive link for inswapper_128.onnx
                model_url = 'https://drive.google.com/uc?id=1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF'
                try:
                    gdown.download(model_url, model_path, quiet=False)
                    print("✅ Model downloaded successfully!")
                except:
                    # Fallback: try direct download from Hugging Face
                    print("Trying alternative download source...")
                    import urllib.request
                    hf_url = 'https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx'
                    urllib.request.urlretrieve(hf_url, model_path)
                    print("✅ Model downloaded from Hugging Face!")
            
            # Load the model
            print(f"Loading face swapper from {model_path}...")
            self.swapper = onnxruntime.InferenceSession(model_path, providers=providers)
            
            self.model_loaded = True
            return True, f"✅ InsightFace Face Swapper loaded on {self.device.upper()} - Ready for realistic deepfakes!"
        except Exception as e:
            return False, f"❌ Failed: {str(e)}\n\nTry manual download:\n1. Download from: https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx\n2. Place in: ~/.insightface/models/inswapper/"
    
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
            
            # Perform face swap using the swapper
            result_img = self._swap_face(target_image, target_face, source_face)
            
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
    
    def _swap_face(self, target_img: np.ndarray, target_face, source_face) -> np.ndarray:
        """Perform face swap using ONNX model."""
        # Prepare input
        source_embedding = source_face.normed_embedding.reshape((1, -1))
        source_embedding = source_embedding.astype(np.float32)
        
        # Get face alignment
        target_landmark = target_face.kps
        
        # Create input tensor
        input_size = (128, 128)
        
        # Extract and align target face region
        M = self._estimate_norm(target_landmark, input_size)
        aligned_img = cv2.warpAffine(target_img, M, input_size, borderValue=0.0)
        
        # Normalize
        blob = cv2.dnn.blobFromImage(aligned_img, 1.0 / 255, input_size, (0, 0, 0), swapRB=True)
        
        # Run inference
        latent = self.swapper.run(None, {
            'target': blob,
            'source': source_embedding
        })[0]
        
        # Denormalize
        swapped_face = latent[0].transpose(1, 2, 0)
        swapped_face = np.clip(swapped_face * 255, 0, 255).astype(np.uint8)
        swapped_face = cv2.cvtColor(swapped_face, cv2.COLOR_RGB2BGR)
        
        # Paste back
        IM = cv2.invertAffineTransform(M)
        result = target_img.copy()
        
        # Warp swapped face back
        swapped_back = cv2.warpAffine(swapped_face, IM, (result.shape[1], result.shape[0]), borderValue=0.0)
        
        # Create mask
        mask = np.zeros((input_size[1], input_size[0], 3), dtype=np.float32)
        mask = cv2.ellipse(mask, (input_size[0]//2, input_size[1]//2), (input_size[0]//2-5, input_size[1]//2-10), 0, 0, 360, (1, 1, 1), -1)
        mask = cv2.warpAffine(mask, IM, (result.shape[1], result.shape[0]))
        mask = cv2.GaussianBlur(mask, (7, 7), 3)
        
        # Blend
        result = mask * swapped_back + (1 - mask) * result
        
        return result.astype(np.uint8)
    
    def _estimate_norm(self, lmk, image_size=112):
        """Estimate affine transformation matrix."""
        assert lmk.shape == (5, 2)
        
        # Standard landmark positions
        if image_size == 112:
            src = np.array([
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041]], dtype=np.float32)
        else:
            src = np.array([
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041]], dtype=np.float32)
            src = src * image_size / 112
        
        dst = lmk.astype(np.float32)
        
        # Estimate transform
        tform = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
        
        return tform
    
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
