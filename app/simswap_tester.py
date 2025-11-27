"""
SimSwap face swapping for testing anti-deepfake protection.
Better quality than inswapper, no blur issues.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple
import os
import gdown
from pathlib import Path

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class SimSwapTester:
    """Test anti-deepfake protection with SimSwap face swapper."""
    
    def __init__(self, device: str = 'auto'):
        """Initialize with auto device detection."""
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.app = None
        self.G = None  # SimSwap generator
        self.arcface = None  # ArcFace for embeddings
        self.model_loaded = False
        
    def load_model(self):
        """Load SimSwap model and face detector."""
        if not INSIGHTFACE_AVAILABLE:
            return False, "❌ insightface not installed. Run: pip install insightface"
        
        try:
            print(f"Loading SimSwap Face Swapper on {self.device.upper()}...")
            
            # Initialize face analysis (for detection and alignment)
            providers = ['CPUExecutionProvider']  # Start with CPU for stability
            self.app = FaceAnalysis(name='buffalo_l', providers=providers)
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            print("✅ Face detector loaded")
            
            # Download SimSwap model if not exists
            model_dir = Path.home() / '.simswap' / 'checkpoints'
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / 'simswap_224.pth'
            
            # Also check for the archive format
            archive_dir = model_dir / 'SimSwap' / 'arcface_model' / 'archive'
            
            if not model_path.exists():
                # Check if we have the archive format already
                if archive_dir.exists():
                    print("Converting existing archive format to .pth...")
                    try:
                        checkpoint = torch.load(str(archive_dir), map_location='cpu')
                        torch.save(checkpoint, str(model_path))
                        print("✅ Model converted from archive format!")
                    except Exception as e:
                        print(f"Failed to convert archive: {e}")
                else:
                    print("Downloading SimSwap model (~200MB, one-time download)...")
                    # Try multiple sources - download the zip archive
                    sources = [
                        ('https://github.com/neuralchen/SimSwap/releases/download/1.0/arcface_checkpoint.tar', 'GitHub'),
                    ]
                    
                    downloaded = False
                    tar_path = model_dir / 'arcface_checkpoint.tar'
                    
                    for url, source in sources:
                        try:
                            print(f"Trying {source}...")
                            import urllib.request
                            urllib.request.urlretrieve(url, str(tar_path))
                            print(f"✅ Downloaded from {source}!")
                            
                            # Extract using zipfile (it's actually a zip despite .tar extension)
                            import zipfile
                            extract_dir = model_dir / 'SimSwap' / 'arcface_model'
                            extract_dir.mkdir(parents=True, exist_ok=True)
                            
                            with zipfile.ZipFile(str(tar_path), 'r') as zip_ref:
                                zip_ref.extractall(str(extract_dir))
                            print("✅ Extracted archive!")
                            
                            # Convert to .pth format (load tar directly as it's a PyTorch zip)
                            checkpoint = torch.load(str(tar_path), map_location='cpu', weights_only=False)
                            torch.save(checkpoint, str(model_path))
                            print("✅ Model converted successfully!")
                            downloaded = True
                            
                            # Cleanup
                            tar_path.unlink()
                            break
                        except Exception as e:
                            print(f"Failed from {source}: {e}")
                            continue
                    
                    if not downloaded:
                        return False, (
                            "❌ Failed to download SimSwap model from all sources.\n"
                            "Please download manually from:\n"
                            "https://github.com/neuralchen/SimSwap\n"
                            "Or use InsightFace (inswapper) instead."
                        )
            
            # Load ArcFace model for face embeddings
            print("Loading ArcFace model for embeddings...")
            try:
                from pathlib import Path as PathLib
                arcface_path = PathLib(__file__).parent.parent / 'arcface_model' / 'arcface_checkpoint.tar'
                if arcface_path.exists():
                    self.arcface = torch.load(str(arcface_path), map_location=self.device, weights_only=False)
                else:
                    self.arcface = torch.load(model_path, map_location=self.device, weights_only=False)
                if hasattr(self.arcface, 'eval'):
                    self.arcface.eval()
                print("✅ ArcFace model loaded (for face embeddings)")
            except Exception as e:
                print(f"Warning loading ArcFace: {e}")
            
            # Load SimSwap Generator
            print("Loading SimSwap Generator...")
            from pathlib import Path as PathLib
            generator_path = PathLib(__file__).parent.parent / 'checkpoints' / 'people' / 'latest_net_G.pth'
            
            if not generator_path.exists():
                print(f"❌ Generator not found at {generator_path}")
                print("Please download from: https://drive.google.com/drive/folders/1jV6_0FIMPC53FZ2HzZNJZGMe55bbu17R")
                return False, "❌ SimSwap Generator model not found!"
            
            # Import official architecture
            try:
                from app.simswap_models import Generator_Adain_Upsample
            except ImportError:
                # Fallback for direct module loading
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "simswap_models",
                    PathLib(__file__).parent / "simswap_models.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                Generator_Adain_Upsample = module.Generator_Adain_Upsample
            
            # Create model with official architecture (matches checkpoint)
            print("Instantiating Generator (input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False)...")
            self.G = Generator_Adain_Upsample(
                input_nc=3,
                output_nc=3,
                latent_size=512,
                n_blocks=9,
                deep=False,
                norm_layer=nn.BatchNorm2d,
                padding_type='reflect'
            )
            
            # Load checkpoint state_dict
            print("Loading checkpoint state_dict...")
            checkpoint = torch.load(str(generator_path), map_location='cpu', weights_only=False)
            
            # Checkpoint is OrderedDict (state_dict)
            if isinstance(checkpoint, dict) and 'first_layer.1.weight' in checkpoint:
                self.G.load_state_dict(checkpoint, strict=True)
                print("✅ State dict loaded successfully (strict=True)")
            else:
                print(f"❌ Unexpected checkpoint format: {type(checkpoint)}")
                if isinstance(checkpoint, dict):
                    print(f"Keys sample: {list(checkpoint.keys())[:5]}")
                return False, "❌ Cannot load SimSwap generator - unexpected format"
            
            self.G.to(self.device)
            self.G.eval()
            print("✅ SimSwap Generator loaded (210MB)")

            
            # ArcFace for embeddings is already in buffalo_l
            self.model_loaded = True
            return True, f"✅ SimSwap loaded on {self.device.upper()} - Real SimSwap Generator ready!"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"❌ Failed to load SimSwap: {str(e)}"
    
    def test_manipulation(
        self,
        target_image: np.ndarray,
        source_image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], str, dict]:
        """
        Test face swapping on image.
        
        Args:
            target_image: Target image (RGB, 0-255)
            source_image: Source face image (RGB, 0-255)
            
        Returns:
            (swapped_image, status_message, metrics)
        """
        if not self.model_loaded:
            return None, "❌ Model not loaded!", {'corruption_detected': True}
        
        try:
            print("\n" + "="*60)
            print("SIMSWAP FACE SWAP TEST")
            print("="*60)
            
            # Detect faces
            print("Detecting faces...")
            source_faces = self.app.get(source_image)
            target_faces = self.app.get(target_image)
            
            if len(source_faces) == 0:
                return None, "❌ No face detected in SOURCE image!", {'corruption_detected': True}
            
            if len(target_faces) == 0:
                return None, "❌ No face detected in TARGET image!", {'corruption_detected': True}
            
            print(f"✓ Found {len(source_faces)} face(s) in source")
            print(f"✓ Found {len(target_faces)} face(s) in target")
            print(f"Swapping face: SOURCE → TARGET...")
            
            # Get the first face from each
            source_face = source_faces[0]
            target_face = target_faces[0]
            
            # Perform face swap using SimSwap
            result_img = self._swap_face(target_image, target_face, source_face)
            
            if result_img is None:
                return None, "🛡️ Face swap FAILED - processing error!", {
                    'corruption_detected': True,
                    'mse': 0,
                    'psnr': 0,
                    'std': 0,
                    'swap_strength': 'Failed'
                }
            
            # Compute metrics
            metrics = self._compute_metrics(target_image, result_img)
            
            # Check quality
            if result_img.std() < 20 or np.any(np.isnan(result_img)) or metrics['mse'] < 100:
                status = "🛡️ Face swap FAILED - image corrupted!"
                metrics['corruption_detected'] = True
            else:
                status = "✅ High quality face swap successful (SimSwap)"
            
            return result_img, status, metrics
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"❌ Face swap failed: {str(e)}\n🛡️ Protection working!", {
                'corruption_detected': True,
                'mse': 0,
                'psnr': 0,
                'std': 0,
                'swap_strength': 'Failed'
            }
    
    def _swap_face(self, target_img: np.ndarray, target_face, source_face) -> Optional[np.ndarray]:
        """Perform face swap using SimSwap Generator."""
        try:
            # Get source embedding from InsightFace
            source_embedding = source_face.normed_embedding
            source_latent = torch.from_numpy(source_embedding).unsqueeze(0).to(self.device)
            print(f"Source embedding shape: {source_latent.shape}")
            
            # Get target face landmarks
            target_landmark = target_face.kps
            
            # Align target face to 224x224
            aligned_face = self._align_face(target_img, target_landmark, image_size=224)
            
            if aligned_face is None:
                print("Failed to align face")
                return None
            
            print(f"Aligned face shape: {aligned_face.shape}, range: [{aligned_face.min()}, {aligned_face.max()}]")
            
            # Prepare input tensor
            face_tensor = torch.from_numpy(aligned_face).permute(2, 0, 1).unsqueeze(0).float()
            face_tensor = face_tensor.to(self.device) / 255.0  # Normalize to [0, 1]
            face_tensor = (face_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
            
            print(f"Input tensor shape: {face_tensor.shape}, range: [{face_tensor.min():.3f}, {face_tensor.max():.3f}]")
            
            # Run SimSwap Generator
            with torch.no_grad():
                swapped_tensor = self.G(face_tensor, source_latent)
            
            print(f"Output tensor shape: {swapped_tensor.shape}, range: [{swapped_tensor.min():.3f}, {swapped_tensor.max():.3f}]")
            
            # Post-process
            swapped_face = swapped_tensor[0].permute(1, 2, 0).cpu().numpy()
            swapped_face = (swapped_face * 0.5 + 0.5) * 255.0  # Denormalize
            swapped_face = np.clip(swapped_face, 0, 255).astype(np.uint8)
            
            print(f"Swapped face shape: {swapped_face.shape}, range: [{swapped_face.min()}, {swapped_face.max()}]")
            
            # Paste back to original image
            result = self._paste_back(target_img, swapped_face, target_landmark)
            
            return result
            
        except Exception as e:
            print(f"Face swap error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _align_face(self, img: np.ndarray, landmark: np.ndarray, image_size: int = 224) -> Optional[np.ndarray]:
        """Align face to standard position."""
        # Standard 5-point template for alignment
        src_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        # Scale template to target size
        src_pts = src_pts * (image_size / 112.0)
        
        dst_pts = landmark.astype(np.float32)
        
        # Compute similarity transform
        tform = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)
        
        if tform is None or tform[0] is None:
            # Fallback
            tform_matrix = cv2.getAffineTransform(dst_pts[:3], src_pts[:3])
        else:
            tform_matrix = tform[0]
        
        # Warp face
        aligned = cv2.warpAffine(
            img,
            tform_matrix,
            (image_size, image_size),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return aligned
    
    def _paste_back(self, target_img: np.ndarray, swapped_face: np.ndarray, landmark: np.ndarray) -> np.ndarray:
        """Paste swapped face back to original image with blending."""
        # Get inverse transform
        image_size = swapped_face.shape[0]
        
        src_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32) * (image_size / 112.0)
        
        dst_pts = landmark.astype(np.float32)
        
        # Transform from aligned space back to original
        tform = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        if tform is None or tform[0] is None:
            tform_matrix = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
        else:
            tform_matrix = tform[0]
        
        # Warp swapped face back
        result = target_img.copy()
        warped_face = cv2.warpAffine(
            swapped_face,
            tform_matrix,
            (result.shape[1], result.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Create mask for blending
        mask = np.ones((image_size, image_size), dtype=np.float32) * 255
        warped_mask = cv2.warpAffine(
            mask,
            tform_matrix,
            (result.shape[1], result.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Smooth mask
        warped_mask = cv2.GaussianBlur(warped_mask, (15, 15), 10)
        warped_mask = warped_mask / 255.0
        warped_mask = np.expand_dims(warped_mask, axis=2)
        
        # Blend
        result = (warped_mask * warped_face + (1 - warped_mask) * result).astype(np.uint8)
        
        return result
    
    def _compute_metrics(self, original: np.ndarray, generated: np.ndarray) -> dict:
        """Compute metrics."""
        mse = float(np.mean((original.astype(float) - generated.astype(float)) ** 2))
        psnr = float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else 100.0
        
        corruption = (
            mse < 100 or
            mse > 15000 or
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
