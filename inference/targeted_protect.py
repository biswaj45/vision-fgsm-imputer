"""
Targeted FGSM Protection - Apply perturbations only to facial landmarks
This makes protection invisible to humans while disrupting deepfake models.
"""

import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Optional, Tuple, List
import insightface
from insightface.app import FaceAnalysis

class TargetedNoiseImputer:
    """
    Apply FGSM perturbations only to critical facial regions:
    - Eyes (left and right)
    - Nose
    - Mouth (optional)
    
    This is less visible to humans but disrupts face swap models
    which rely heavily on eye and nose alignment.
    """
    
    def __init__(
        self,
        model_path: str,
        epsilon: float = 0.30,
        target_regions: List[str] = ['left_eyebrow', 'right_eyebrow', 'nose_tip', 'inner_eyes'],
        feather_radius: int = 20,
        device: str = 'auto'
    ):
        """
        Args:
            model_path: Path to trained FGSM model
            epsilon: Perturbation strength (higher = more protection, less visible with targeting)
            target_regions: Which facial features to protect
                Options: 'left_eyebrow', 'right_eyebrow', 'nose_tip', 'nose_bridge',
                        'inner_eyes' (eye corners), 'hairline', 'nostrils'
            feather_radius: Smooth transition at mask edges (reduces visibility)
            device: 'cuda', 'cpu', or 'auto'
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.epsilon = epsilon
        self.target_regions = target_regions
        self.feather_radius = feather_radius
        
        # Load FGSM model
        from models.unet_tiny import TinyUNet
        self.model = TinyUNet()
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ FGSM model loaded on {self.device}")
        
        # Initialize face detector for landmarks
        self.face_app = None
        self._init_face_detector()
    
    def _init_face_detector(self):
        """Initialize InsightFace for facial landmark detection."""
        try:
            self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=-1, det_size=(640, 640))
            print("✅ Face landmark detector loaded")
        except Exception as e:
            print(f"⚠️  Face detector error: {e}")
            self.face_app = None
    
    def _get_facial_region_masks(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Detect faces and create masks for specific facial regions.
        
        Returns:
            combined_mask: Binary mask of all target regions
            region_info: Dict with region details
        """
        if self.face_app is None:
            # Fallback: protect center region (likely contains face)
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (w//2, h//2), min(h, w)//3, 1.0, -1)
            return mask, {'method': 'fallback_center'}
        
        # Detect faces
        faces = self.face_app.get(image)
        
        if len(faces) == 0:
            print("⚠️  No face detected, protecting center region")
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (w//2, h//2), min(h, w)//3, 1.0, -1)
            return mask, {'method': 'no_face_fallback'}
        
        # Use first detected face
        face = faces[0]
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Get 5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
        kps = face.kps.astype(np.int32)
        
        # Get face bounding box for additional features
        bbox = face.bbox.astype(np.int32)
        
        # Define landmark indices and derived points
        left_eye = kps[0]
        right_eye = kps[1]
        nose_tip = kps[2]
        left_mouth = kps[3]
        right_mouth = kps[4]
        
        # Calculate additional feature points
        eye_center_y = (left_eye[1] + right_eye[1]) // 2
        eye_distance = int(np.linalg.norm(right_eye - left_eye))
        
        # Eyebrow positions (above eyes)
        left_eyebrow = np.array([left_eye[0], left_eye[1] - int(eye_distance * 0.3)])
        right_eyebrow = np.array([right_eye[0], right_eye[1] - int(eye_distance * 0.3)])
        
        # Nose bridge (between eyes)
        nose_bridge = np.array([
            (left_eye[0] + right_eye[0]) // 2,
            eye_center_y
        ])
        
        # Nostrils (below and beside nose tip)
        nostril_offset = int(eye_distance * 0.15)
        left_nostril = np.array([nose_tip[0] - nostril_offset, nose_tip[1] + nostril_offset // 2])
        right_nostril = np.array([nose_tip[0] + nostril_offset, nose_tip[1] + nostril_offset // 2])
        
        # Inner eye corners (tearducts)
        inner_left_eye = np.array([left_eye[0] + int(eye_distance * 0.15), left_eye[1]])
        inner_right_eye = np.array([right_eye[0] - int(eye_distance * 0.15), right_eye[1]])
        
        # Hairline (top of forehead)
        hairline = np.array([(left_eye[0] + right_eye[0]) // 2, bbox[1] + int(eye_distance * 0.2)])
        
        landmarks = {
            'left_eye': left_eye,
            'right_eye': right_eye,
            'nose_tip': nose_tip,
            'left_eyebrow': left_eyebrow,
            'right_eyebrow': right_eyebrow,
            'nose_bridge': nose_bridge,
            'left_nostril': left_nostril,
            'right_nostril': right_nostril,
            'nostrils': nose_tip,  # Alias for both
            'inner_left_eye': inner_left_eye,
            'inner_right_eye': inner_right_eye,
            'inner_eyes': nose_bridge,  # Alias for area between eyes
            'hairline': hairline,
            'mouth': (left_mouth + right_mouth) // 2
        }
        
        region_info = {'landmarks': landmarks, 'regions_protected': []}
        
        # Create masks for each target region
        for region_name in self.target_regions:
            if region_name not in landmarks:
                continue
            
            point = landmarks[region_name]
            
            # Determine region size based on feature (smaller = less visible)
            if 'eyebrow' in region_name:
                radius = int(w * 0.025)  # 2.5% - very small, just eyebrow hair
            elif 'nose_tip' in region_name:
                radius = int(w * 0.02)  # 2% - just the tip
            elif 'nose_bridge' in region_name:
                radius = int(w * 0.03)  # 3% - bridge area
            elif 'nostril' in region_name or region_name == 'nostrils':
                radius = int(w * 0.015)  # 1.5% - tiny nostril holes
            elif 'inner' in region_name:
                radius = int(w * 0.02)  # 2% - eye corners/tearducts
            elif 'hairline' in region_name:
                radius = int(w * 0.04)  # 4% - hairline edge
            elif 'eye' in region_name:
                radius = int(w * 0.035)  # 3.5% - smaller than before
            elif region_name == 'mouth':
                radius = int(w * 0.05)  # 5%
            else:
                radius = int(w * 0.03)  # 3% default
            
            # Draw circular region
            cv2.circle(mask, tuple(point), radius, 1.0, -1)
            region_info['regions_protected'].append({
                'name': region_name,
                'center': point.tolist(),
                'radius': radius
            })
        
        # Apply feathering (Gaussian blur) for smooth edges
        if self.feather_radius > 0:
            mask = cv2.GaussianBlur(mask, (self.feather_radius*2+1, self.feather_radius*2+1), 0)
        
        return mask, region_info
    
    def impute_from_array(
        self,
        image: np.ndarray,
        return_mask: bool = False
    ) -> np.ndarray:
        """
        Apply targeted FGSM perturbations to facial regions only.
        
        Args:
            image: Input RGB image (H, W, 3), range [0, 255]
            return_mask: If True, return (protected_image, mask, region_info)
        
        Returns:
            protected_image: Image with targeted perturbations
        """
        # Get facial region masks
        mask, region_info = self._get_facial_region_masks(image)
        
        # Normalize image for model
        img_normalized = image.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(img_tensor)
        
        # Compute loss (MSE between input and output)
        loss = torch.nn.functional.mse_loss(output, img_tensor)
        
        # Backward pass to get gradients
        self.model.zero_grad()
        loss.backward()
        
        # Get gradient and create perturbation
        gradient = img_tensor.grad.data
        perturbation = self.epsilon * gradient.sign()
        
        # Convert to numpy
        perturbation_np = perturbation.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Apply mask to perturbation (only affect facial regions)
        mask_3ch = np.stack([mask] * 3, axis=-1)
        targeted_perturbation = perturbation_np * mask_3ch
        
        # Add targeted perturbation to original image
        protected = img_normalized + targeted_perturbation
        protected = np.clip(protected, 0, 1)
        protected = (protected * 255).astype(np.uint8)
        
        # Calculate protection statistics
        perturb_strength = np.abs(targeted_perturbation).mean()
        coverage = (mask > 0.1).sum() / mask.size * 100
        
        print(f"✅ Targeted protection applied:")
        print(f"   Regions: {', '.join(self.target_regions)}")
        print(f"   Coverage: {coverage:.1f}% of image")
        print(f"   Avg perturbation: {perturb_strength:.4f}")
        print(f"   Protected areas: {len(region_info.get('regions_protected', []))}")
        
        if return_mask:
            # Visualize mask for debugging
            mask_vis = (mask * 255).astype(np.uint8)
            mask_colored = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
            return protected, mask_colored, region_info
        
        return protected
    
    def visualize_protection(
        self,
        original: np.ndarray,
        protected: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Create visualization showing original, mask, and protected images.
        
        Returns:
            comparison: Side-by-side visualization
        """
        h, w = original.shape[:2]
        
        # Resize for display
        display_h = 400
        scale = display_h / h
        new_w = int(w * scale)
        
        orig_resized = cv2.resize(original, (new_w, display_h))
        prot_resized = cv2.resize(protected, (new_w, display_h))
        mask_resized = cv2.resize(mask, (new_w, display_h))
        
        # Create difference map
        diff = cv2.absdiff(orig_resized, prot_resized)
        diff_enhanced = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        
        # Stack horizontally
        row1 = np.hstack([orig_resized, mask_resized])
        row2 = np.hstack([prot_resized, diff_enhanced])
        comparison = np.vstack([row1, row2])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'ORIGINAL', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'PROTECTION MASK', (new_w + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'PROTECTED', (10, display_h + 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'DIFFERENCE (enhanced)', (new_w + 10, display_h + 30), font, 0.7, (255, 255, 255), 2)
        
        return comparison


def compare_global_vs_targeted(
    image_path: str,
    model_path: str,
    epsilon: float = 0.30
):
    """
    Compare global FGSM vs targeted FGSM protection.
    """
    from PIL import Image
    from inference.predict import NoiseImputer  # Original global method
    
    # Load image
    image = np.array(Image.open(image_path))
    
    print("="*80)
    print("COMPARING PROTECTION METHODS")
    print("="*80)
    
    # Method 1: Global protection (current)
    print("\n1. GLOBAL PROTECTION (current method):")
    global_imputer = NoiseImputer(model_path=model_path, epsilon=epsilon)
    protected_global = global_imputer.impute_from_array(image)
    
    # Method 2: Targeted protection (new)
    print("\n2. TARGETED PROTECTION (facial features only):")
    targeted_imputer = TargetedNoiseImputer(
        model_path=model_path,
        epsilon=epsilon,
        target_regions=['left_eye', 'right_eye', 'nose']
    )
    protected_targeted, mask_vis, region_info = targeted_imputer.impute_from_array(
        image,
        return_mask=True
    )
    
    # Compare visibility
    print("\n" + "="*80)
    print("VISIBILITY COMPARISON:")
    print("="*80)
    
    diff_global = np.abs(image.astype(float) - protected_global.astype(float)).mean()
    diff_targeted = np.abs(image.astype(float) - protected_targeted.astype(float)).mean()
    
    print(f"Global protection - Avg pixel change: {diff_global:.2f}")
    print(f"Targeted protection - Avg pixel change: {diff_targeted:.2f}")
    print(f"Reduction in visibility: {(1 - diff_targeted/diff_global)*100:.1f}%")
    
    # Save results
    output_dir = Path('/content') if Path('/content').exists() else Path('.')
    
    Image.fromarray(protected_global).save(output_dir / 'protected_global.jpg')
    Image.fromarray(protected_targeted).save(output_dir / 'protected_targeted.jpg')
    Image.fromarray(mask_vis).save(output_dir / 'protection_mask.jpg')
    
    # Create comparison visualization
    comparison = targeted_imputer.visualize_protection(image, protected_targeted, mask_vis)
    Image.fromarray(comparison).save(output_dir / 'comparison_visualization.jpg')
    
    print(f"\n✅ Results saved to {output_dir}")
    
    return protected_global, protected_targeted, mask_vis
