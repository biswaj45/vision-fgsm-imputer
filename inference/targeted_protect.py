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
        epsilon: float = 0.40,
        target_regions: List[str] = ['face_contour', 'jawline', 'nose_bridge'],
        feather_radius: int = 25,
        device: str = 'auto'
    ):
        """
        Args:
            model_path: Path to trained FGSM model
            epsilon: Perturbation strength (higher = more protection, less visible with targeting)
            target_regions: Which facial features to protect
                Options: 'face_contour', 'jawline', 'cheekbones', 'nose_bridge',
                        'eye_sockets', 'forehead_edges'
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
            # Use buffalo_l which has 106 landmarks
            self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=-1, det_size=(640, 640))
            print("✅ Face landmark detector loaded (106 points)")
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
        
        # Get landmarks - InsightFace buffalo_l provides 106 landmarks if available
        # Fallback to 5-point if 106 not available
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            # 106 landmarks available - use proper face contour
            landmarks_all = face.landmark_2d_106.astype(np.int32)
            
            # Key landmark groups in 106-point model:
            # 0-32: Face contour (jawline + cheeks)
            # 33-42: Left eyebrow
            # 43-52: Right eyebrow  
            # 53-72: Nose
            # 73-86: Left eye
            # 87-100: Right eye
            # 101-105: Mouth outer
            
            landmarks = {
                'face_contour': landmarks_all[0:33],  # Complete face outline
                'jawline': landmarks_all[0:17],  # Just the jaw
                'left_eyebrow': landmarks_all[33:43],
                'right_eyebrow': landmarks_all[43:53],
                'nose_bridge': landmarks_all[53:58],  # Nose bridge line
                'nose_tip': landmarks_all[58:64],  # Nose bottom
                'left_eye': landmarks_all[73:87],
                'right_eye': landmarks_all[87:101],
                'mouth': landmarks_all[101:106],
                'cheekbones': [landmarks_all[1], landmarks_all[15]],  # Side points
            }
        else:
            # Fallback to 5-point landmarks (less accurate)
            kps = face.kps.astype(np.int32)
            bbox = face.bbox.astype(np.int32)
            
            # Use simple circular regions as fallback
            landmarks = {
                'left_eye': [kps[0]],
                'right_eye': [kps[1]],
                'nose_bridge': [kps[2]],
                'face_contour': [bbox[[0,1]], bbox[[2,1]], bbox[[2,3]], bbox[[0,3]]],  # Bbox corners
            }
            print("⚠️  Using 5-point fallback (less accurate)")
        
        region_info = {'landmarks': {}, 'regions_protected': []}
        
        # Create masks for each target region
        for region_name in self.target_regions:
            if region_name not in landmarks:
                continue
            
            landmark_data = landmarks[region_name]
            
            # Determine region size - thinner lines for edges
            if 'contour' in region_name or 'jawline' in region_name:
                line_thickness = max(1, int(w * 0.008))  # 0.8% - very thin edge line
            elif 'eyebrow' in region_name:
                line_thickness = max(1, int(w * 0.006))  # 0.6% - thin eyebrow line
            elif 'nose' in region_name:
                line_thickness = max(1, int(w * 0.01))  # 1% - nose line
            elif 'eye' in region_name:
                line_thickness = max(1, int(w * 0.008))  # 0.8% - eye outline
            else:
                line_thickness = max(1, int(w * 0.01))  # 1% default
            
            # Draw polyline or circles
            if isinstance(landmark_data, np.ndarray) and len(landmark_data) > 1:
                # Multiple points - draw connected line
                pts = landmark_data.reshape((-1, 1, 2))
                cv2.polylines(mask, [pts], isClosed=False, color=1.0, thickness=line_thickness)
            elif isinstance(landmark_data, list) and len(landmark_data) > 1:
                # List of points - draw connected line
                pts = np.array(landmark_data, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(mask, [pts], isClosed=False, color=1.0, thickness=line_thickness)
            else:
                # Single point - draw small circle
                if isinstance(landmark_data, np.ndarray):
                    pt = tuple(landmark_data.flatten())
                elif isinstance(landmark_data, list) and len(landmark_data) == 1:
                    pt = tuple(landmark_data[0])
                else:
                    continue
                cv2.circle(mask, pt, line_thickness*2, 1.0, -1)
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
