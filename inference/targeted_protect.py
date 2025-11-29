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
        
        # Get face bounding box
        bbox = face.bbox.astype(np.int32)
        face_w = bbox[2] - bbox[0]
        face_h = bbox[3] - bbox[1]
        
        # Define basic landmarks
        left_eye = kps[0]
        right_eye = kps[1]
        nose_tip = kps[2]
        left_mouth = kps[3]
        right_mouth = kps[4]
        
        eye_distance = int(np.linalg.norm(right_eye - left_eye))
        
        # Calculate face structure lines/edges
        face_center_x = (bbox[0] + bbox[2]) // 2
        face_center_y = (bbox[1] + bbox[3]) // 2
        
        # Jawline points (bottom curve of face)
        jaw_left = np.array([bbox[0] + int(face_w * 0.1), bbox[3] - int(face_h * 0.15)])
        jaw_right = np.array([bbox[2] - int(face_w * 0.1), bbox[3] - int(face_h * 0.15)])
        jaw_center = np.array([face_center_x, bbox[3] - int(face_h * 0.05)])
        
        # Cheekbone points (sides of face at eye level)
        cheek_left = np.array([bbox[0] + int(face_w * 0.05), left_eye[1]])
        cheek_right = np.array([bbox[2] - int(face_w * 0.05), right_eye[1]])
        
        # Face contour (oval outline)
        contour_top = np.array([face_center_x, bbox[1] + int(face_h * 0.15)])
        contour_left_top = np.array([bbox[0] + int(face_w * 0.08), bbox[1] + int(face_h * 0.3)])
        contour_right_top = np.array([bbox[2] - int(face_w * 0.08), bbox[1] + int(face_h * 0.3)])
        contour_left_mid = np.array([bbox[0] + int(face_w * 0.03), face_center_y])
        contour_right_mid = np.array([bbox[2] - int(face_w * 0.03), face_center_y])
        
        # Nose bridge (vertical line from eyes to nose)
        nose_bridge_top = np.array([face_center_x, (left_eye[1] + right_eye[1]) // 2])
        nose_bridge_mid = np.array([face_center_x, (nose_bridge_top[1] + nose_tip[1]) // 2])
        
        # Eye sockets (bone structure around eyes)
        left_eye_socket_inner = np.array([left_eye[0] + int(eye_distance * 0.2), left_eye[1]])
        left_eye_socket_outer = np.array([left_eye[0] - int(eye_distance * 0.15), left_eye[1]])
        right_eye_socket_inner = np.array([right_eye[0] - int(eye_distance * 0.2), right_eye[1]])
        right_eye_socket_outer = np.array([right_eye[0] + int(eye_distance * 0.15), right_eye[1]])
        
        # Forehead edges
        forehead_left = np.array([bbox[0] + int(face_w * 0.15), bbox[1] + int(face_h * 0.2)])
        forehead_right = np.array([bbox[2] - int(face_w * 0.15), bbox[1] + int(face_h * 0.2)])
        
        landmarks = {
            # Individual points
            'left_eye': left_eye,
            'right_eye': right_eye,
            'nose_tip': nose_tip,
            
            # Jawline
            'jaw_left': jaw_left,
            'jaw_right': jaw_right,
            'jaw_center': jaw_center,
            'jawline': [jaw_left, jaw_center, jaw_right],  # Line
            
            # Cheekbones
            'cheek_left': cheek_left,
            'cheek_right': cheek_right,
            'cheekbones': [cheek_left, cheek_right],
            
            # Face contour (outline)
            'contour_top': contour_top,
            'face_contour': [contour_left_top, contour_left_mid, jaw_left, 
                           contour_right_top, contour_right_mid, jaw_right],
            
            # Nose bridge
            'nose_bridge_top': nose_bridge_top,
            'nose_bridge_mid': nose_bridge_mid,
            'nose_bridge': [nose_bridge_top, nose_bridge_mid, nose_tip],
            
            # Eye sockets
            'left_eye_socket': [left_eye_socket_inner, left_eye_socket_outer],
            'right_eye_socket': [right_eye_socket_inner, right_eye_socket_outer],
            'eye_sockets': [left_eye_socket_inner, left_eye_socket_outer, 
                          right_eye_socket_inner, right_eye_socket_outer],
            
            # Forehead
            'forehead_left': forehead_left,
            'forehead_right': forehead_right,
            'forehead_edges': [forehead_left, forehead_right],
        }
        
        region_info = {'landmarks': landmarks, 'regions_protected': []}
        
        # Create masks for each target region
        for region_name in self.target_regions:
            if region_name not in landmarks:
                continue
            
            landmark_data = landmarks[region_name]
            
            # Handle both single points and lines (lists of points)
            points_to_draw = []
            if isinstance(landmark_data, list):
                points_to_draw = landmark_data
            else:
                points_to_draw = [landmark_data]
            
            # Determine region size based on feature (lines are longer/thinner)
            if 'contour' in region_name or 'jawline' in region_name:
                radius = int(w * 0.015)  # 1.5% - thin line along edges
            elif 'cheekbone' in region_name:
                radius = int(w * 0.02)  # 2% - cheekbone edges
            elif 'nose_bridge' in region_name:
                radius = int(w * 0.018)  # 1.8% - thin nose line
            elif 'eye_socket' in region_name:
                radius = int(w * 0.015)  # 1.5% - socket edges
            elif 'forehead' in region_name:
                radius = int(w * 0.02)  # 2% - forehead edges
            else:
                radius = int(w * 0.02)  # 2% default
            
            # Draw circles at each point (or draw line between points)
            if len(points_to_draw) > 1:
                # Draw line connecting points
                for i in range(len(points_to_draw) - 1):
                    pt1 = tuple(points_to_draw[i])
                    pt2 = tuple(points_to_draw[i + 1])
                    cv2.line(mask, pt1, pt2, 1.0, thickness=radius)
                # Also draw circles at endpoints
                for point in points_to_draw:
                    cv2.circle(mask, tuple(point), radius, 1.0, -1)
            else:
                # Single point - draw circle
                cv2.circle(mask, tuple(points_to_draw[0]), radius, 1.0, -1)
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
