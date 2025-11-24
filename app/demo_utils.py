"""
Demo utilities for Gradio app.
"""

import numpy as np
import cv2
from typing import Tuple
import time


def create_side_by_side(
    original: np.ndarray,
    perturbed: np.ndarray,
    add_labels: bool = True
) -> np.ndarray:
    """
    Create side-by-side comparison.
    
    Args:
        original: Original image (H, W, C)
        perturbed: Perturbed image (H, W, C)
        add_labels: Whether to add text labels
    
    Returns:
        Side-by-side image (H, W*2, C)
    """
    # Ensure same size
    if original.shape != perturbed.shape:
        perturbed = cv2.resize(perturbed, (original.shape[1], original.shape[0]))
    
    # Stack horizontally
    combined = np.hstack([original, perturbed])
    
    if add_labels:
        # Add labels
        h, w = original.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (255, 255, 255)
        
        # Original label
        cv2.putText(combined, "Original", (10, 30), font, font_scale, color, thickness)
        
        # Protected label
        cv2.putText(combined, "Protected", (w + 10, 30), font, font_scale, color, thickness)
    
    return combined


def compute_difference_heatmap(
    original: np.ndarray,
    perturbed: np.ndarray,
    scale: float = 20.0
) -> np.ndarray:
    """
    Create a heatmap showing differences.
    
    Args:
        original: Original image (H, W, C)
        perturbed: Perturbed image (H, W, C)
        scale: Scale factor for visualization
    
    Returns:
        Difference heatmap (H, W, C)
    """
    # Compute absolute difference
    diff = np.abs(perturbed.astype(np.float32) - original.astype(np.float32))
    
    # Convert to grayscale
    diff_gray = np.mean(diff, axis=2)
    
    # Scale for visibility
    diff_gray = np.clip(diff_gray * scale, 0, 255).astype(np.uint8)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
    
    return heatmap


def create_comparison_grid(
    original: np.ndarray,
    perturbed: np.ndarray,
    show_heatmap: bool = True
) -> np.ndarray:
    """
    Create a comprehensive comparison grid.
    
    Args:
        original: Original image (H, W, C)
        perturbed: Perturbed image (H, W, C)
        show_heatmap: Whether to include difference heatmap
    
    Returns:
        Comparison grid
    """
    if show_heatmap:
        heatmap = compute_difference_heatmap(original, perturbed)
        grid = np.hstack([original, perturbed, heatmap])
    else:
        grid = np.hstack([original, perturbed])
    
    return grid


def format_inference_time(elapsed_ms: float) -> str:
    """
    Format inference time for display.
    
    Args:
        elapsed_ms: Elapsed time in milliseconds
    
    Returns:
        Formatted string
    """
    if elapsed_ms < 1000:
        return f"{elapsed_ms:.2f}ms"
    else:
        return f"{elapsed_ms/1000:.2f}s"


def get_image_info(image: np.ndarray) -> str:
    """
    Get image information string.
    
    Args:
        image: Input image (H, W, C)
    
    Returns:
        Information string
    """
    h, w, c = image.shape
    size_kb = (image.nbytes / 1024)
    
    info = f"Size: {w}x{h}x{c}\n"
    info += f"Memory: {size_kb:.2f} KB\n"
    info += f"Range: [{image.min()}, {image.max()}]"
    
    return info


def add_protection_badge(
    image: np.ndarray,
    position: str = 'bottom-right'
) -> np.ndarray:
    """
    Add a "Protected" badge to the image.
    
    Args:
        image: Input image (H, W, C)
        position: Badge position ('bottom-right', 'bottom-left', 'top-right', 'top-left')
    
    Returns:
        Image with badge
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    # Badge parameters
    text = "PROTECTED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    color = (0, 255, 0)  # Green
    
    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Calculate position
    padding = 10
    if position == 'bottom-right':
        x = w - text_size[0] - padding
        y = h - padding
    elif position == 'bottom-left':
        x = padding
        y = h - padding
    elif position == 'top-right':
        x = w - text_size[0] - padding
        y = text_size[1] + padding
    else:  # top-left
        x = padding
        y = text_size[1] + padding
    
    # Add background rectangle
    rect_padding = 5
    cv2.rectangle(
        img,
        (x - rect_padding, y - text_size[1] - rect_padding),
        (x + text_size[0] + rect_padding, y + rect_padding),
        (0, 0, 0),
        -1
    )
    
    # Add text
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return img


class Timer:
    """Simple timer context manager."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = (time.time() - self.start_time) * 1000  # Convert to ms
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed


if __name__ == "__main__":
    # Test utilities
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Test side-by-side
    comparison = create_side_by_side(dummy_img, dummy_img)
    print(f"Side-by-side shape: {comparison.shape}")
    
    # Test image info
    info = get_image_info(dummy_img)
    print(f"Image info:\n{info}")
    
    # Test timer
    with Timer() as timer:
        time.sleep(0.1)
    print(f"Timer test: {format_inference_time(timer.get_elapsed_ms())}")
