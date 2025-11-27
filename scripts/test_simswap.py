"""
Quick test script for SimSwap face swapping.
Tests if SimSwap can successfully swap faces before testing with perturbations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from PIL import Image
import argparse

from app.simswap_tester import SimSwapTester


def test_simswap(source_path: str, target_path: str, output_path: str = 'simswap_test_result.jpg'):
    """Test SimSwap on two images."""
    
    print("\n" + "="*60)
    print("SIMSWAP FACE SWAP TEST")
    print("="*60)
    
    # Load images
    print(f"\nLoading images...")
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    
    if source_img is None:
        print(f"❌ Failed to load source: {source_path}")
        return
    
    if target_img is None:
        print(f"❌ Failed to load target: {target_path}")
        return
    
    # Convert BGR to RGB
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    
    print(f"✓ Source image: {source_img.shape}")
    print(f"✓ Target image: {target_img.shape}")
    
    # Initialize SimSwap
    print(f"\nInitializing SimSwap...")
    tester = SimSwapTester(device='auto')
    success, message = tester.load_model()
    print(message)
    
    if not success:
        print("❌ Failed to load SimSwap model")
        return
    
    # Test face swap
    print(f"\nTesting face swap...")
    result, status, metrics = tester.test_manipulation(target_img, source_img)
    
    print(f"\n{status}")
    if metrics:
        print(f"Metrics:")
        print(f"  MSE: {metrics.get('mse', 0):.2f}")
        print(f"  PSNR: {metrics.get('psnr', 0):.2f} dB")
        print(f"  Swap Strength: {metrics.get('swap_strength', 'Unknown')}")
        print(f"  Quality (std): {metrics.get('std', 0):.2f}")
    
    # Save result
    if result is not None:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_bgr)
        print(f"\n✅ Result saved to: {output_path}")
        print(f"   Compare with original target to see face swap quality")
    else:
        print(f"\n❌ Face swap failed - no result generated")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SimSwap face swapping")
    parser.add_argument('--source', type=str, required=True,
                       help='Path to source face image')
    parser.add_argument('--target', type=str, required=True,
                       help='Path to target face image')
    parser.add_argument('--output', type=str, default='simswap_test_result.jpg',
                       help='Output path for result')
    
    args = parser.parse_args()
    
    test_simswap(args.source, args.target, args.output)
