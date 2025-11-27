"""
Quick SimSwap test script for Colab - handles uploaded filenames
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def find_uploaded_images():
    """Find uploaded images in /content/"""
    content_dir = Path('/content')
    
    # Look for common image extensions
    extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    found = []
    
    for ext in extensions:
        found.extend(content_dir.glob(ext))
    
    return sorted(found)

def test_simswap():
    print("="*70)
    print("SIMSWAP FACE SWAP TEST - COLAB VERSION")
    print("="*70)
    
    # Find uploaded images
    print("\n1. Finding uploaded images...")
    images = find_uploaded_images()
    
    if not images:
        print("❌ No images found in /content/")
        print("Please upload images using:")
        print("  from google.colab import files")
        print("  uploaded = files.upload()")
        return False
    
    print(f"✅ Found {len(images)} image(s):")
    for i, img in enumerate(images):
        size_mb = img.stat().st_size / (1024*1024)
        print(f"   [{i}] {img.name} ({size_mb:.2f} MB)")
    
    # Select source and target
    if len(images) < 2:
        print("\n❌ Need at least 2 images (source face + target face)")
        print("Upload more images and try again.")
        return False
    
    source_path = images[0]
    target_path = images[1] if len(images) > 1 else images[0]
    
    print(f"\n2. Using:")
    print(f"   Source: {source_path.name}")
    print(f"   Target: {target_path.name}")
    
    # Load images
    print("\n3. Loading images...")
    source = cv2.imread(str(source_path))
    target = cv2.imread(str(target_path))
    
    if source is None:
        print(f"❌ Failed to load source: {source_path}")
        return False
    if target is None:
        print(f"❌ Failed to load target: {target_path}")
        return False
    
    # Convert BGR to RGB
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    
    print(f"✅ Images loaded:")
    print(f"   Source shape: {source.shape}")
    print(f"   Target shape: {target.shape}")
    
    # Initialize SimSwap
    print("\n4. Loading SimSwap model...")
    from app.simswap_tester import SimSwapTester
    
    tester = SimSwapTester()
    success, msg = tester.load_model()
    
    if not success:
        print(f"❌ Model loading failed: {msg}")
        return False
    
    print(f"✅ {msg}")
    
    # Run face swap
    print("\n5. Running face swap...")
    swapped, status, metrics = tester.test_manipulation(target, source)
    
    print(f"\nStatus: {status}")
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    # Save result
    if swapped is not None:
        output_path = Path('/content/swapped_simswap.jpg')
        swapped_bgr = cv2.cvtColor(swapped, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), swapped_bgr)
        print(f"\n✅ Result saved to: {output_path}")
        
        # Display in Colab
        try:
            from IPython.display import Image, display
            from PIL import Image as PILImage
            
            # Create comparison
            fig_width = 1200
            h, w = target.shape[:2]
            scale = fig_width / (w * 3)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize for display
            source_display = cv2.resize(source, (new_w, new_h))
            target_display = cv2.resize(target, (new_w, new_h))
            swapped_display = cv2.resize(swapped, (new_w, new_h))
            
            # Stack horizontally
            comparison = np.hstack([source_display, target_display, swapped_display])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, 'SOURCE', (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(comparison, 'TARGET', (new_w + 10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(comparison, 'SWAPPED', (new_w*2 + 10, 30), font, 1, (255, 255, 255), 2)
            
            # Save comparison
            comparison_path = Path('/content/simswap_comparison.jpg')
            comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(comparison_path), comparison_bgr)
            
            print(f"✅ Comparison saved to: {comparison_path}")
            
            # Display
            print("\n" + "="*70)
            print("RESULTS:")
            print("="*70)
            display(Image(str(comparison_path)))
            
        except Exception as e:
            print(f"⚠️  Display error (results still saved): {e}")
        
        return True
    else:
        print(f"\n❌ Face swap failed!")
        return False

if __name__ == "__main__":
    success = test_simswap()
    
    if success:
        print("\n" + "="*70)
        print("✅ TEST COMPLETE - Face swap successful!")
        print("="*70)
        print("\nNext steps:")
        print("1. Apply FGSM protection to target image")
        print("2. Re-run face swap on protected image")
        print("3. Compare metrics to verify protection effectiveness")
    else:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
    
    sys.exit(0 if success else 1)
