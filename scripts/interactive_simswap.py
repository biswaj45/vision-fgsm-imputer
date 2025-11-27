"""
Interactive SimSwap face swap with image selection and result display
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def find_images():
    """Find all images in /content/"""
    search_dirs = [Path('/content'), Path.cwd()]
    extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    
    found = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for ext in extensions:
            found.extend(search_dir.glob(ext))
    
    return list(set(found))

def display_image_grid(images_dict, output_path='/content/result_grid.jpg'):
    """Create and display grid of images"""
    from IPython.display import Image, display
    
    # Load images
    imgs = []
    labels = []
    for label, path in images_dict.items():
        img = cv2.imread(str(path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
            labels.append(label)
    
    if not imgs:
        print("No images to display")
        return
    
    # Resize all to same height
    target_height = 400
    resized = []
    for img in imgs:
        h, w = img.shape[:2]
        new_w = int(w * target_height / h)
        resized.append(cv2.resize(img, (new_w, target_height)))
    
    # Stack horizontally
    grid = np.hstack(resized)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    x_pos = 0
    for i, (img, label) in enumerate(zip(resized, labels)):
        cv2.putText(grid, label, (x_pos + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(grid, label, (x_pos + 10, 30), font, 1, (0, 0, 0), 1)
        x_pos += img.shape[1]
    
    # Save and display
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, grid_bgr)
    
    print(f"\n✅ Result grid saved to: {output_path}")
    display(Image(output_path))

def main():
    print("="*70)
    print("INTERACTIVE SIMSWAP FACE SWAP")
    print("="*70)
    
    # Find images
    print("\n1. Finding images...")
    images = find_images()
    
    if not images:
        print("❌ No images found!")
        print("\nUpload images first:")
        print("  from google.colab import files")
        print("  uploaded = files.upload()")
        return
    
    print(f"\n✅ Found {len(images)} image(s):")
    for i, img in enumerate(images):
        size_mb = img.stat().st_size / (1024*1024)
        print(f"   [{i}] {img.name} ({size_mb:.2f} MB)")
    
    # Select source
    print("\n" + "="*70)
    print("SELECT SOURCE IMAGE (whose face identity to steal)")
    print("="*70)
    while True:
        try:
            source_idx = int(input(f"Enter number [0-{len(images)-1}]: "))
            if 0 <= source_idx < len(images):
                source_path = images[source_idx]
                break
            print(f"Invalid! Choose 0-{len(images)-1}")
        except ValueError:
            print("Enter a number!")
    
    print(f"✅ SOURCE: {source_path.name}")
    
    # Select target
    print("\n" + "="*70)
    print("SELECT TARGET IMAGE (whose face will be replaced)")
    print("="*70)
    while True:
        try:
            target_idx = int(input(f"Enter number [0-{len(images)-1}]: "))
            if 0 <= target_idx < len(images):
                target_path = images[target_idx]
                break
            print(f"Invalid! Choose 0-{len(images)-1}")
        except ValueError:
            print("Enter a number!")
    
    print(f"✅ TARGET: {target_path.name}")
    
    # Load images
    print("\n2. Loading images...")
    source = cv2.imread(str(source_path))
    target = cv2.imread(str(target_path))
    
    if source is None or target is None:
        print("❌ Failed to load images!")
        return
    
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    
    print(f"✅ Loaded: Source {source.shape}, Target {target.shape}")
    
    # Load SimSwap
    print("\n3. Loading SimSwap model...")
    from app.simswap_tester import SimSwapTester
    
    tester = SimSwapTester()
    success, msg = tester.load_model()
    
    if not success:
        print(f"❌ {msg}")
        return
    
    print(f"✅ {msg}")
    
    # Run face swap
    print("\n4. Running face swap...")
    print("   This may take 30-60 seconds on CPU...")
    
    swapped, status, metrics = tester.test_manipulation(target, source)
    
    print(f"\n{status}")
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    # Save and display results
    if swapped is not None:
        # Save individual result
        result_path = Path('/content/swapped_result.jpg')
        swapped_bgr = cv2.cvtColor(swapped, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(result_path), swapped_bgr)
        
        # Display grid
        print("\n" + "="*70)
        print("RESULTS:")
        print("="*70)
        
        display_image_grid({
            'SOURCE (identity)': source_path,
            'TARGET (original)': target_path,
            'RESULT (swapped)': result_path
        })
        
        print("\n📥 Download results:")
        try:
            from google.colab import files
            files.download(str(result_path))
            files.download('/content/result_grid.jpg')
        except:
            print("   (Download available in Colab files panel)")
        
        print("\n✅ SUCCESS!")
    else:
        print("\n❌ Face swap failed!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
