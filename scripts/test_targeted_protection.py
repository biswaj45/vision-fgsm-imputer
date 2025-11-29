"""
Test and compare Global vs Targeted FGSM protection
"""

import sys
from pathlib import Path

# Setup paths
project_root = Path('/content/vision-fgsm-imputer') if Path('/content/vision-fgsm-imputer').exists() else Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image
import cv2

def find_or_upload_images():
    """Find existing images or prompt for upload."""
    from pathlib import Path
    
    # Search for existing images
    search_dirs = [
        Path('/content'),
        Path('/content/vision-fgsm-imputer'),
        Path('/content/vision-fgsm-imputer/test_images'),
    ]
    
    extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.png', '*.PNG']
    found = []
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for ext in extensions:
            found.extend(search_dir.glob(ext))
            found.extend(search_dir.glob(f'*/{ext}'))
    
    found = [f for f in found if 'sample_data' not in str(f)]
    found = list(set(found))
    
    if len(found) >= 2:
        print(f"\n✅ Found {len(found)} existing images:")
        for i, img in enumerate(found[:5]):
            print(f"   [{i}] {img.name}")
        
        # Smart selection
        pic2 = [f for f in found if 'PIC2' in f.name or 'pic2' in f.name.lower()]
        others = [f for f in found if f not in pic2]
        
        if pic2 and others:
            source = str(pic2[0])
            target = str(others[0])
        else:
            source = str(found[0])
            target = str(found[1]) if len(found) > 1 else str(found[0])
        
        print(f"\n📌 Using:")
        print(f"   SOURCE (identity): {Path(source).name}")
        print(f"   TARGET (to protect): {Path(target).name}")
        
        return source, target
    
    # No images found, upload
    print("\n📤 No images found. Please upload 2 images:")
    print("   1. SOURCE (identity donor)")
    print("   2. TARGET (person to protect)")
    
    try:
        from google.colab import files
        
        print("\n1️⃣ Upload SOURCE image (identity donor):")
        uploaded = files.upload()
        source = '/content/' + list(uploaded.keys())[0]
        
        print("\n2️⃣ Upload TARGET image (person to protect):")
        uploaded = files.upload()
        target = '/content/' + list(uploaded.keys())[0]
        
        print(f"\n✅ Images uploaded!")
        return source, target
    
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None, None

def main():
    print("="*80)
    print("TESTING: GLOBAL vs TARGETED FGSM PROTECTION")
    print("="*80)
    
    # Find or upload images
    source_image, target_image = find_or_upload_images()
    
    if not source_image or not target_image:
        print("❌ No images available. Exiting.")
        return
    
    # Import methods
    from inference.predict import NoiseImputer
    from inference.targeted_protect import TargetedNoiseImputer
    
    # Configuration
    model_path = str(project_root / 'saved_models' / 'best.pth')
    epsilon = 0.30
    
    # Load target image
    original = np.array(Image.open(target_image))
    
    # Run comparison
    print("\n" + "="*80)
    print("STEP 1: APPLYING PROTECTION METHODS")
    print("="*80)
    
    # Method 1: Global protection (current)
    print("\n1️⃣ GLOBAL PROTECTION (entire image):")
    global_imputer = NoiseImputer(model_path=model_path, epsilon=epsilon)
    protected_global = global_imputer.impute_from_array(original)
    
    # Method 2: Targeted protection (new)
    print("\n2️⃣ TARGETED PROTECTION (face contour lines + nose bridge):")
    targeted_imputer = TargetedNoiseImputer(
        model_path=model_path,
        epsilon=0.40,  # Can use higher epsilon on edge lines
        target_regions=['face_contour', 'jawline', 'nose_bridge'],
        feather_radius=25  # Heavy feathering for invisible edges
    )
    protected_targeted, mask_vis, region_info = targeted_imputer.impute_from_array(
        original,
        return_mask=True
    )
    
    # Compare visibility
    print("\n" + "="*80)
    print("VISIBILITY COMPARISON")
    print("="*80)
    
    diff_global = np.abs(original.astype(float) - protected_global.astype(float)).mean()
    diff_targeted = np.abs(original.astype(float) - protected_targeted.astype(float)).mean()
    
    print(f"Global protection - Avg pixel change: {diff_global:.2f}")
    print(f"Targeted protection - Avg pixel change: {diff_targeted:.2f}")
    print(f"Visibility reduction: {(1 - diff_targeted/diff_global)*100:.1f}%")
    
    # Save intermediate results
    output_dir = Path('/content')
    Image.fromarray(protected_global).save(output_dir / 'protected_global.jpg')
    Image.fromarray(protected_targeted).save(output_dir / 'protected_targeted.jpg')
    Image.fromarray(mask_vis).save(output_dir / 'protection_mask.jpg')
    
    print(f"\n✅ Protected images saved to {output_dir}")
    
    # Test with face swap
    print("\n" + "="*80)
    print("STEP 2: TESTING WITH SIMSWAP")
    print("="*80)
    
    from app.simswap_tester import SimSwapTester
    
    # Load SimSwap
    tester = SimSwapTester()
    success, msg = tester.load_model()
    
    if not success:
        print(f"❌ SimSwap loading failed: {msg}")
        return
    
    # Load source face
    if not Path(source_image).exists():
        print(f"❌ Source image not found: {source_image}")
        return
    
    source = cv2.imread(source_image)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    
    # Test 1: Original (no protection)
    print("\n1️⃣ Testing ORIGINAL (no protection)...")
    swap_orig, status_orig, metrics_orig = tester.test_manipulation(original, source)
    
    print(f"   {status_orig}")
    print(f"   Swap strength: {metrics_orig.get('swap_strength', 'N/A')}")
    print(f"   MSE: {metrics_orig.get('mse', 0):.2f}")
    
    # Test 2: Global protection
    print("\n2️⃣ Testing GLOBAL PROTECTION...")
    swap_global, status_global, metrics_global = tester.test_manipulation(protected_global, source)
    
    print(f"   {status_global}")
    print(f"   Swap strength: {metrics_global.get('swap_strength', 'N/A')}")
    print(f"   MSE: {metrics_global.get('mse', 0):.2f}")
    
    # Test 3: Targeted protection
    print("\n3️⃣ Testing TARGETED PROTECTION (eyes + nose only)...")
    swap_targeted, status_targeted, metrics_targeted = tester.test_manipulation(protected_targeted, source)
    
    print(f"   {status_targeted}")
    print(f"   Swap strength: {metrics_targeted.get('swap_strength', 'N/A')}")
    print(f"   MSE: {metrics_targeted.get('mse', 0):.2f}")
    
    # Comparison results
    print("\n" + "="*80)
    print("🎯 FINAL COMPARISON")
    print("="*80)
    
    print(f"\n{'Method':<25} {'Swap Strength':<15} {'MSE':<10} {'Visibility'}")
    print("-" * 70)
    print(f"{'Original':<25} {metrics_orig.get('swap_strength', 'N/A'):<15} {metrics_orig.get('mse', 0):<10.0f} N/A")
    print(f"{'Global Protection':<25} {metrics_global.get('swap_strength', 'N/A'):<15} {metrics_global.get('mse', 0):<10.0f} HIGH (hazy)")
    print(f"{'Targeted Protection':<25} {metrics_targeted.get('swap_strength', 'N/A'):<15} {metrics_targeted.get('mse', 0):<10.0f} LOW (natural)")
    
    # Verdict
    print("\n" + "="*80)
    print("💡 CONCLUSION")
    print("="*80)
    
    if metrics_targeted.get('swap_strength') in ['Failed', 'Weak']:
        print("✅ TARGETED PROTECTION WORKS!")
        print("   • Disrupts face swap effectively")
        print("   • Less visible to human eyes")
        print("   • Better user experience")
    elif metrics_global.get('swap_strength') == 'Weak' and metrics_targeted.get('swap_strength') == 'Strong':
        print("⚠️ TARGETED PROTECTION WEAKER")
        print("   • Global method more effective")
        print("   • But global is more visible")
        print("   • Consider hybrid approach")
    else:
        print("📊 MIXED RESULTS")
        print("   • Both methods have trade-offs")
        print("   • User should choose based on needs")
    
    # Save comparison grid
    print("\n📸 Creating visual comparison...")
    
    h = 300
    imgs = []
    labels = ['ORIGINAL', 'GLOBAL', 'TARGETED']
    
    for img, label in zip([original, protected_global, protected_targeted], labels):
        resized = cv2.resize(img, (h, h))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(resized, label, (10, 30), font, 0.8, (255, 255, 255), 2)
        imgs.append(resized)
    
    comparison_grid = np.hstack(imgs)
    output_path = '/content/protection_comparison_methods.jpg'
    comparison_bgr = cv2.cvtColor(comparison_grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, comparison_bgr)
    
    print(f"✅ Saved to: {output_path}")
    
    # Display in Colab
    try:
        from IPython.display import Image as IPImage, display
        display(IPImage(output_path))
    except:
        pass
    
    print("\n✅ TEST COMPLETE!")

if __name__ == "__main__":
    main()
