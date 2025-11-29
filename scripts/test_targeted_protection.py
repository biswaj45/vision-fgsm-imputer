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

def main():
    print("="*80)
    print("TESTING: GLOBAL vs TARGETED FGSM PROTECTION")
    print("="*80)
    
    # Import methods
    from inference.predict import NoiseImputer
    from inference.targeted_protect import TargetedNoiseImputer, compare_global_vs_targeted
    
    # Configuration
    model_path = str(project_root / 'models' / 'best.pth')
    target_image = '/content/987568892057a9c58344cd6086a4d26e.jpg'
    epsilon = 0.30
    
    # Run comparison
    print("\n📊 Running comparison...")
    protected_global, protected_targeted, mask_vis = compare_global_vs_targeted(
        target_image,
        model_path,
        epsilon
    )
    
    # Test with face swap
    print("\n" + "="*80)
    print("TESTING WITH SIMSWAP")
    print("="*80)
    
    from app.simswap_tester import SimSwapTester
    
    # Load SimSwap
    tester = SimSwapTester()
    success, msg = tester.load_model()
    
    if not success:
        print(f"❌ SimSwap loading failed: {msg}")
        return
    
    # Load source face
    source_path = '/content/PIC2.JPG'
    if not Path(source_path).exists():
        print(f"⚠️  Source image not found: {source_path}")
        print("Please upload PIC2.JPG or adjust source_path")
        return
    
    source = cv2.imread(source_path)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    
    # Test 1: Original (no protection)
    print("\n1️⃣ Testing ORIGINAL (no protection)...")
    original = np.array(Image.open(target_image))
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
