"""
Test stronger protection (epsilon=0.30) and compare results
"""

import sys
import os

# Set up paths
project_root = '/content/vision-fgsm-imputer'
sys.path.insert(0, project_root)
os.chdir(project_root)

import torch
import cv2
import numpy as np
from PIL import Image

def main():
    print("="*80)
    print("TESTING STRONGER PROTECTION (epsilon=0.30)")
    print("="*80)
    
    # Import after path setup
    from inference.predict import NoiseImputer
    from app.simswap_tester import SimSwapTester
    
    # Step 1: Apply stronger protection
    print("\n1. Applying STRONG protection (epsilon=0.30)...")
    
    imputer = NoiseImputer(
        model_path=f'{project_root}/saved_models/best.pth',
        epsilon=0.30
    )
    
    target_path = '/content/987568892057a9c58344cd6086a4d26e.jpg'
    target = np.array(Image.open(target_path))
    
    protected_strong = imputer.impute_from_array(target)
    protected_strong_path = '/content/target_protected_strong_0.30.jpg'
    Image.fromarray(protected_strong).save(protected_strong_path)
    
    print(f"✅ Strong protection applied and saved to: {protected_strong_path}")
    
    # Step 2: Test face swap on strong protection
    print("\n2. Testing face swap on STRONGLY protected image...")
    
    source_path = '/content/PIC2.JPG'
    source = cv2.imread(source_path)
    target_strong = cv2.imread(protected_strong_path)
    
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    target_strong = cv2.cvtColor(target_strong, cv2.COLOR_BGR2RGB)
    
    tester = SimSwapTester()
    tester.load_model()
    
    swapped_strong, status_strong, metrics_strong = tester.test_manipulation(target_strong, source)
    
    print(f"\n{status_strong}")
    print("\n📊 Metrics with STRONG protection (epsilon=0.30):")
    for k, v in metrics_strong.items():
        print(f"   {k}: {v}")
    
    if swapped_strong is not None:
        result_path = '/content/swapped_strong_protection_0.30.jpg'
        swapped_bgr = cv2.cvtColor(swapped_strong, cv2.COLOR_RGB2BGR)
        cv2.imwrite(result_path, swapped_bgr)
        print(f"\n✅ Result saved to: {result_path}")
    
    # Step 3: Load and compare with previous results
    print("\n" + "="*80)
    print("COMPARISON: epsilon=0.15 vs epsilon=0.30")
    print("="*80)
    
    # Previous results (from earlier test)
    print("\nPrevious results:")
    print("  ORIGINAL (no protection):")
    print("    - Swap strength: Strong")
    print("    - MSE: 1945.97")
    print("    - PSNR: 15.24")
    print("    - Corruption: False")
    
    print("\n  PROTECTED (epsilon=0.15):")
    print("    - Swap strength: Weak")
    print("    - MSE: 251.64")
    print("    - PSNR: 24.12")
    print("    - Corruption: False")
    
    print(f"\n  PROTECTED STRONG (epsilon=0.30):")
    print(f"    - Swap strength: {metrics_strong.get('swap_strength', 'N/A')}")
    print(f"    - MSE: {metrics_strong.get('mse', 0):.2f}")
    print(f"    - PSNR: {metrics_strong.get('psnr', 0):.2f}")
    print(f"    - Corruption: {metrics_strong.get('corruption_detected', False)}")
    
    # Verdict
    print("\n" + "="*80)
    print("🎯 VERDICT:")
    print("="*80)
    
    if metrics_strong.get('corruption_detected', False):
        print("✅ EXCELLENT: Corruption detected with epsilon=0.30!")
    elif metrics_strong.get('swap_strength') in ['Failed', 'Weak']:
        print("✅ SUCCESS: Strong protection degraded face swap significantly!")
    else:
        print("⚠️  Try even stronger: epsilon=0.50 (extreme)")
    
    # Display
    print("\n📸 Visual comparison:")
    try:
        from IPython.display import Image as IPImage, display
        
        # Show side-by-side
        print("Displaying results...")
        display(IPImage(result_path))
        
        # Download
        from google.colab import files
        files.download(result_path)
        files.download(protected_strong_path)
        
    except Exception as e:
        print(f"⚠️  Display/download error: {e}")
    
    print("\n✅ TEST COMPLETE!")

if __name__ == "__main__":
    main()
