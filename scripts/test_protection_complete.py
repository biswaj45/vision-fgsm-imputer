"""
Complete FGSM protection test: Apply protection and test face swap on both original and protected images
"""

import sys
from pathlib import Path
sys.path.insert(0, '/content/vision-fgsm-imputer')

import cv2
import numpy as np
from PIL import Image

def main():
    print("="*80)
    print("FGSM ANTI-DEEPFAKE PROTECTION TEST")
    print("="*80)
    
    # Paths
    source_path = '/content/PIC2.JPG'  # Person A (identity to steal)
    target_path = '/content/987568892057a9c58344cd6086a4d26e.jpg'  # Person B (to protect)
    
    # Step 1: Apply FGSM protection
    print("\n1. Applying FGSM protection to TARGET image...")
    from inference.predict import NoiseImputer
    
    # Initialize protector
    imputer = NoiseImputer(
        model_path='/content/vision-fgsm-imputer/saved_models/best.pth',
        epsilon=0.15
    )
    
    target_img = Image.open(target_path)
    target_np = np.array(target_img)
    
    # Apply protection using impute_from_array
    protected_np = imputer.impute_from_array(target_np, return_perturbation=False)
    
    protected_path = '/content/target_protected.jpg'
    Image.fromarray(protected_np).save(protected_path)
    print(f"✅ FGSM protection applied (epsilon=0.15)")
    print(f"✅ Protected image saved to: {protected_path}")
    
    # Step 2: Load SimSwap
    print("\n2. Loading SimSwap model...")
    from app.simswap_tester import SimSwapTester
    
    tester = SimSwapTester()
    success, load_msg = tester.load_model()
    
    if not success:
        print(f"❌ {load_msg}")
        return
    
    print(f"✅ {load_msg}")
    
    # Step 3: Test on ORIGINAL target
    print("\n" + "="*80)
    print("3. Testing face swap on ORIGINAL target (baseline)")
    print("="*80)
    
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    
    swapped_orig, status_orig, metrics_orig = tester.test_manipulation(target, source)
    
    print(f"\nStatus: {status_orig}")
    print("\nMetrics (ORIGINAL):")
    for key, value in metrics_orig.items():
        print(f"   {key}: {value}")
    
    if swapped_orig is not None:
        swapped_orig_path = '/content/swapped_original.jpg'
        swapped_bgr = cv2.cvtColor(swapped_orig, cv2.COLOR_RGB2BGR)
        cv2.imwrite(swapped_orig_path, swapped_bgr)
        print(f"\n✅ Saved to: {swapped_orig_path}")
    
    # Step 4: Test on PROTECTED target
    print("\n" + "="*80)
    print("4. Testing face swap on PROTECTED target (with FGSM defense)")
    print("="*80)
    
    target_protected = cv2.imread(protected_path)
    target_protected = cv2.cvtColor(target_protected, cv2.COLOR_BGR2RGB)
    
    swapped_prot, status_prot, metrics_prot = tester.test_manipulation(target_protected, source)
    
    print(f"\nStatus: {status_prot}")
    print("\nMetrics (PROTECTED):")
    for key, value in metrics_prot.items():
        print(f"   {key}: {value}")
    
    if swapped_prot is not None:
        swapped_prot_path = '/content/swapped_protected.jpg'
        swapped_bgr = cv2.cvtColor(swapped_prot, cv2.COLOR_RGB2BGR)
        cv2.imwrite(swapped_prot_path, swapped_bgr)
        print(f"\n✅ Saved to: {swapped_prot_path}")
    
    # Step 5: Comparison
    print("\n" + "="*80)
    print("5. PROTECTION EFFECTIVENESS ANALYSIS")
    print("="*80)
    
    print("\n📊 Comparison:")
    print("-" * 80)
    print(f"{'Metric':<25} {'ORIGINAL':<25} {'PROTECTED':<25} {'Change':<25}")
    print("-" * 80)
    
    for key in ['mse', 'psnr', 'corruption_detected', 'swap_strength']:
        if key in metrics_orig and key in metrics_prot:
            orig_val = metrics_orig[key]
            prot_val = metrics_prot[key]
            
            if isinstance(orig_val, bool):
                change = '✅ PROTECTED' if prot_val and not orig_val else '❌ NOT PROTECTED'
            elif isinstance(orig_val, (int, float)):
                diff = prot_val - orig_val
                change = f"{diff:+.2f}"
            else:
                change = f"{orig_val} → {prot_val}"
            
            print(f"{key:<25} {str(orig_val):<25} {str(prot_val):<25} {change:<25}")
    
    print("-" * 80)
    
    # Final verdict
    print("\n🎯 VERDICT:")
    if metrics_prot.get('corruption_detected', False):
        print("✅ SUCCESS: Protection DETECTED as corrupted!")
        print("   The FGSM perturbations successfully degraded the face swap.")
    elif metrics_prot.get('mse', 0) > metrics_orig.get('mse', 0) * 1.5:
        print("✅ SUCCESS: Protection significantly degraded swap quality!")
        print(f"   MSE increased by {(metrics_prot['mse']/metrics_orig['mse']-1)*100:.1f}%")
    elif metrics_prot.get('psnr', 100) < metrics_orig.get('psnr', 100) * 0.8:
        print("✅ PARTIAL SUCCESS: Protection reduced swap quality")
        print(f"   PSNR decreased by {(1-metrics_prot['psnr']/metrics_orig['psnr'])*100:.1f}%")
    else:
        print("⚠️  LIMITED EFFECT: Protection had minimal impact")
        print("   Consider increasing epsilon or boost parameters")
    
    # Display results
    print("\n📸 Visual Comparison:")
    try:
        from IPython.display import Image as IPImage, display
        
        # Create comparison grid
        imgs_to_show = []
        if swapped_orig is not None:
            imgs_to_show.append(('ORIGINAL SWAP', swapped_orig))
        if swapped_prot is not None:
            imgs_to_show.append(('PROTECTED SWAP', swapped_prot))
        
        if len(imgs_to_show) == 2:
            # Stack side by side
            h = max(img.shape[0] for _, img in imgs_to_show)
            resized = []
            for label, img in imgs_to_show:
                scale = 400 / img.shape[0]
                new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
                resized.append(cv2.resize(img, (new_w, new_h)))
            
            comparison = np.hstack(resized)
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, 'ORIGINAL SWAP', (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(comparison, 'PROTECTED SWAP', (resized[0].shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
            
            comparison_path = '/content/protection_comparison.jpg'
            comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
            cv2.imwrite(comparison_path, comparison_bgr)
            
            print(f"✅ Comparison saved to: {comparison_path}")
            display(IPImage(comparison_path))
        
    except Exception as e:
        print(f"⚠️  Could not display images: {e}")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
