"""
Download and verify the FGSM protection model
"""

import os
import requests
from pathlib import Path

def download_model():
    """Download best.pth model from GitHub."""
    
    model_dir = Path('/content/vision-fgsm-imputer/models')
    model_path = model_dir / 'best.pth'
    
    # Remove corrupted file if exists
    if model_path.exists():
        print(f"⚠️  Removing existing file: {model_path}")
        model_path.unlink()
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    url = 'https://github.com/biswaj45/vision-fgsm-imputer/raw/main/models/best.pth'
    
    print(f"📥 Downloading model from GitHub...")
    print(f"   URL: {url}")
    print(f"   Destination: {model_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"   Size: {total_size / 1024**2:.2f} MB")
        
        with open(model_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\r   Progress: {progress:.1f}%", end='')
        
        print(f"\n✅ Downloaded successfully!")
        
        # Verify file
        actual_size = model_path.stat().st_size
        print(f"\n📊 Verification:")
        print(f"   File size: {actual_size / 1024**2:.2f} MB")
        
        if actual_size < 1024 * 1024:  # Less than 1 MB
            print(f"❌ File too small! Likely corrupted or HTML error page.")
            model_path.unlink()
            return False
        
        # Try loading with torch
        import torch
        print(f"   Testing torch.load...")
        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
        
        print(f"✅ Model is valid!")
        print(f"   Keys: {list(checkpoint.keys())}")
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        if 'model_state_dict' in checkpoint:
            print(f"   Model params: {len(checkpoint['model_state_dict'])} tensors")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        if model_path.exists():
            model_path.unlink()
        return False

if __name__ == "__main__":
    print("="*80)
    print("MODEL DOWNLOAD & VERIFICATION")
    print("="*80)
    
    success = download_model()
    
    if success:
        print("\n✅ Model ready! You can now run:")
        print("   !python /content/vision-fgsm-imputer/scripts/test_targeted_protection.py")
    else:
        print("\n❌ Model download failed. Please check:")
        print("   1. Internet connection")
        print("   2. GitHub repository access")
        print("   3. File availability at: https://github.com/biswaj45/vision-fgsm-imputer/tree/main/models")
