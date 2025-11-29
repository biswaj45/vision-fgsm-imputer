"""
Setup SimSwap models for Gradio app.
Downloads models to the correct locations.
"""

import os
from pathlib import Path
import requests
from tqdm import tqdm

def download_file(url, dest_path):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

def main():
    """Setup SimSwap models."""
    print("="*60)
    print("SimSwap Model Setup")
    print("="*60)
    
    # Get base directory
    base_dir = Path(__file__).parent.parent
    
    # Model paths
    arcface_dir = base_dir / 'arcface_model'
    generator_dir = base_dir / 'checkpoints' / 'people'
    
    arcface_path = arcface_dir / 'arcface_checkpoint.tar'
    generator_path = generator_dir / 'latest_net_G.pth'
    
    print("\n📦 Required models:")
    print(f"1. ArcFace: {arcface_path}")
    print(f"2. Generator: {generator_path}")
    
    # Check existing models
    models_exist = arcface_path.exists() and generator_path.exists()
    
    if models_exist:
        print("\n✅ All models already downloaded!")
        print("\nModel sizes:")
        print(f"  ArcFace: {arcface_path.stat().st_size / 1024**2:.1f} MB")
        print(f"  Generator: {generator_path.stat().st_size / 1024**2:.1f} MB")
        return
    
    print("\n⚠️  Models not found. Manual download required.")
    print("\n" + "="*60)
    print("DOWNLOAD INSTRUCTIONS")
    print("="*60)
    
    print("\n1️⃣ ArcFace Model (~200 MB):")
    print("   Download from: https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view")
    print(f"   Save to: {arcface_path}")
    
    print("\n2️⃣ SimSwap Generator (~210 MB):")
    print("   Download from: https://drive.google.com/file/d/1TY2YSajIx-Zqwqj_IZ0rIh7LfSIsrV_V/view")
    print(f"   Save to: {generator_path}")
    
    print("\n" + "="*60)
    print("OR use gdown to download directly:")
    print("="*60)
    print("\n# Install gdown if needed:")
    print("!pip install gdown")
    print("\n# Download ArcFace:")
    print(f"!gdown 1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn -O {arcface_path}")
    print("\n# Download Generator:")
    print(f"!gdown 1TY2YSajIx-Zqwqj_IZ0rIh7LfSIsrV_V -O {generator_path}")
    
    print("\n" + "="*60)
    
    # Attempt auto-download using gdown
    try:
        import gdown
        print("\n🚀 Attempting automatic download with gdown...")
        
        if not arcface_path.exists():
            print("\nDownloading ArcFace...")
            arcface_dir.mkdir(parents=True, exist_ok=True)
            gdown.download(
                'https://drive.google.com/uc?id=1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn',
                str(arcface_path),
                quiet=False
            )
            print("✅ ArcFace downloaded!")
        
        if not generator_path.exists():
            print("\nDownloading Generator...")
            generator_dir.mkdir(parents=True, exist_ok=True)
            gdown.download(
                'https://drive.google.com/uc?id=1TY2YSajIx-Zqwqj_IZ0rIh7LfSIsrV_V',
                str(generator_path),
                quiet=False
            )
            print("✅ Generator downloaded!")
        
        print("\n✅ All models downloaded successfully!")
        
    except ImportError:
        print("\n⚠️  gdown not installed. Install with: pip install gdown")
    except Exception as e:
        print(f"\n❌ Auto-download failed: {e}")
        print("Please download manually using the instructions above.")

if __name__ == "__main__":
    main()
