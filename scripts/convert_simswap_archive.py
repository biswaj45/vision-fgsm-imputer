"""
Convert SimSwap archive format to standard .pth file.
Run this in Colab after extracting arcface_checkpoint.tar
"""

import torch
from pathlib import Path

def convert_archive():
    """Convert PyTorch archive format to .pth"""
    
    # Paths
    archive_path = Path.home() / '.simswap' / 'checkpoints' / 'SimSwap' / 'arcface_model' / 'archive'
    output_path = Path.home() / '.simswap' / 'checkpoints' / 'simswap_224.pth'
    
    print(f"Loading from: {archive_path}")
    
    if not archive_path.exists():
        print(f"❌ Archive not found at {archive_path}")
        print("\nExpected structure:")
        print("  ~/.simswap/checkpoints/SimSwap/arcface_model/archive/")
        print("    ├── data.pkl")
        print("    ├── data/")
        print("    │   ├── 0, 1, 2, ..., 795")
        print("    └── version")
        return False
    
    try:
        # Load the archive format (PyTorch pickle)
        print("Loading model...")
        checkpoint = torch.load(str(archive_path), map_location='cpu')
        
        print(f"Model type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"Keys: {checkpoint.keys()}")
        
        # Save as standard .pth
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving to: {output_path}")
        torch.save(checkpoint, str(output_path))
        
        # Verify
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model converted successfully!")
        print(f"   Size: {file_size_mb:.2f} MB")
        print(f"   Location: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("="*60)
    print("SimSwap Archive to .pth Converter")
    print("="*60)
    success = convert_archive()
    exit(0 if success else 1)
