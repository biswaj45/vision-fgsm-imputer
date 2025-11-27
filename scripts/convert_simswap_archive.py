"""
Convert SimSwap archive format to standard .pth file.
Run this in Colab after extracting arcface_checkpoint.tar
"""

import torch
from pathlib import Path

def convert_archive():
    """Convert PyTorch archive format to .pth"""
    
    # Paths - the actual model file to load
    # PyTorch saves models as a zip with archive/ subfolder containing data files
    model_dir = Path.home() / '.simswap' / 'checkpoints' / 'SimSwap' / 'arcface_model'
    archive_path = model_dir / 'archive'
    output_path = Path.home() / '.simswap' / 'checkpoints' / 'simswap_224.pth'
    
    print(f"Model directory: {model_dir}")
    print(f"Archive path: {archive_path}")
    
    if not archive_path.exists():
        print(f"❌ Archive not found at {archive_path}")
        print("\nExpected structure:")
        print("  ~/.simswap/checkpoints/SimSwap/arcface_model/")
        print("    └── archive/")
        print("        ├── data.pkl")
        print("        ├── data/")
        print("        │   ├── 0, 1, 2, ..., 795")
        print("        └── version")
        return False
    
    try:
        # PyTorch stores models in a zip format with an 'archive' folder
        # We can load by creating a zip file or by reconstructing the path
        
        print("Loading model (this is actually an ArcFace backbone for embeddings)...")
        
        # The archive folder structure is PyTorch's internal format
        # Let's try loading the parent tar/zip file if it exists
        tar_file = model_dir.parent / 'arcface_checkpoint.tar'
        
        if tar_file.exists():
            print(f"Loading from tar file: {tar_file}")
            # It's actually a PyTorch zip despite the .tar extension
            # Need weights_only=False for custom ArcFace model classes
            checkpoint = torch.load(tar_file, map_location='cpu', weights_only=False)
        else:
            # If no tar file, we need to reconstruct from the archive directory
            print("No tar file found, reconstructing from archive directory...")
            print("Note: This model is the ArcFace checkpoint for face embeddings.")
            
            # For SimSwap, the arcface_checkpoint.tar contains ArcFace model
            # We actually need to load it differently - it's not a direct state dict
            
            # Let's check what's in data.pkl
            import pickle
            data_pkl = archive_path / 'data.pkl'
            if data_pkl.exists():
                print(f"Reading data.pkl...")
                with open(data_pkl, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"Metadata type: {type(metadata)}")
                if isinstance(metadata, (list, tuple)):
                    print(f"Metadata length: {len(metadata)}")
                    print(f"First items: {metadata[:3] if len(metadata) > 3 else metadata}")
            
            print("\n⚠️ The extracted archive needs to be loaded as a zip file.")
            print("Please use the tar file directly or re-extract properly.")
            return False
        
        print(f"Model type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            print(f"Keys: {keys[:10]}...")  # Show first 10 keys
        
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
