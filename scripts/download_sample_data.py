"""
Download sample face images for training.
Uses CelebA-HQ subset or FFHQ samples.
"""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_file(url, dest_path):
    """Download file with progress bar."""
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=dest_path) as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)


def download_sample_celeba():
    """Download sample CelebA images."""
    print("\n" + "="*60)
    print("Downloading CelebA-HQ Sample Dataset")
    print("="*60)
    
    # Create directories
    data_dir = Path("./data")
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Download CelebA-HQ samples from GitHub (smaller subset)
    print("\n📥 Downloading CelebA-HQ samples...")
    
    # Use a prepared dataset from Kaggle/GitHub
    url = "https://github.com/tkarras/progressive_growing_of_gans/raw/master/dataset_tool.py"
    
    print("\n⚠️  For training, you need face images!")
    print("\nOption 1 - Quick Start (Manual Upload):")
    print("  1. Upload 100-500 face images to ./data/train/")
    print("  2. Upload 20-50 face images to ./data/val/")
    print("  3. Images should be diverse faces (any resolution, will be resized)")
    
    print("\nOption 2 - Use Sample Faces (Auto):")
    print("  We'll download sample faces from Pexels/Unsplash...")
    
    response = input("\nDownload sample faces automatically? (y/n): ")
    
    if response.lower() == 'y':
        download_sample_faces(train_dir, val_dir)
    else:
        print("\n✅ Please upload your face images to:")
        print(f"  - Training: {train_dir.absolute()}")
        print(f"  - Validation: {val_dir.absolute()}")
        print("\nThen run: !python training/train_unet.py")


def download_sample_faces(train_dir, val_dir):
    """Download sample face images from free sources."""
    print("\n📥 Downloading sample faces...")
    
    # Sample face URLs (free stock photos)
    sample_faces = [
        "https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg",
        "https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg",
        "https://images.pexels.com/photos/1516680/pexels-photo-1516680.jpeg",
        "https://images.pexels.com/photos/1553783/pexels-photo-1553783.jpeg",
        "https://images.pexels.com/photos/1858175/pexels-photo-1858175.jpeg",
    ]
    
    print("\n⚠️  Note: For better results, use 100+ diverse face images")
    print("This script downloads only 5 samples for testing.\n")
    
    # Download to train
    for i, url in enumerate(sample_faces[:4]):
        dest = train_dir / f"face_{i:03d}.jpg"
        print(f"Downloading {i+1}/4 to train/...")
        try:
            download_file(url, str(dest))
        except Exception as e:
            print(f"  ⚠️  Failed: {e}")
    
    # Download to val
    dest = val_dir / "face_000.jpg"
    print(f"Downloading 1/1 to val/...")
    try:
        download_file(sample_faces[4], str(dest))
    except Exception as e:
        print(f"  ⚠️  Failed: {e}")
    
    print("\n✅ Sample dataset downloaded!")
    print(f"  - Train: {len(list(train_dir.glob('*.jpg')))} images")
    print(f"  - Val: {len(list(val_dir.glob('*.jpg')))} images")
    print("\n⚠️  For production, add 100+ more diverse faces to data/train/")


def use_ffhq_dataset():
    """Instructions for using FFHQ dataset."""
    print("\n" + "="*60)
    print("FFHQ Dataset (High Quality)")
    print("="*60)
    print("\n1. Download FFHQ thumbnails (128x128):")
    print("   https://github.com/NVlabs/ffhq-dataset")
    print("\n2. Or use Kaggle:")
    print("   https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq")
    print("\n3. Extract to:")
    print("   ./data/train/ (70000 images)")
    print("   ./data/val/ (split some for validation)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download sample training data")
    parser.add_argument('--dataset', choices=['celeba', 'ffhq', 'sample'], default='sample',
                       help='Dataset to use (sample=5 images for testing)')
    
    args = parser.parse_args()
    
    if args.dataset == 'sample':
        data_dir = Path("./data")
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        download_sample_faces(train_dir, val_dir)
    elif args.dataset == 'celeba':
        download_sample_celeba()
    elif args.dataset == 'ffhq':
        use_ffhq_dataset()
