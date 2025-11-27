"""
Prepare dataset for training.
Downloads and organizes CelebA or VGGFace2 dataset.
"""

import os
import sys
from pathlib import Path
import shutil
import argparse
from tqdm import tqdm
import zipfile
import tarfile
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_directory_structure(root_dir: str):
    """
    Create standard directory structure.
    
    Args:
        root_dir: Root directory for dataset
    """
    dirs = [
        f"{root_dir}/train",
        f"{root_dir}/val",
        f"{root_dir}/test"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_samples: Optional[int] = None
):
    """
    Split dataset into train/val/test.
    
    Args:
        source_dir: Source directory with images
        output_dir: Output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        max_samples: Maximum number of samples to use
    """
    print(f"\nSplitting dataset from {source_dir}...")
    
    # Get all image files
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    
    for ext in extensions:
        image_files.extend(Path(source_dir).rglob(f"*{ext}"))
    
    image_files = sorted(image_files)
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    total = len(image_files)
    print(f"Found {total} images")
    
    # Calculate split sizes
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    
    print(f"Split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create directories
    create_directory_structure(output_dir)
    
    # Split files
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]
    
    # Copy files
    print("\nCopying files...")
    
    for files, split in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
        dst_dir = Path(output_dir) / split
        
        for src_path in tqdm(files, desc=f"Copying {split}"):
            dst_path = dst_dir / src_path.name
            shutil.copy2(src_path, dst_path)
    
    print("✓ Dataset split complete!")


def download_celeba(output_dir: str):
    """
    Instructions for downloading CelebA dataset.
    
    Args:
        output_dir: Output directory
    """
    print("\n" + "="*60)
    print("CELEBA DATASET DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\nCelebA dataset requires manual download due to authentication.")
    print("\nSteps:")
    print("1. Visit: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    print("2. Download 'Align&Cropped Images' (img_align_celeba.zip)")
    print("3. Extract the zip file")
    print("4. Run this script with --source_dir pointing to extracted folder")
    print("\nExample:")
    print(f"  python prepare_dataset.py --source_dir path/to/img_align_celeba --output_dir {output_dir}")
    print("="*60)


def download_vggface2_instructions(output_dir: str):
    """
    Instructions for downloading VGGFace2 dataset.
    
    Args:
        output_dir: Output directory
    """
    print("\n" + "="*60)
    print("VGGFACE2 DATASET DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\nVGGFace2 requires registration and manual download.")
    print("\nSteps:")
    print("1. Visit: https://github.com/ox-vgg/vgg_face2")
    print("2. Follow registration instructions")
    print("3. Download the dataset")
    print("4. Extract files")
    print("5. Run this script with --source_dir pointing to extracted folder")
    print("\nExample:")
    print(f"  python prepare_dataset.py --source_dir path/to/vggface2 --output_dir {output_dir}")
    print("="*60)


def verify_dataset(data_dir: str):
    """
    Verify dataset structure and contents.
    
    Args:
        data_dir: Dataset directory
    """
    print("\n" + "="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    total_images = 0
    
    for split in splits:
        split_dir = Path(data_dir) / split
        
        if not split_dir.exists():
            print(f"❌ Missing: {split_dir}")
            continue
        
        # Count images
        count = 0
        for ext in extensions:
            count += len(list(split_dir.glob(f"*{ext}")))
        
        total_images += count
        print(f"✓ {split}: {count} images")
    
    print(f"\nTotal: {total_images} images")
    
    if total_images > 0:
        print("✅ Dataset verification passed!")
    else:
        print("❌ No images found!")
    
    print("="*60)


def create_sample_dataset(output_dir: str, num_samples: int = 500):
    """
    Create a small sample dataset with synthetic images for testing.
    
    Args:
        output_dir: Output directory
        num_samples: Number of synthetic samples per split
    """
    import cv2
    import numpy as np
    
    print(f"\nCreating sample dataset with {num_samples} images per split...")
    
    create_directory_structure(output_dir)
    
    for split in ['train', 'val', 'test']:
        split_dir = Path(output_dir) / split
        
        for i in tqdm(range(num_samples), desc=f"Creating {split}"):
            # Generate random "face-like" image
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            
            # Add some structure (circles to simulate face)
            center = (128, 128)
            cv2.circle(img, center, 80, (255, 200, 180), -1)  # Face
            cv2.circle(img, (100, 110), 15, (100, 100, 150), -1)  # Eye
            cv2.circle(img, (156, 110), 15, (100, 100, 150), -1)  # Eye
            cv2.ellipse(img, (128, 160), (30, 15), 0, 0, 180, (180, 100, 100), -1)  # Mouth
            
            # Save
            img_path = split_dir / f"sample_{i:04d}.jpg"
            cv2.imwrite(str(img_path), img)
    
    print("✓ Sample dataset created!")
    verify_dataset(output_dir)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['celeba', 'vggface2', 'custom', 'sample'],
        default='sample',
        help='Dataset type'
    )
    parser.add_argument(
        '--source_dir',
        type=str,
        help='Source directory with images (for custom dataset)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data',
        help='Output directory for organized dataset'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Training set ratio'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Validation set ratio'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.1,
        help='Test set ratio'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        help='Maximum number of samples to use (for splitting existing datasets)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=500,
        help='Number of samples per split for sample dataset (default: 500)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify existing dataset'
    )
    
    args = parser.parse_args()
    
    # Verify mode
    if args.verify:
        verify_dataset(args.output_dir)
        return
    
    # Handle different dataset types
    if args.dataset == 'celeba':
        if args.source_dir and Path(args.source_dir).exists():
            split_dataset(
                args.source_dir,
                args.output_dir,
                args.train_ratio,
                args.val_ratio,
                args.test_ratio,
                args.max_samples
            )
            verify_dataset(args.output_dir)
        else:
            download_celeba(args.output_dir)
    
    elif args.dataset == 'vggface2':
        if args.source_dir and Path(args.source_dir).exists():
            split_dataset(
                args.source_dir,
                args.output_dir,
                args.train_ratio,
                args.val_ratio,
                args.test_ratio,
                args.max_samples
            )
            verify_dataset(args.output_dir)
        else:
            download_vggface2_instructions(args.output_dir)
    
    elif args.dataset == 'custom':
        if not args.source_dir:
            print("Error: --source_dir required for custom dataset")
            return
        
        if not Path(args.source_dir).exists():
            print(f"Error: Source directory not found: {args.source_dir}")
            return
        
        split_dataset(
            args.source_dir,
            args.output_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.max_samples
        )
        verify_dataset(args.output_dir)
    
    elif args.dataset == 'sample':
        create_sample_dataset(args.output_dir, num_samples=args.num_samples)
    
    print("\n✓ Dataset preparation complete!")


if __name__ == "__main__":
    main()
