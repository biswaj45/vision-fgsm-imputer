"""
Save trained model to GitHub repository.
Useful for persisting model across Colab sessions.
"""

import os
import sys
from pathlib import Path
import subprocess
import argparse

def save_model_to_git(
    model_path: str = "outputs/checkpoints/best.pth",
    commit_message: str = "Save trained model checkpoint"
):
    """
    Save model to git and push to GitHub.
    
    Args:
        model_path: Path to model file
        commit_message: Git commit message
    """
    model_file = Path(model_path)
    
    if not model_file.exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    # Check file size
    size_mb = model_file.stat().st_size / (1024 * 1024)
    print(f"\n📦 Model size: {size_mb:.2f} MB")
    
    if size_mb > 100:
        print("⚠️  Warning: GitHub has 100MB file size limit!")
        print("   Consider using Git LFS or alternative storage")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    try:
        # Check if git repo
        result = subprocess.run(
            ['git', 'status'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print("❌ Not a git repository. Initialize with: git init")
            return False
        
        print("\n🔧 Preparing to save model to git...")
        
        # Create models directory if doesn't exist in repo
        saved_models_dir = Path("saved_models")
        saved_models_dir.mkdir(exist_ok=True)
        
        # Copy model to saved_models directory
        import shutil
        dest_path = saved_models_dir / model_file.name
        shutil.copy2(model_path, dest_path)
        print(f"✅ Copied model to: {dest_path}")
        
        # Git add
        subprocess.run(['git', 'add', str(dest_path)], check=True)
        print("✅ Added to git staging")
        
        # Git commit
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        print("✅ Committed to git")
        
        # Git push
        print("\n⬆️  Pushing to GitHub...")
        result = subprocess.run(['git', 'push'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model successfully pushed to GitHub!")
            print(f"\n📍 Model saved as: {dest_path}")
            print(f"   You can now fetch it with: git pull")
            return True
        else:
            print(f"❌ Git push failed: {result.stderr}")
            print("\nTry manually:")
            print(f"  git push origin main")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def download_model_from_git(
    model_name: str = "best.pth",
    output_path: str = "outputs/checkpoints/best.pth"
):
    """
    Download model from git repo.
    
    Args:
        model_name: Name of model file
        output_path: Where to save downloaded model
    """
    source_path = Path("saved_models") / model_name
    
    if not source_path.exists():
        print(f"❌ Model not found in saved_models: {model_name}")
        print("\nAvailable models:")
        saved_models_dir = Path("saved_models")
        if saved_models_dir.exists():
            models = list(saved_models_dir.glob("*.pth"))
            if models:
                for model in models:
                    size_mb = model.stat().st_size / (1024 * 1024)
                    print(f"  - {model.name} ({size_mb:.2f} MB)")
            else:
                print("  (none)")
        return False
    
    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy model
    import shutil
    shutil.copy2(source_path, output_file)
    
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✅ Model downloaded: {output_path} ({size_mb:.2f} MB)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save/load model to/from GitHub")
    parser.add_argument('--action', choices=['save', 'load'], default='save',
                       help='Action to perform')
    parser.add_argument('--model_path', default='outputs/checkpoints/best.pth',
                       help='Path to model file (for save)')
    parser.add_argument('--model_name', default='best.pth',
                       help='Model name in saved_models (for load)')
    parser.add_argument('--output_path', default='outputs/checkpoints/best.pth',
                       help='Output path (for load)')
    parser.add_argument('--message', default='Save trained model checkpoint',
                       help='Commit message (for save)')
    
    args = parser.parse_args()
    
    if args.action == 'save':
        save_model_to_git(args.model_path, args.message)
    else:
        download_model_from_git(args.model_name, args.output_path)
