"""
Check if model exists in GitHub repository without cloning entire repo.
"""

import requests
import sys

def check_model_in_github(
    owner: str = "biswaj45",
    repo: str = "vision-fgsm-imputer",
    file_path: str = "saved_models/best.pth"
):
    """
    Check if model file exists in GitHub repository.
    
    Args:
        owner: GitHub username
        repo: Repository name
        file_path: Path to model file in repo
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    
    print(f"🔍 Checking GitHub repo: {owner}/{repo}")
    print(f"📁 Looking for: {file_path}\n")
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            size_bytes = data.get('size', 0)
            size_mb = size_bytes / (1024 * 1024)
            
            print("✅ Model found in GitHub!")
            print(f"📦 File: {data['name']}")
            print(f"📏 Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
            print(f"🔗 Download URL: {data['download_url']}")
            print(f"🌐 GitHub URL: https://github.com/{owner}/{repo}/blob/main/{file_path}")
            
            return True
            
        elif response.status_code == 404:
            print("❌ Model NOT found in GitHub repository")
            print(f"\nThe file '{file_path}' does not exist in the repo.")
            print("\nPossible reasons:")
            print("  1. Model not pushed yet - run: git push")
            print("  2. Wrong file path - check saved_models/ directory")
            print("  3. Push failed due to authentication issues")
            
            return False
            
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")
            print(f"Message: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False


def download_model_from_github(
    owner: str = "biswaj45",
    repo: str = "vision-fgsm-imputer",
    file_path: str = "saved_models/best.pth",
    output_path: str = "outputs/checkpoints/best.pth"
):
    """
    Download model directly from GitHub.
    
    Args:
        owner: GitHub username
        repo: Repository name
        file_path: Path to model file in repo
        output_path: Local path to save model
    """
    from pathlib import Path
    
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}"
    
    print(f"⬇️  Downloading model from GitHub...")
    print(f"   {url}\n")
    
    try:
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            total_mb = total_size / (1024 * 1024)
            
            print(f"📦 Size: {total_mb:.2f} MB")
            
            # Create output directory
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress
            downloaded = 0
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Show progress
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r⏳ Progress: {progress:.1f}%", end='', flush=True)
            
            print(f"\n✅ Model downloaded successfully!")
            print(f"📁 Saved to: {output_path}")
            
            return True
            
        elif response.status_code == 404:
            print("❌ Model not found in GitHub")
            return False
        else:
            print(f"❌ Download failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check and download model from GitHub")
    parser.add_argument('--action', choices=['check', 'download'], default='check',
                       help='Action to perform')
    parser.add_argument('--owner', default='biswaj45',
                       help='GitHub username')
    parser.add_argument('--repo', default='vision-fgsm-imputer',
                       help='Repository name')
    parser.add_argument('--file_path', default='saved_models/best.pth',
                       help='Path to model in repo')
    parser.add_argument('--output_path', default='outputs/checkpoints/best.pth',
                       help='Local path to save downloaded model')
    
    args = parser.parse_args()
    
    if args.action == 'check':
        check_model_in_github(args.owner, args.repo, args.file_path)
    else:
        download_model_from_github(args.owner, args.repo, args.file_path, args.output_path)
