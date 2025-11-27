"""
Colab Setup Script - Install all dependencies and setup environment.
Run this first in Google Colab.
"""

# Install dependencies
print("📦 Installing dependencies...")
print("This may take 2-3 minutes...")

# Core dependencies
!pip install -q torch torchvision numpy opencv-python Pillow
!pip install -q albumentations tqdm pyyaml tensorboard matplotlib
!pip install -q gradio

# Face swapping dependencies
print("\n🎭 Installing face swapping tools...")
!pip install -q insightface
!pip install -q onnxruntime-gpu  # Use onnxruntime if no GPU
!pip install -q gdown

# Additional utilities
!pip install -q onnx

print("\n✅ All dependencies installed!")
print("\n📥 Now clone the repository:")
print("!git clone https://github.com/biswaj45/vision-fgsm-imputer.git")
print("%cd vision-fgsm-imputer")
