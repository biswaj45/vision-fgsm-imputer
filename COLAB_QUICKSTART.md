# 🚀 Google Colab Quick Start Guide

## 🎯 Workflow Overview

This project follows a **hybrid GPU/CPU workflow**:

1. **📚 Training**: Use Colab T4 GPU for fast training (~7x speedup)
2. **🚀 Inference**: Use CPU for deployment (Gradio demo, <400ms target)

**Why this approach?**
- ✅ Training on GPU: Fast model development (2-3 min per epoch)
- ✅ Inference on CPU: Production-ready, cost-effective deployment
- ✅ Model stays tiny (<5M params): Works great on both GPU and CPU

---

## 📋 Setup in Colab

### 1. Check GPU Availability

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

**Expected Output on T4:**
```
CUDA Available: True
GPU: Tesla T4
Memory: 15.00 GB
```

### 2. Clone and Install

```python
# Clone repository (or upload files)
!git clone https://github.com/your-username/vision_fgsm_imputer.git
%cd vision_fgsm_imputer

# Install dependencies
!pip install -q torch torchvision albumentations gradio opencv-python pyyaml tensorboard tqdm
```

### 3. Prepare Sample Dataset

```python
# Create sample dataset for testing
!python scripts/prepare_dataset.py --dataset sample --output_dir ./data

# Verify dataset
!python scripts/prepare_dataset.py --verify --output_dir ./data
```

### 4. Train on T4 GPU

```python
# Train with GPU acceleration and mixed precision
!python training/train_unet.py

# Monitor with TensorBoard
%load_ext tensorboard
%tensorboard --logdir outputs/logs
```

**Training Configuration for T4:**
- Batch size: 16-32 (adjust based on GPU memory)
- Mixed precision: Enabled (2x faster training)
- Expected speed: ~200-300ms per batch
- Expected training time: ~1-2 hours for 50 epochs on sample dataset

### 5. Benchmark Performance

```python
# Benchmark GPU inference
!python scripts/benchmark_inference.py \
    --model_path outputs/checkpoints/best.pth \
    --num_images 100

# Expected results on T4:
# - Model load: ~50-100ms
# - Inference: ~10-30ms per image (GPU)
# - FPS: 30-100
```

## 🎨 Launch Gradio Demo

```python
# Launch demo (uses CPU for production-ready deployment)
!python app/gradio_app.py \
    --model_path outputs/checkpoints/best.pth \
    --share

# Expected: <400ms inference time on CPU
# Click the public URL to access the demo

# Optional: Force GPU for inference (testing only)
!python app/gradio_app.py \
    --model_path outputs/checkpoints/best.pth \
    --share \
    --gpu
```

## 💡 Quick Training Script

```python
from training.train_unet import Trainer
from training.dataset import create_dataloaders
from training.transforms import get_training_transforms, get_validation_transforms
import yaml

# Load config
with open('config/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Override for Colab T4
config['batch_size'] = 32  # Larger batch for GPU
config['mixed_precision'] = True  # Enable AMP
config['device'] = 'auto'  # Auto-detect GPU

# Create dataloaders
train_transform = get_training_transforms(256)
val_transform = get_validation_transforms(256)

train_loader, val_loader = create_dataloaders(
    train_dir='./data/train',
    val_dir='./data/val',
    train_transform=train_transform,
    val_transform=val_transform,
    batch_size=config['batch_size'],
    num_workers=2  # Optimal for Colab
)

# Train
trainer = Trainer(config, device='auto')
trainer.train(train_loader, val_loader, config['num_epochs'])

print("✅ Training complete!")
```

## 🔥 Quick Inference Script

```python
from inference.predict import impute_noise
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Upload your image or use sample
# For Colab: files.upload() or use sample
image_path = "path/to/your/image.jpg"

# Protect image with GPU acceleration
result = impute_noise(
    image_path=image_path,
    model_path="outputs/checkpoints/best.pth",
    output_path="protected.jpg",
    device='auto'  # Uses GPU automatically
)

# Display results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(Image.open(image_path))
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(result)
axes[1].set_title('Protected')
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

## 📊 Performance Comparison: Training vs Inference

### Training (GPU Recommended)
| Metric | CPU (Colab) | **T4 GPU (Colab)** | Speedup |
|--------|-------------|-------------------|---------|
| Per epoch | ~15-20 min | **~2-3 min** | **~7x faster** ⚡ |
| 50 epochs | ~12-16 hours | **~2 hours** | **~7x faster** |
| Memory | N/A | 2-4GB VRAM | Efficient |
| Mixed precision | ❌ | ✅ (2x boost) | 2x faster |

### Inference (CPU Optimized for Deployment)
| Metric | CPU (Production) | GPU (Optional) | Notes |
|--------|-----------------|----------------|-------|
| Per image | **200-300ms** ✅ | ~10-30ms | CPU meets <400ms target |
| 100 images | **~25s** | ~2s | CPU fast enough |
| Deployment cost | **Low** ✅ | High | CPU = cost-effective |
| Model size | 19MB | 19MB | Same model works everywhere |

**Recommendation**: 
- 🏋️ **Train on GPU** (T4) for speed
- 🚀 **Deploy on CPU** for cost-effectiveness and meets performance targets

## 🎯 GPU Optimization Tips

### 1. Enable Mixed Precision Training

```python
# In training_config.yaml
mixed_precision: true  # 2x faster on T4
```

### 2. Increase Batch Size

```python
# In training_config.yaml
batch_size: 32  # Larger batches utilize GPU better
```

### 3. Enable CUDNN Benchmark

```python
import torch
torch.backends.cudnn.benchmark = True  # Auto-enabled in code
```

### 4. Pin Memory

```python
# In DataLoader (already enabled)
pin_memory = True  # Faster CPU-to-GPU transfer
```

## 🐛 Troubleshooting

### Out of Memory Error

```python
# Reduce batch size
config['batch_size'] = 16  # or 8

# Clear cache
import torch
torch.cuda.empty_cache()
```

### Slow Training

```python
# Check if GPU is being used
import torch
print(f"Using device: {torch.cuda.current_device()}")

# Enable mixed precision
config['mixed_precision'] = True
```

### CUDA Out of Memory

```python
# Monitor GPU memory
!nvidia-smi

# Reduce batch size or image size
config['batch_size'] = 8
config['image_size'] = 128  # Smaller images
```

## 📦 Export Model for Deployment

```python
# Export to ONNX (CPU/GPU compatible)
!python scripts/export_model.py \
    --model_path outputs/checkpoints/best.pth \
    --format onnx torchscript \
    --optimize

# Exported models in ./exports/
```

## 🎓 Complete Colab Notebook

Here's a complete notebook cell sequence:

```python
# Cell 1: Setup
!pip install -q torch torchvision albumentations gradio opencv-python pyyaml tensorboard tqdm
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Cell 2: Prepare Data
!python scripts/prepare_dataset.py --dataset sample --output_dir ./data

# Cell 3: Train
!python training/train_unet.py

# Cell 4: Benchmark
!python scripts/benchmark_inference.py --model_path outputs/checkpoints/best.pth --num_images 100

# Cell 5: Launch Demo
!python app/gradio_app.py --model_path outputs/checkpoints/best.pth --share
```

## ⚡ Expected Performance on T4

- **Training**: 50 epochs in ~1-2 hours (with mixed precision)
- **Inference**: ~10-30ms per image (30-100 FPS)
- **Model Size**: 4.8M params, ~19MB with optimizer states
- **Memory Usage**: ~2-4GB VRAM during training
- **Batch Size**: Up to 64 for 256×256 images

## 🚀 Pro Tips

1. **Use Mixed Precision**: 2x speedup with no accuracy loss
2. **Batch Process**: Process multiple images together for better GPU utilization
3. **Pin Memory**: Faster data loading (already enabled)
4. **CUDNN Benchmark**: Auto-optimizes convolutions (already enabled)
5. **Monitor GPU**: Use `nvidia-smi` or `gpustat` to monitor usage

---

**Ready to go! Your anti-deepfake protection system is now GPU-optimized for Colab T4! 🎉**
