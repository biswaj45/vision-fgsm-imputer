# 🎯 Quick Reference Card

## Training vs Inference Strategy

```
┌──────────────────────────────────────────────────────────┐
│  📝 5-Day POC: FGSM Anti-Deepfake Protection System      │
└──────────────────────────────────────────────────────────┘

PHASE 1: TRAINING ⚡
├─ Device: GPU (Colab T4)
├─ Why: ~7x faster development
├─ Time: 2-3 min per epoch
├─ Config: mixed_precision = true
└─ Command: python training/train_unet.py

PHASE 2: INFERENCE 🚀
├─ Device: CPU (default)
├─ Why: Production-ready, cost-effective
├─ Speed: 200-300ms (target: <400ms) ✅
├─ Config: force_cpu = true (default)
└─ Command: python app/gradio_app.py --model_path model.pth
```

## Key Commands

### Training (Auto-detects GPU)
```bash
# Automatically uses T4 GPU if available
python training/train_unet.py

# Expected: 2-3 min per epoch on GPU
#          15-20 min per epoch on CPU
```

### Inference (Defaults to CPU)
```bash
# CPU inference (default, production-ready)
python app/gradio_app.py --model_path outputs/checkpoints/best.pth

# Optional: GPU inference (testing only)
python app/gradio_app.py --model_path outputs/checkpoints/best.pth --gpu
```

### Benchmarking
```bash
# Benchmark on current device (auto-detects)
python scripts/benchmark_inference.py --model_path outputs/checkpoints/best.pth
```

## Performance at a Glance

| Metric | GPU Training | CPU Inference |
|--------|--------------|---------------|
| Speed | 2-3 min/epoch | 200-300ms/image |
| Target | Fast dev | <400ms ✅ |
| Cost | Free (Colab) | Low (any CPU) |
| Use | Development | Production |

## Device Detection Logic

```python
# TRAINING (wants GPU)
device = 'auto'  # → uses GPU if available, else CPU

# INFERENCE (wants CPU for demo)
device = 'cpu'  # → always CPU for deployment
# OR
device = 'auto' with --gpu flag  # → optional GPU
```

## Verification Commands

```python
# Check if GPU is available
import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Expected on Colab T4:
# > CUDA: True
# > GPU: Tesla T4
```

## Quick Colab Setup

```python
# 1. Verify GPU
!nvidia-smi

# 2. Install
!pip install -q torch torchvision albumentations gradio opencv-python pyyaml tensorboard tqdm

# 3. Prepare data
!python scripts/prepare_dataset.py --dataset sample --output_dir ./data

# 4. Train (uses GPU automatically)
!python training/train_unet.py

# 5. Demo (uses CPU for deployment)
!python app/gradio_app.py --model_path outputs/checkpoints/best.pth --share
```

## Model Specifications

- **Size**: <8MB ✅
- **Parameters**: 4.8M (U-Net), 2.5M (Autoencoder) ✅
- **Input**: RGB (3, 256, 256)
- **Output**: Perturbation (3, 256, 256)
- **CPU Optimized**: <300ms inference ✅
- **GPU Compatible**: 2-3 min/epoch training ✅

## Troubleshooting

### "Out of Memory" during training
```bash
# Reduce batch size in config/training_config.yaml
batch_size: 16  # or 8
```

### "Slow inference" on CPU
```bash
# Check model size
ls -lh outputs/checkpoints/best.pth
# Should be ~19MB (with optimizer states)

# Expected: 200-300ms per image
# If slower, model may not be loaded correctly
```

### GPU not detected
```bash
# Verify in Colab: Runtime → Change runtime type → T4 GPU
!nvidia-smi

# Check in code
import torch
print(torch.cuda.is_available())  # Should be True
```

---

## 📚 Full Documentation

- **Complete Guide**: [README.md](README.md)
- **Colab Tutorial**: [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md)
- **Workflow Details**: [WORKFLOW.md](WORKFLOW.md)
- **Copilot Instructions**: [COPILOT_INSTRUCTIONS.md](COPILOT_INSTRUCTIONS.md)
