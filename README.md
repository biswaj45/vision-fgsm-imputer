# 🛡️ Vision FGSM Imputer - Anti-Deepfake Protection

A lightweight, fast CPU-optimized neural network system for protecting images against deepfake manipulation using FGSM (Fast Gradient Sign Method) perturbations.

## 🎯 Features

- **Hybrid GPU/CPU Workflow**: 
  - 🏋️ **Training**: GPU accelerated (Colab T4) with mixed precision - **~7x faster**
  - 🚀 **Inference**: CPU optimized for deployment - **<300ms per image**
- **Tiny Models**: <5M parameters, <8MB file size
- **Two Architecture Options**:
  - Tiny U-Net (~4.8M parameters)
  - Lightweight Autoencoder (~2.5M parameters)
- **FGSM Protection**: Adds imperceptible perturbations that disrupt AI-based manipulation
- **Mixed Precision Training**: 2x faster training on GPU with AMP
- **Web Interface**: Easy-to-use Gradio application (CPU-optimized)
- **Production Ready**: Fast CPU inference for real-world deployment

## 📁 Repository Structure

```
vision_fgsm_imputer/
│
├── models/                    # Model architectures
│   ├── unet_tiny.py          # Tiny U-Net implementation
│   ├── autoencoder.py        # Lightweight autoencoder
│   └── __init__.py
│
├── training/                  # Training components
│   ├── dataset.py            # Dataset loader
│   ├── train_unet.py         # Training script
│   ├── transforms.py         # Albumentations transforms
│   └── utils.py              # FGSM utilities
│
├── inference/                 # Inference pipeline
│   ├── predict.py            # Main prediction API
│   ├── perturb.py            # Perturbation generation
│   └── postprocess.py        # Post-processing utilities
│
├── app/                       # Web application
│   ├── gradio_app.py         # Gradio interface
│   ├── demo_utils.py         # Demo utilities
│   └── __init__.py
│
├── scripts/                   # Utility scripts
│   ├── benchmark_inference.py # Performance benchmarking
│   ├── prepare_dataset.py    # Dataset preparation
│   └── export_model.py       # Model export (ONNX, TorchScript)
│
├── config/                    # Configuration files
│   ├── training_config.yaml  # Training configuration
│   └── model_config.yaml     # Model configuration
│
├── requirements.txt
├── README.md
└── COPILOT_INSTRUCTIONS.md
```

## 🚀 Quick Start (Google Colab)

```python
# 1. Install dependencies
!wget -q https://raw.githubusercontent.com/biswaj45/vision-fgsm-imputer/main/scripts/setup_colab.py
!python setup_colab.py

# Or manually:
!pip install insightface onnxruntime-gpu gdown gradio torch torchvision albumentations

# 2. Clone repository
!git clone https://github.com/biswaj45/vision-fgsm-imputer.git
%cd vision-fgsm-imputer

# 3. Download trained model
!python scripts/check_model_in_repo.py --action download

# 4. Launch Gradio app
!python app/gradio_app.py --share
```

## 📦 Installation (Local)

### Installation

```bash
# Clone the repository
cd vision_fgsm_imputer

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

```bash
# Create sample dataset for testing
python scripts/prepare_dataset.py --dataset sample --output_dir ./data

# Or prepare your own dataset
python scripts/prepare_dataset.py --dataset custom --source_dir /path/to/images --output_dir ./data
```

### Training

```bash
# Train with default config (auto-detects GPU)
python training/train_unet.py

# Training automatically uses GPU if available
# For Colab T4: ~7x faster than CPU with mixed precision!
```

### Inference

```python
from inference.predict import impute_noise

# Protect an image
result = impute_noise(
    image_path="input.jpg",
    model_path="outputs/checkpoints/best.pth",
    output_path="protected.jpg",
    epsilon=0.02
)
```

### Web Interface

```bash
# Launch Gradio app (CPU inference for deployment)
python app/gradio_app.py --model_path outputs/checkpoints/best.pth

# With public sharing
python app/gradio_app.py --model_path outputs/checkpoints/best.pth --share

# Optional: Use GPU for inference (testing only)
python app/gradio_app.py --model_path outputs/checkpoints/best.pth --gpu
```

**Note**: Demo uses CPU by default for production-ready deployment (<400ms target).

## 📊 Benchmarking

```bash
# Benchmark inference performance
python scripts/benchmark_inference.py --model_path outputs/checkpoints/best.pth --num_images 100

# Expected results:
# - Model load time: <100ms
# - Inference time: <300ms per image
# - FPS: >3
# - Model size: <8MB
```

## 🔧 Model Export

```bash
# Export to ONNX
python scripts/export_model.py --model_path outputs/checkpoints/best.pth --format onnx

# Export to TorchScript
python scripts/export_model.py --model_path outputs/checkpoints/best.pth --format torchscript

# Export with optimization
python scripts/export_model.py --model_path outputs/checkpoints/best.pth --format onnx torchscript --optimize
```

## 🎓 Google Colab Support (T4 GPU for Training!)

This project follows a **hybrid workflow**: 
- ✅ **Train on GPU** (Colab T4) - Fast training with mixed precision
- ✅ **Infer on CPU** (Gradio demo) - Production-ready deployment

See [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md) for detailed instructions.

```python
# Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Install dependencies
!pip install -q torch torchvision albumentations gradio opencv-python pyyaml tensorboard tqdm

# Prepare sample dataset
!python scripts/prepare_dataset.py --dataset sample --output_dir ./data

# Train model on T4 GPU (automatically uses GPU with mixed precision!)
!python training/train_unet.py
# Expected: ~2-3 min per epoch on T4 (vs ~15-20 min on CPU)

# Launch Gradio app (uses CPU for fast deployment)
!python app/gradio_app.py --model_path outputs/checkpoints/best.pth --share
# Expected: <400ms per image on CPU
```

**Training Performance on Colab T4:**
- 🚀 Training: **~7x faster** than CPU (2-3 min vs 15-20 min per epoch)
- 💾 Memory: Uses ~2-4GB VRAM
- ⚡ Mixed Precision: 2x additional speedup

**Inference Performance (CPU Demo):**
- 🎯 Latency: **<300ms** per image
- 📊 Throughput: **3-5 FPS**
- 💻 Deployment: Production-ready for CPU servers

## 🧪 Model Architectures

### Tiny U-Net
- **Parameters**: ~4.8M
- **Architecture**: 3 down blocks, 3 up blocks with skip connections
- **Base channels**: 32
- **Output**: Perturbation map (tanh activation)

### Lightweight Autoencoder
- **Parameters**: ~2.5M
- **Latent dimension**: 128
- **Architecture**: Encoder-decoder with bottleneck
- **Output**: Perturbation map (tanh activation)

## 📈 Training Details

### Loss Function
- MSE between predicted perturbation and FGSM target perturbation

### Optimizer
- Adam (lr=1e-3, weight_decay=1e-5)
- ReduceLROnPlateau scheduler

### Data Augmentation (Albumentations)
- Horizontal flip
- Random brightness/contrast
- Hue/saturation adjustment
- Gaussian noise
- Motion blur / Gaussian blur

### FGSM Generation
- Epsilon: 0.02 (default)
- On-the-fly generation during training
- Target: perturbed version of input image

## 🎯 Usage Examples

### Simple API

```python
from inference.predict import impute_noise

# Protect an image
result = impute_noise("input.jpg", output_path="protected.jpg")
```

### Advanced Usage

```python
from inference.predict import NoiseImputer

# Initialize imputer
imputer = NoiseImputer(
    model_path="outputs/checkpoints/best.pth",
    model_type='unet',
    epsilon=0.02,
    device='cpu'
)

# Process from numpy array
import cv2
image = cv2.imread("input.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

perturbed, perturbation_map = imputer.impute_from_array(
    image,
    return_perturbation=True
)

# Benchmark performance
stats = imputer.benchmark(num_iterations=100)
print(f"Mean inference time: {stats['mean_ms']:.2f}ms")
print(f"FPS: {stats['fps']:.2f}")
```

## 🛠️ Development

### Running Tests

```python
# Test model architectures
python models/unet_tiny.py
python models/autoencoder.py

# Test inference
python inference/predict.py

# Test FGSM utilities
python training/utils.py
```

### Code Style
- Follow PEP8
- Use type hints
- Write docstrings for all functions
- Keep functions focused and modular

## 📝 Configuration

### Training Config (`config/training_config.yaml`)
```yaml
model_type: unet
batch_size: 16
num_epochs: 50
learning_rate: 0.001
epsilon: 0.02
image_size: 256
```

### Model Config (`config/model_config.yaml`)
```yaml
architecture:
  type: unet
  unet:
    in_channels: 3
    out_channels: 3
    base_channels: 32
    depth: 3
```

## 🎯 Performance Targets

- ✅ Model size: <8MB
- ✅ Parameters: <5M
- ✅ CPU inference: <300ms per 256×256 image
- ✅ FPS: >3
- ✅ Gradio app: <400ms total (including pre/post-processing)

## 📚 Datasets

### Supported Datasets
- **CelebA**: Aligned & Cropped Images
- **VGGFace2**: Large-scale face dataset
- **Custom**: Any folder-based image dataset

### Dataset Preparation
See `scripts/prepare_dataset.py` for detailed instructions on downloading and preparing datasets.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows project style guidelines
- Models remain lightweight (<5M params)
- CPU inference remains fast (<300ms)
- All tests pass

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- FGSM method by Goodfellow et al.
- U-Net architecture by Ronneberger et al.
- Albumentations library for data augmentation
- Gradio for web interface

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ for protecting images against deepfake manipulation**
