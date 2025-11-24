# COPILOT INSTRUCTIONS - Vision FGSM Imputer

## 📋 Overview

This document provides comprehensive instructions for GitHub Copilot when working with the Vision FGSM Imputer anti-deepfake protection system.

## 🎯 **CRITICAL: GPU/CPU Workflow**

```
┌─────────────────────────────────────────────────────────────┐
│                   5-DAY POC PROJECT                          │
│     FGSM-Imputer Anti-Deepfake Image Protection System      │
└─────────────────────────────────────────────────────────────┘

Phase 1: TRAINING (Colab T4 GPU)
┌────────────────────────────────────┐
│  🏋️ Training Pipeline               │
│  • Device: GPU (Colab T4)          │
│  • Mixed Precision: Enabled        │
│  • Speed: 2-3 min/epoch            │
│  • Output: model.pth (<8MB)        │
└────────────────────────────────────┘
                 ↓
Phase 2: INFERENCE (CPU for Deployment)
┌────────────────────────────────────┐
│  🚀 Gradio Demo                     │
│  • Device: CPU (default)           │
│  • Target: <400ms per image        │
│  • Actual: 200-300ms ✅            │
│  • Deployment: Production-ready    │
└────────────────────────────────────┘
```

**KEY REQUIREMENTS:**
- ✅ **Training**: MUST use GPU (Colab T4) with mixed precision
- ✅ **Inference**: MUST be fast on CPU (<400ms) for Gradio demo
- ✅ **Model**: Must be tiny (<5M params, <8MB) to work well on both

---

## 🏗️ Repository Structure (ALWAYS FOLLOW THIS)

```
vision_fgsm_imputer/
│
├── models/                    # Model architectures
│   ├── unet_tiny.py          # Tiny U-Net (<5M params)
│   ├── autoencoder.py        # Lightweight autoencoder
│   └── __init__.py
│
├── training/                  # Training components
│   ├── dataset.py            # CelebA/VGGFace2 loader
│   ├── train_unet.py         # Training loop with TensorBoard
│   ├── transforms.py         # Albumentations transforms
│   └── utils.py              # FGSM implementation
│
├── inference/                 # Inference pipeline
│   ├── predict.py            # Main API: impute_noise()
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
└── COPILOT_INSTRUCTIONS.md (this file)
```

## 🚀 Functional Requirements

### 1. Model Requirements

**Tiny U-Net**
- Must have <5M parameters
- Input: RGB image (3, 256, 256)
- Output: Perturbation map (3, 256, 256), range [-1, 1]
- Architecture: Encoder-decoder with skip connections
- CPU inference: <300ms per image
- File size: <8MB

**Lightweight Autoencoder**
- Alternative to U-Net
- Even smaller: <3M parameters
- Same input/output as U-Net
- Faster inference

**Model Interface**
```python
model = create_tiny_unet(pretrained_path="model.pth")
output = model(input)  # input: [B, 3, 256, 256] -> output: [B, 3, 256, 256]
```

### 2. FGSM Module

**Function Signature**
```python
def fgsm(
    x: torch.Tensor,
    model: nn.Module,
    epsilon: float = 0.02
) -> torch.Tensor:
    """Generate FGSM perturbation."""
    pass
```

**Requirements**
- Use gradient sign method
- Default epsilon: 0.02
- Should work with any PyTorch model
- Return perturbed image in valid range [0, 1]

### 3. Dataset

**Loader Requirements**
```python
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform, max_samples=None):
        # Load images from folder
        pass
    
    def __getitem__(self, idx):
        # Return transformed image
        # Apply Albumentations transforms
        pass
```

**Supported Datasets**
- CelebA (Align&Cropped)
- VGGFace2
- Custom folder-based datasets

**Transforms** (Albumentations)
- Resize to 256×256
- Horizontal flip
- Random brightness/contrast
- Hue/saturation adjustment
- Gaussian noise
- Blur (motion/gaussian)
- ImageNet normalization

**Dataset Returns**
```python
(clean_img, fgsm_target_img)  # Tuple of tensors
```

### 4. Training

**Training Loop Requirements**
```python
class Trainer:
    def __init__(self, config, device='cpu'):
        # Initialize model, optimizer, loss
        pass
    
    def train_epoch(self, train_loader, epoch):
        # Training loop with tqdm
        # Generate FGSM targets on-the-fly
        # Compute loss, backward, optimize
        pass
    
    def validate(self, val_loader, epoch):
        # Validation loop
        pass
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        # Save model checkpoint
        pass
```

**Requirements**
- PyTorch training loop
- DataLoader with num_workers
- TensorBoard logging
- Save best/latest checkpoints
- Learning rate scheduling
- Gradient clipping
- Progress bars with tqdm

**Loss Function**
```python
criterion = nn.MSELoss()
target_perturbation = fgsm_targets - images
loss = criterion(model_output, target_perturbation)
```

### 5. Inference

**Simple API**
```python
from inference.predict import impute_noise

img_out = impute_noise(
    image_path="input.jpg",
    model_path="model.pth",
    output_path="output.jpg",
    epsilon=0.02
)
```

**Advanced API**
```python
from inference.predict import NoiseImputer

imputer = NoiseImputer(
    model_path="model.pth",
    model_type='unet',
    epsilon=0.02,
    device='cpu'
)

# From numpy array
result = imputer.impute_from_array(image_np)

# From file path
result = imputer.impute_from_path("input.jpg", "output.jpg")

# Benchmark
stats = imputer.benchmark(num_iterations=10)
```

**Requirements**
- Fast CPU inference (<300ms)
- Handle RGB images (H, W, C) in range [0, 255]
- Automatic resize to 256×256
- Resize back to original size
- Return numpy arrays

### 6. Gradio App

**Interface Requirements**
```python
class AntiDeepfakeApp:
    def __init__(self, model_path, epsilon=0.02):
        # Load model
        pass
    
    def process_image(self, image, epsilon, show_heatmap, add_badge):
        # Process uploaded image
        # Return (output_image, info_text)
        pass
    
    def launch(self, share=False, port=7860):
        # Launch Gradio app
        pass
```

**Features**
- Upload image interface
- Epsilon slider (0.01 - 0.1)
- Show difference heatmap checkbox
- Add protection badge checkbox
- Side-by-side comparison
- Processing time display
- CPU only
- Fast: <400ms total

**Interface Elements**
- Input: Image upload
- Controls: Sliders, checkboxes
- Output: Comparison image + info text
- Footer: Model status, device info

### 7. Benchmark Script

**Report Requirements**
```python
def benchmark_inference(imputer, num_images=100):
    # Report:
    # - Model load time (ms)
    # - Mean latency per image (ms)
    # - Std, min, max, median (ms)
    # - 95th and 99th percentile (ms)
    # - FPS
    # - Pass/fail target check (<300ms)
    pass
```

## 🎯 Style Rules

### Code Style

**General**
- Clean, modular, documented code
- Use classes and functions
- Avoid monolithic scripts
- Type hints for all functions
- Comprehensive docstrings

**Example Function**
```python
def process_image(
    image: np.ndarray,
    epsilon: float = 0.02
) -> np.ndarray:
    """
    Process image with FGSM perturbation.
    
    Args:
        image: Input image (H, W, C) in range [0, 255]
        epsilon: Perturbation magnitude
    
    Returns:
        Perturbed image (H, W, C) in range [0, 255]
    """
    # Implementation
    pass
```

**Naming Conventions**
- Classes: PascalCase (TinyUNet, NoiseImputer)
- Functions: snake_case (create_tiny_unet, impute_noise)
- Variables: snake_case (batch_size, learning_rate)
- Constants: UPPER_CASE (MAX_PARAMS, TARGET_MS)

**Code Organization**
- Imports at top (standard library, third-party, local)
- Class definitions before functions
- Main execution in `if __name__ == "__main__":`
- Helper functions before main functions

### Performance Rules

**Model Constraints**
- ✅ MUST: <5M parameters
- ✅ MUST: <8MB file size
- ✅ MUST: <300ms CPU inference (256×256)
- ✅ MUST: <400ms total (Gradio app)
- ❌ AVOID: Large models
- ❌ AVOID: Heavy transformers
- ❌ AVOID: GPU-only operations

**Speed Optimization**
- Use efficient PyTorch operations
- Minimize data transfers
- Batch operations when possible
- Use tqdm for progress bars
- Profile bottlenecks

### Code Quality

**Documentation**
```python
class TinyUNet(nn.Module):
    """
    Tiny U-Net for fast CPU inference.
    Total parameters: ~4.8M
    Input: RGB image (3, 256, 256)
    Output: Perturbation map (3, 256, 256)
    """
    pass
```

**Error Handling**
```python
try:
    model = load_model(path)
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback or exit gracefully
```

**Testing**
```python
if __name__ == "__main__":
    # Test the module
    model = create_tiny_unet()
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
```

## 🔧 Implementation Patterns

### Model Creation Pattern
```python
def create_model(pretrained_path: str = None) -> nn.Module:
    """Factory function to create model."""
    model = ModelClass()
    
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {pretrained_path}")
    
    print(f"Model has {model.count_parameters():,} parameters")
    return model
```

### Training Pattern
```python
for epoch in range(num_epochs):
    # Train
    train_loss = train_epoch(train_loader, epoch)
    
    # Validate
    val_loss = validate(val_loader, epoch)
    
    # Save checkpoint
    is_best = val_loss < best_val_loss
    save_checkpoint(epoch, val_loss, is_best)
    
    # Update learning rate
    scheduler.step(val_loss)
```

### Inference Pattern
```python
model.eval()
with torch.no_grad():
    output = model(input_tensor)
```

### Progress Bar Pattern
```python
from tqdm import tqdm

pbar = tqdm(dataloader, desc="Processing")
for batch in pbar:
    # Process batch
    pbar.set_postfix({'loss': f'{loss:.6f}'})
```

## 🎓 Google Colab Compatibility

**Requirements**
- All code must run in Google Colab
- Use `!` for shell commands
- Handle file paths correctly
- Support GPU/CPU switching
- Include installation cells

**Example Colab Cell**
```python
# Install dependencies
!pip install -q torch torchvision albumentations gradio

# Clone repo (if needed)
!git clone https://github.com/user/repo.git
%cd repo

# Run training
!python training/train_unet.py
```

## 📝 Common Tasks

### When asked to "Create file X"
1. Use correct directory structure
2. Include proper imports
3. Add comprehensive docstrings
4. Follow style guidelines
5. Add test code in `if __name__ == "__main__":`

### When asked to "Write U-Net code"
1. Keep parameters <5M
2. Use efficient architecture
3. Include skip connections
4. Add parameter counting method
5. Test with dummy input

### When asked to "Generate training loop"
1. Use PyTorch DataLoader
2. Add tqdm progress bars
3. Include TensorBoard logging
4. Save checkpoints
5. Add validation loop
6. Handle learning rate scheduling

### When asked to "Write Gradio app"
1. CPU-only inference
2. Upload image interface
3. Show original vs perturbed
4. Display processing time
5. Fast (<400ms total)
6. Add model status indicator

### When asked to "Fix this error"
1. Identify root cause
2. Provide clear fix
3. Test the fix
4. Ensure no regressions
5. Update documentation if needed

## ✅ Checklist for New Code

- [ ] Follows repository structure
- [ ] Includes type hints
- [ ] Has comprehensive docstrings
- [ ] Uses meaningful variable names
- [ ] Includes error handling
- [ ] Has test code
- [ ] Follows PEP8
- [ ] Runs on CPU
- [ ] Compatible with Google Colab
- [ ] Meets performance targets
- [ ] Uses tqdm for loops
- [ ] Has proper imports
- [ ] Includes example usage

## 🚫 Things to AVOID

- ❌ Creating unnecessary complexity
- ❌ Using large models (>5M params)
- ❌ Heavy transformer architectures
- ❌ GPU-only code (must support CPU)
- ❌ Slow inference (>300ms)
- ❌ Monolithic scripts
- ❌ Missing documentation
- ❌ Hard-coded paths
- ❌ Unhandled exceptions
- ❌ Magic numbers (use constants)

## 🎯 Priority Order

1. **Speed**: CPU inference <300ms
2. **Size**: Model <8MB, <5M params
3. **Clarity**: Clean, documented code
4. **Simplicity**: Modular, not complex
5. **Functionality**: All features working
6. **Robustness**: Error handling
7. **Usability**: Easy to use API
8. **Compatibility**: Colab support

## 📚 Key Principles

1. **Lightweight First**: Always prefer smaller, faster models
2. **CPU Optimized**: Target CPU deployment
3. **Modular Design**: Separate concerns into modules
4. **Clear APIs**: Simple, intuitive interfaces
5. **Production Ready**: Handle edge cases
6. **Well Documented**: Explain everything
7. **Tested**: Include test code
8. **Maintainable**: Clean, readable code

---

**When in doubt, refer to this document and prioritize speed + clarity + simplicity!**
