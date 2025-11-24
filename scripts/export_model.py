"""
Export trained model for deployment.
Supports ONNX export and optimization.
"""

import torch
import sys
from pathlib import Path
import argparse
import onnx
import onnxruntime as ort
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.unet_tiny import create_tiny_unet
from models.autoencoder import create_autoencoder


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple = (1, 3, 256, 256),
    opset_version: int = 11
):
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Output path for ONNX model
        input_shape: Input tensor shape
        opset_version: ONNX opset version
    """
    print(f"\nExporting to ONNX: {output_path}")
    
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print("✓ ONNX export complete")
    
    # Verify
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Get file size
    file_size_mb = Path(output_path).stat().st_size / (1024 ** 2)
    print(f"File size: {file_size_mb:.2f} MB")


def test_onnx_inference(
    onnx_path: str,
    num_tests: int = 10
):
    """
    Test ONNX model inference.
    
    Args:
        onnx_path: Path to ONNX model
        num_tests: Number of test runs
    """
    print(f"\nTesting ONNX inference...")
    
    # Create inference session
    session = ort.InferenceSession(onnx_path)
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Test inference
    import time
    times = []
    
    for i in range(num_tests):
        dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
        
        start = time.time()
        output = session.run([output_name], {input_name: dummy_input})
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
    
    mean_time = np.mean(times)
    print(f"✓ ONNX inference test passed")
    print(f"  Mean inference time: {mean_time:.2f}ms")
    print(f"  Output shape: {output[0].shape}")


def export_torchscript(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple = (1, 3, 256, 256)
):
    """
    Export model to TorchScript format.
    
    Args:
        model: PyTorch model
        output_path: Output path for TorchScript model
        input_shape: Input tensor shape
    """
    print(f"\nExporting to TorchScript: {output_path}")
    
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save
    traced_model.save(output_path)
    
    print("✓ TorchScript export complete")
    
    # Get file size
    file_size_mb = Path(output_path).stat().st_size / (1024 ** 2)
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Test loading
    print("\nTesting TorchScript model...")
    loaded_model = torch.jit.load(output_path)
    with torch.no_grad():
        output = loaded_model(dummy_input)
    print(f"✓ TorchScript model loads and runs successfully")
    print(f"  Output shape: {output.shape}")


def optimize_model(
    model: torch.nn.Module,
    output_path: str
):
    """
    Optimize model for inference.
    
    Args:
        model: PyTorch model
        output_path: Output path for optimized model
    """
    print(f"\nOptimizing model...")
    
    model.eval()
    
    # Apply optimizations
    # 1. Fuse operations
    try:
        model = torch.jit.optimize_for_inference(torch.jit.script(model))
        print("✓ Applied TorchScript optimizations")
    except:
        print("⚠ Could not apply TorchScript optimizations")
    
    # Save optimized model
    if hasattr(model, 'state_dict'):
        torch.save(model.state_dict(), output_path)
    else:
        torch.jit.save(model, output_path)
    
    print(f"✓ Optimized model saved to {output_path}")
    
    # Get file size
    file_size_mb = Path(output_path).stat().st_size / (1024 ** 2)
    print(f"File size: {file_size_mb:.2f} MB")


def create_model_info(
    model: torch.nn.Module,
    output_path: str,
    model_path: str,
    model_type: str
):
    """
    Create model info file with metadata.
    
    Args:
        model: PyTorch model
        output_path: Output path for info file
        model_path: Path to model weights
        model_type: Model type
    """
    print(f"\nCreating model info file: {output_path}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get model size
    model_size_mb = Path(model_path).stat().st_size / (1024 ** 2)
    
    # Create info dict
    info = {
        'model_type': model_type,
        'input_shape': [1, 3, 256, 256],
        'output_shape': [1, 3, 256, 256],
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
        'framework': 'PyTorch',
        'usage': {
            'python': 'from inference.predict import impute_noise; result = impute_noise("input.jpg", model_path="model.pth")',
            'cli': 'python inference/predict.py input.jpg output.jpg'
        }
    }
    
    # Save to file
    import json
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print("✓ Model info saved")
    
    # Print summary
    print("\nModel Summary:")
    print(f"  Type: {model_type}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Size: {model_size_mb:.2f} MB")


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description="Export trained model")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model weights (.pth)'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='unet',
        choices=['unet', 'autoencoder'],
        help='Model type'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./exports',
        help='Output directory for exported models'
    )
    parser.add_argument(
        '--format',
        type=str,
        nargs='+',
        default=['onnx'],
        choices=['onnx', 'torchscript', 'pytorch'],
        help='Export formats'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Apply optimization for inference'
    )
    
    args = parser.parse_args()
    
    # Check model path
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        print("\nCreating dummy model for testing...")
        
        # Create dummy model
        if args.model_type == 'unet':
            model = create_tiny_unet()
        else:
            model = create_autoencoder()
        
        torch.save(model.state_dict(), args.model_path)
        print(f"✓ Created dummy model at {args.model_path}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    if args.model_type == 'unet':
        model = create_tiny_unet(pretrained_path=args.model_path)
    else:
        model = create_autoencoder(pretrained_path=args.model_path)
    
    model.eval()
    print("✓ Model loaded")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to requested formats
    base_name = Path(args.model_path).stem
    
    if 'onnx' in args.format:
        onnx_path = output_dir / f"{base_name}.onnx"
        export_to_onnx(model, str(onnx_path))
        test_onnx_inference(str(onnx_path))
    
    if 'torchscript' in args.format:
        ts_path = output_dir / f"{base_name}.pt"
        export_torchscript(model, str(ts_path))
    
    if 'pytorch' in args.format or args.optimize:
        optimized_path = output_dir / f"{base_name}_optimized.pth"
        if args.optimize:
            optimize_model(model, str(optimized_path))
        else:
            torch.save(model.state_dict(), str(optimized_path))
            print(f"✓ PyTorch model saved to {optimized_path}")
    
    # Create model info
    info_path = output_dir / f"{base_name}_info.json"
    create_model_info(model, str(info_path), args.model_path, args.model_type)
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"Exported models saved to: {output_dir}")


if __name__ == "__main__":
    main()
