"""
Benchmark inference performance.
Reports model load time, latency per image, and FPS.
"""

import torch
import numpy as np
import time
from pathlib import Path
import sys
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.predict import NoiseImputer
from models.unet_tiny import create_tiny_unet
from models.autoencoder import create_autoencoder


def benchmark_model_loading(model_path: str, model_type: str = 'unet') -> dict:
    """
    Benchmark model loading time.
    
    Args:
        model_path: Path to model weights
        model_type: Model type
    
    Returns:
        Dictionary with loading statistics
    """
    print("\n" + "="*60)
    print("BENCHMARK: Model Loading")
    print("="*60)
    
    times = []
    num_trials = 5
    
    for i in range(num_trials):
        start = time.time()
        
        if model_type == 'unet':
            model = create_tiny_unet(pretrained_path=model_path)
        else:
            model = create_autoencoder(pretrained_path=model_path)
        
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"Trial {i+1}: {elapsed:.2f}ms")
    
    stats = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times)
    }
    
    print(f"\nResults:")
    print(f"  Mean: {stats['mean_ms']:.2f}ms")
    print(f"  Std:  {stats['std_ms']:.2f}ms")
    print(f"  Min:  {stats['min_ms']:.2f}ms")
    print(f"  Max:  {stats['max_ms']:.2f}ms")
    
    return stats


def benchmark_inference(
    imputer: NoiseImputer,
    num_images: int = 100,
    batch_sizes: list = [1]
) -> dict:
    """
    Benchmark inference performance.
    
    Args:
        imputer: NoiseImputer instance
        num_images: Number of test images
        batch_sizes: List of batch sizes to test
    
    Returns:
        Dictionary with performance statistics
    """
    print("\n" + "="*60)
    print("BENCHMARK: Inference Performance")
    print("="*60)
    print(f"Device: {imputer.device.upper()}")
    if imputer.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate dummy images
        dummy_images = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for _ in range(num_images)
        ]
        
        # Warmup
        for _ in range(5):
            _ = imputer.impute_from_array(dummy_images[0])
        
        # Benchmark
        times = []
        
        for img in tqdm(dummy_images, desc="Processing"):
            start = time.time()
            _ = imputer.impute_from_array(img)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        # Compute statistics
        stats = {
            'batch_size': batch_size,
            'num_images': num_images,
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'fps': 1000.0 / np.mean(times)
        }
        
        results[batch_size] = stats
        
        # Print results
        print(f"\n  Mean latency:   {stats['mean_ms']:.2f}ms")
        print(f"  Std:            {stats['std_ms']:.2f}ms")
        print(f"  Min:            {stats['min_ms']:.2f}ms")
        print(f"  Max:            {stats['max_ms']:.2f}ms")
        print(f"  Median:         {stats['median_ms']:.2f}ms")
        print(f"  95th percentile: {stats['p95_ms']:.2f}ms")
        print(f"  99th percentile: {stats['p99_ms']:.2f}ms")
        print(f"  FPS:            {stats['fps']:.2f}")
        
        # Check target
        target_ms = 300
        if stats['mean_ms'] < target_ms:
            print(f"  ✅ Target met: <{target_ms}ms")
        else:
            print(f"  ❌ Target missed: >{target_ms}ms")
    
    return results


def benchmark_memory(model_path: str, model_type: str = 'unet') -> dict:
    """
    Benchmark memory usage.
    
    Args:
        model_path: Path to model weights
        model_type: Model type
    
    Returns:
        Dictionary with memory statistics
    """
    print("\n" + "="*60)
    print("BENCHMARK: Memory Usage")
    print("="*60)
    
    # Load model
    if model_type == 'unet':
        model = create_tiny_unet(pretrained_path=model_path)
    else:
        model = create_autoencoder(pretrained_path=model_path)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    param_size_mb = (total_params * 4) / (1024 ** 2)  # 4 bytes per float32
    
    # Get model file size
    model_file_size_mb = Path(model_path).stat().st_size / (1024 ** 2)
    
    stats = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_size_mb': param_size_mb,
        'file_size_mb': model_file_size_mb
    }
    
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Param memory:         {param_size_mb:.2f} MB")
    print(f"  File size:            {model_file_size_mb:.2f} MB")
    
    # Check target
    target_mb = 8
    if model_file_size_mb < target_mb:
        print(f"  ✅ Target met: <{target_mb}MB")
    else:
        print(f"  ❌ Target missed: >{target_mb}MB")
    
    return stats


def save_benchmark_report(results: dict, output_path: str = "benchmark_report.txt"):
    """Save benchmark results to file."""
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ANTI-DEEPFAKE MODEL BENCHMARK REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Model loading
        if 'loading' in results:
            f.write("MODEL LOADING\n")
            f.write("-"*60 + "\n")
            for key, value in results['loading'].items():
                f.write(f"{key}: {value:.2f}ms\n")
            f.write("\n")
        
        # Memory
        if 'memory' in results:
            f.write("MEMORY USAGE\n")
            f.write("-"*60 + "\n")
            mem = results['memory']
            f.write(f"Total parameters: {mem['total_params']:,}\n")
            f.write(f"Model size: {mem['file_size_mb']:.2f} MB\n")
            f.write("\n")
        
        # Inference
        if 'inference' in results:
            f.write("INFERENCE PERFORMANCE\n")
            f.write("-"*60 + "\n")
            for batch_size, stats in results['inference'].items():
                f.write(f"\nBatch size: {batch_size}\n")
                f.write(f"  Mean latency: {stats['mean_ms']:.2f}ms\n")
                f.write(f"  FPS: {stats['fps']:.2f}\n")
            f.write("\n")
        
        f.write("="*60 + "\n")
    
    print(f"\n✓ Report saved to {output_path}")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark anti-deepfake model")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='unet',
        choices=['unet', 'autoencoder'],
        help='Model type'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=100,
        help='Number of images for inference benchmark'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.02,
        help='Perturbation magnitude'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_report.txt',
        help='Output report path'
    )
    
    args = parser.parse_args()
    
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
    
    # Run benchmarks
    all_results = {}
    
    # 1. Model loading
    loading_stats = benchmark_model_loading(args.model_path, args.model_type)
    all_results['loading'] = loading_stats
    
    # 2. Memory usage
    memory_stats = benchmark_memory(args.model_path, args.model_type)
    all_results['memory'] = memory_stats
    
    # 3. Inference performance
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nBenchmarking on: {device.upper()}")
    
    imputer = NoiseImputer(
        model_path=args.model_path,
        model_type=args.model_type,
        epsilon=args.epsilon,
        device=device
    )
    
    inference_stats = benchmark_inference(
        imputer,
        num_images=args.num_images,
        batch_sizes=[1]
    )
    all_results['inference'] = inference_stats
    
    # Save report
    save_benchmark_report(all_results, args.output)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
