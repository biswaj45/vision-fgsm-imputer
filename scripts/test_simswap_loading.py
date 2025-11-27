"""
Quick test to verify SimSwap Generator loads successfully with official architecture
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

def test_simswap_loading():
    print("="*70)
    print("SIMSWAP GENERATOR LOADING TEST")
    print("="*70)
    
    # Import official architecture
    print("\n1. Importing official architecture...")
    from app.simswap_models import Generator_Adain_Upsample
    print("✅ Architecture imported")
    
    # Create model
    print("\n2. Creating Generator (input_nc=3, output_nc=3, latent_size=512, n_blocks=9)...")
    model = Generator_Adain_Upsample(
        input_nc=3,
        output_nc=3,
        latent_size=512,
        n_blocks=9,
        deep=False,
        norm_layer=nn.BatchNorm2d,
        padding_type='reflect'
    )
    print("✅ Generator created")
    
    # Print model structure
    print("\n3. Model structure:")
    print(f"   - first_layer: {type(model.first_layer)}")
    print(f"   - down1: {type(model.down1)}")
    print(f"   - down2: {type(model.down2)}")
    print(f"   - down3: {type(model.down3)}")
    print(f"   - BottleNeck: {type(model.BottleNeck)} with {len(model.BottleNeck)} blocks")
    print(f"   - up3: {type(model.up3)}")
    print(f"   - up2: {type(model.up2)}")
    print(f"   - up1: {type(model.up1)}")
    print(f"   - last_layer: {type(model.last_layer)}")
    
    # Load checkpoint
    print("\n4. Loading checkpoint...")
    checkpoint_path = Path(__file__).parent.parent / 'checkpoints' / 'people' / 'latest_net_G.pth'
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("Please download from: https://drive.google.com/drive/folders/1jV6_0FIMPC53FZ2HzZNJZGMe55bbu17R")
        return False
    
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    print(f"✅ Checkpoint loaded: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"   - Checkpoint has {len(checkpoint)} keys")
        print(f"   - Sample keys: {list(checkpoint.keys())[:5]}")
    
    # Load state_dict
    print("\n5. Loading state_dict into model...")
    try:
        model.load_state_dict(checkpoint, strict=True)
        print("✅ State dict loaded successfully (strict=True)")
    except Exception as e:
        print(f"❌ Failed to load state dict: {e}")
        return False
    
    # Test forward pass
    print("\n6. Testing forward pass...")
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_latent = torch.randn(1, 512)
        
        try:
            output = model(dummy_input, dummy_latent)
            print(f"✅ Forward pass successful!")
            print(f"   - Input shape: {dummy_input.shape}")
            print(f"   - Latent shape: {dummy_latent.shape}")
            print(f"   - Output shape: {output.shape}")
            print(f"   - Output range: [{output.min():.3f}, {output.max():.3f}]")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - SimSwap Generator ready to use!")
    print("="*70)
    return True

if __name__ == "__main__":
    success = test_simswap_loading()
    sys.exit(0 if success else 1)
