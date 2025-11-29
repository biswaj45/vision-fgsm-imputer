"""
Tiny U-Net implementation for fast inference (<5M params).
Outputs a noise/perturbation map for anti-deepfake protection.
"""

import torch
import torch.nn as nn
from typing import Tuple


class DoubleConv(nn.Module):
    """Two consecutive Conv2d + BatchNorm + ReLU blocks."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TinyUNet(nn.Module):
    """
    Tiny U-Net for fast CPU inference.
    Total parameters: ~4.8M
    Input: RGB image (3, 256, 256)
    Output: Perturbation map (3, 256, 256)
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        
        # Decoder
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        
        # Output layer
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Output perturbation map (range: -1 to 1)
        perturbation = self.tanh(self.outc(x))
        return perturbation
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_tiny_unet(pretrained_path: str = None) -> TinyUNet:
    """
    Factory function to create TinyUNet model.
    
    Args:
        pretrained_path: Path to pretrained weights (.pth file)
    
    Returns:
        TinyUNet model instance
    """
    model = TinyUNet(in_channels=3, out_channels=3)
    
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format: checkpoint dict with model_state_dict
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pretrained weights from {pretrained_path} (epoch {checkpoint.get('epoch', '?')})")
        else:
            # Old format: direct state dict
            model.load_state_dict(checkpoint)
            print(f"Loaded pretrained weights from {pretrained_path}")
    
    print(f"Model has {model.count_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Test the model
    model = create_tiny_unet()
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {model.count_parameters():,}")
