"""
Lightweight autoencoder alternative for noise generation.
Even smaller than U-Net for ultra-fast inference.
"""

import torch
import torch.nn as nn
from typing import Tuple


class Encoder(nn.Module):
    """Encoder network that compresses input to latent space."""
    
    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Flatten and compress to latent vector
        self.fc = nn.Linear(256 * 16 * 16, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        latent = self.fc(x)
        return latent


class Decoder(nn.Module):
    """Decoder network that reconstructs from latent space."""
    
    def __init__(self, latent_dim: int = 128, out_channels: int = 3):
        super().__init__()
        
        # Expand from latent vector
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
        
        self.decoder = nn.Sequential(
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output perturbation in range [-1, 1]
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.fc(latent)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.decoder(x)
        return x


class LightweightAutoencoder(nn.Module):
    """
    Lightweight autoencoder for noise generation.
    Total parameters: ~2.5M (smaller than TinyUNet)
    Input: RGB image (3, 256, 256)
    Output: Perturbation map (3, 256, 256)
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        perturbation = self.decoder(latent)
        return perturbation
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        return self.encoder(x)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent."""
        return self.decoder(latent)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_autoencoder(pretrained_path: str = None, latent_dim: int = 128) -> LightweightAutoencoder:
    """
    Factory function to create LightweightAutoencoder model.
    
    Args:
        pretrained_path: Path to pretrained weights (.pth file)
        latent_dim: Size of latent representation
    
    Returns:
        LightweightAutoencoder model instance
    """
    model = LightweightAutoencoder(in_channels=3, out_channels=3, latent_dim=latent_dim)
    
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    print(f"Model has {model.count_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Test the model
    model = create_autoencoder()
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {model.count_parameters():,}")
