"""
SimSwap Generator Architecture.
Simplified implementation for face swapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with AdaIN."""
    
    def __init__(self, dim, latent_size):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        
        # AdaIN parameters from latent
        self.adain1 = AdaptiveInstanceNorm(dim, latent_size)
        self.adain2 = AdaptiveInstanceNorm(dim, latent_size)
        
    def forward(self, x, latent):
        residual = x
        out = self.conv1(x)
        out = self.adain1(out, latent)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.adain2(out, latent)
        return out + residual


class AdaptiveInstanceNorm(nn.Module):
    """Adaptive Instance Normalization."""
    
    def __init__(self, num_features, latent_size):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        
        # Learn scale and bias from latent
        self.fc = nn.Linear(latent_size, num_features * 2)
        
    def forward(self, x, latent):
        # Normalize
        h = self.norm(x)
        
        # Get adaptive parameters
        style = self.fc(latent)
        gamma, beta = style.chunk(2, dim=1)
        gamma = gamma.view(-1, x.size(1), 1, 1)
        beta = beta.view(-1, x.size(1), 1, 1)
        
        # Apply adaptive affine transform
        return gamma * h + beta


class Generator_Adain_Upsample(nn.Module):
    """
    SimSwap Generator with AdaIN for identity injection.
    Input: target face image (3x224x224)
    Latent: source identity embedding (512-dim)
    Output: swapped face (3x224x224)
    """
    
    def __init__(
        self,
        input_nc=3,
        output_nc=3,
        latent_size=512,
        ngf=64,
        n_blocks=9,
        deep=False
    ):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            
            # Downsample
            nn.Conv2d(ngf, ngf * 2, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
        )
        
        # Residual blocks with AdaIN
        self.res_blocks = nn.ModuleList([
            ResidualBlock(ngf * 4, latent_size) for _ in range(n_blocks)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            # Upsample
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            
            # Output
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7, 1, 0),
            nn.Tanh()
        )
        
    def forward(self, x, latent):
        """
        Args:
            x: Target face image (B, 3, 224, 224)
            latent: Source identity (B, 512)
        Returns:
            Swapped face (B, 3, 224, 224)
        """
        # Encode
        feat = self.encoder(x)
        
        # Apply residual blocks with identity injection
        for block in self.res_blocks:
            feat = block(feat, latent)
        
        # Decode
        out = self.decoder(feat)
        
        return out
