"""
Alternative Compressor Implementation - CompressAI-free
Để bypass Windows installation issues với CompressAI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Dict


class SimpleQuantizer(nn.Module):
    """Simple quantizer thay thế round-with-noise"""
    
    def __init__(self, num_levels=256):
        super().__init__()
        self.num_levels = num_levels
    
    def forward(self, x):
        """Quantize với straight-through estimator"""
        # Normalize to [-1, 1]
        x_norm = torch.tanh(x)
        
        # Quantize
        quantized = torch.round(x_norm * (self.num_levels // 2)) / (self.num_levels // 2)
        
        # Straight-through estimator
        return x + (quantized - x).detach()


class SimpleEntropyModel(nn.Module):
    """Simple entropy model thay thế GaussianConditional"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Learnable scale parameters
        self.register_parameter("scales", nn.Parameter(torch.ones(channels)))
        
    def forward(self, x):
        """Estimate likelihoods"""
        # Simple Laplacian distribution assumption
        scales = F.softplus(self.scales).view(1, -1, 1, 1)
        
        # Laplacian likelihood: p(x) = (1/2b) * exp(-|x|/b)
        likelihoods = torch.exp(-torch.abs(x) / scales) / (2 * scales)
        
        # Clamp để avoid numerical issues
        likelihoods = torch.clamp(likelihoods, min=1e-10, max=1.0)
        
        return x, likelihoods


class AlternativeCompressor(nn.Module):
    """
    Alternative Compressor thay thế CompressAI GaussianConditional
    Compatible với WAVENET-MV architecture
    """
    
    def __init__(self, input_channels=128, latent_channels=192, lambda_rd=256):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.lambda_rd = lambda_rd
        
        # Analysis transform (encoder) - giống CompressAI style
        self.analysis = nn.Sequential(
            nn.Conv2d(input_channels, latent_channels//2, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels//2, latent_channels, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, 5, stride=2, padding=2)
        )
        
        # Synthesis transform (decoder)
        self.synthesis = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, latent_channels, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(latent_channels, latent_channels, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(latent_channels, latent_channels//2, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(latent_channels//2, input_channels, 5, stride=2, padding=2, output_padding=1)
        )
        
        # Quantizer
        self.quantizer = SimpleQuantizer()
        
        # Entropy model
        self.entropy_model = SimpleEntropyModel(latent_channels)
        
    def forward(self, x):
        """
        Forward pass tương tự CompressAI
        Args:
            x: Input features [B, input_channels, H, W]
        Returns:
            x_hat: Reconstructed features
            likelihoods: For rate calculation
            y_quantized: Quantized latents
        """
        # Analysis transform
        y = self.analysis(x)
        
        # Quantization
        y_quantized = self.quantizer(y)
        
        # Entropy model
        y_hat, likelihoods = self.entropy_model(y_quantized)
        
        # Synthesis transform
        x_hat = self.synthesis(y_hat)
        
        return x_hat, likelihoods, y_quantized
    
    def compress(self, x):
        """Simplified compression"""
        y = self.analysis(x)
        y_quantized = self.quantizer(y)
        
        # Simple bitstream representation
        return {
            'latents': y_quantized,
            'shape': y_quantized.shape
        }
    
    def decompress(self, bitstream):
        """Simplified decompression"""
        y_hat = bitstream['latents']
        x_hat = self.synthesis(y_hat)
        return x_hat
    
    def compute_rate_distortion_loss(self, x, x_hat, likelihoods):
        """Compute RD loss"""
        # Distortion (MSE)
        distortion = F.mse_loss(x_hat, x)
        
        # Rate estimation
        batch_size = x.size(0)
        num_pixels = x.size(2) * x.size(3)
        
        # Simple rate calculation
        log_likelihoods = torch.log(likelihoods.clamp(min=1e-10))
        rate = -log_likelihoods.sum() / (batch_size * num_pixels * math.log(2))
        
        # RD loss
        rd_loss = self.lambda_rd * distortion + rate
        
        return rd_loss, distortion, rate


class MultiLambdaAlternativeCompressor(nn.Module):
    """Multi-lambda version của alternative compressor"""
    
    def __init__(self, input_channels=128, latent_channels=192):
        super().__init__()
        
        self.compressors = nn.ModuleDict({
            '256': AlternativeCompressor(input_channels, latent_channels, 256),
            '512': AlternativeCompressor(input_channels, latent_channels, 512),
            '1024': AlternativeCompressor(input_channels, latent_channels, 1024)
        })
        
        self.current_lambda = 256
        
    def set_lambda(self, lambda_value):
        """Set current lambda"""
        self.current_lambda = lambda_value
        
    def forward(self, x):
        """Forward với current lambda"""
        return self.compressors[str(self.current_lambda)](x)
    
    def compress(self, x, lambda_value=None):
        """Compress với specific lambda"""
        if lambda_value is None:
            lambda_value = self.current_lambda
        return self.compressors[str(lambda_value)].compress(x)
    
    def decompress(self, bitstream, lambda_value=None):
        """Decompress với specific lambda"""
        if lambda_value is None:
            lambda_value = self.current_lambda
        return self.compressors[str(lambda_value)].decompress(bitstream)
    
    def compute_rate_distortion_loss(self, x, x_hat, likelihoods):
        """Compute RD loss với current lambda"""
        return self.compressors[str(self.current_lambda)].compute_rate_distortion_loss(x, x_hat, likelihoods)


def test_alternative_compressor():
    """Test alternative compressor"""
    print("Testing Alternative Compressor (CompressAI-free)...")
    
    # Test single compressor
    compressor = AlternativeCompressor(input_channels=128)
    
    x = torch.randn(2, 128, 64, 64)
    x_hat, likelihoods, y_quantized = compressor(x)
    
    assert x_hat.shape == x.shape, f"Shape mismatch: {x_hat.shape} vs {x.shape}"
    
    # Test RD loss
    rd_loss, distortion, rate = compressor.compute_rate_distortion_loss(x, x_hat, likelihoods)
    print(f"✓ RD Loss: {rd_loss.item():.4f}, Distortion: {distortion.item():.4f}, Rate: {rate.item():.4f}")
    
    # Test multi-lambda
    multi_compressor = MultiLambdaAlternativeCompressor()
    
    for lambda_val in [256, 512, 1024]:
        multi_compressor.set_lambda(lambda_val)
        x_hat, likelihoods, y_quantized = multi_compressor(x)
        rd_loss, distortion, rate = multi_compressor.compute_rate_distortion_loss(x, x_hat, likelihoods)
        print(f"✓ λ={lambda_val}: RD={rd_loss.item():.4f}")
    
    print("✅ Alternative Compressor tests passed!")


if __name__ == "__main__":
    test_alternative_compressor() 