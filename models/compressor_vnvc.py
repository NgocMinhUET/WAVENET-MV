"""
CompressorVNVC - Video Neural Video Compressor
- Quantizer: round-with-noise
- Entropy model: CompressAI GaussianConditional  
- Loss Stage-2: λ·L_rec + BPP, λ ∈ {256,512,1024}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import GaussianConditional
from compressai.models.utils import conv, deconv
import math


class RoundWithNoise(torch.autograd.Function):
    """Round-with-noise quantizer for training"""
    
    @staticmethod
    def forward(ctx, input):
        # During training, add uniform noise and then round
        if ctx.needs_input_grad[0] and input.requires_grad:
            noise = torch.empty_like(input).uniform_(-0.5, 0.5)
            return torch.round(input + noise)  # FIX: Thêm torch.round()!
        else:
            # During inference, just round
            return torch.round(input)
    
    @staticmethod  
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output


class QuantizerVNVC(nn.Module):
    """Quantizer module với round-with-noise"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        """Apply quantization"""
        return RoundWithNoise.apply(x)
    
    def quantize(self, x):
        """Explicit quantization (for inference)"""
        return torch.round(x)


class EntropyBottleneck(nn.Module):
    """
    Entropy bottleneck với CompressAI GaussianConditional
    """
    
    def __init__(self, channels, tail_mass=1e-9, init_scale=0.1, filters=(3, 3, 3, 3)):
        super().__init__()
        
        self.channels = channels
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)
        
        # Context model để predict scale parameters
        self.context_prediction = nn.Sequential(
            conv(channels, channels, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            conv(channels, channels, kernel_size=5, stride=1), 
            nn.ReLU(inplace=True),
            conv(channels, channels, kernel_size=5, stride=1)
        )
        
        # Scale parameters cho Gaussian entropy model
        self.gaussian_conditional = GaussianConditional(None)
        
        # Initialize scale parameters
        self.register_parameter("_medians", nn.Parameter(torch.zeros(channels)))
        self.register_parameter("_scales", nn.Parameter(torch.ones(channels) * self.init_scale))
        
    def forward(self, y):
        """
        Forward pass qua entropy bottleneck
        Args:
            y: Quantized features [B, C, H, W]
        Returns:
            y_hat: Processed features
            likelihoods: Bit probabilities for rate calculation
        """
        # Context prediction
        scales = self.context_prediction(y)
        scales = torch.exp(scales)  # Ensure positive scales
        
        # Add learned global scales với SMALLER clamping để tăng quantization effect
        scales = scales + self._scales.view(1, -1, 1, 1).clamp(min=0.01, max=2.0)  # FIXED: Smaller range (0.01-2.0 instead of 0.1-10.0)
        
        # Gaussian conditional entropy model
        y_hat, likelihoods = self.gaussian_conditional(y, scales)
        
        # Ensure likelihoods are in reasonable range
        likelihoods = likelihoods.clamp(min=1e-10, max=1.0)
        
        return y_hat, likelihoods
    
    def compress(self, y):
        """Compress to bitstream"""
        scales = self.context_prediction(y)
        scales = torch.exp(scales) + self._scales.view(1, -1, 1, 1)
        
        strings = self.gaussian_conditional.compress(y, scales)
        return strings
    
    def decompress(self, strings, shape):
        """Decompress from bitstream"""
        # Simplified - in practice cần context từ partial reconstruction
        scales = torch.ones(shape).to(self._scales.device) * self._scales.view(1, -1, 1, 1)
        y_hat = self.gaussian_conditional.decompress(strings, scales)
        return y_hat


class CompressorVNVC(nn.Module):
    """
    Complete Video Neural Video Compressor
    Input: (B, C_mix, H, W) từ AdaMixNet 
    Output: bitstream + reconstruction loss
    """
    
    def __init__(self, input_channels=128, latent_channels=192, lambda_rd=256):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.lambda_rd = lambda_rd
        
        # Analysis transform (encoder) - FIXED: Reduce complexity
        self.analysis_transform = nn.Sequential(
            conv(input_channels, latent_channels, kernel_size=5, stride=2),  # /2
            nn.ReLU(inplace=True),
            conv(latent_channels, latent_channels, kernel_size=5, stride=2),  # /4
            nn.ReLU(inplace=True),
            conv(latent_channels, latent_channels, kernel_size=3, stride=1),  # FIXED: stride=1, smaller kernel
            nn.ReLU(inplace=True),
            conv(latent_channels, latent_channels, kernel_size=3, stride=1)   # FIXED: stride=1, smaller kernel
        )
        
        # Synthesis transform (decoder) - FIXED: Match analysis
        self.synthesis_transform = nn.Sequential(
            deconv(latent_channels, latent_channels, kernel_size=3, stride=1),  # FIXED: stride=1, smaller kernel
            nn.ReLU(inplace=True),
            deconv(latent_channels, latent_channels, kernel_size=3, stride=1),  # FIXED: stride=1, smaller kernel
            nn.ReLU(inplace=True), 
            deconv(latent_channels, latent_channels, kernel_size=5, stride=2),  # ×2
            nn.ReLU(inplace=True),
            deconv(latent_channels, input_channels, kernel_size=5, stride=2)    # ×4
        )
        
        # Quantizer
        self.quantizer = QuantizerVNVC()
        
        # Entropy bottleneck
        self.entropy_bottleneck = EntropyBottleneck(latent_channels)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input features [B, C_mix, H, W] từ AdaMixNet
        Returns:
            x_hat: Reconstructed features
            likelihoods: For rate calculation  
            y_quantized: Quantized latents
        """
        # Analysis transform
        y = self.analysis_transform(x)  # [B, latent_channels, H/16, W/16]
        
        # Quantization
        y_quantized = self.quantizer(y)
        
        # Entropy bottleneck
        y_hat, likelihoods = self.entropy_bottleneck(y_quantized)
        
        # Synthesis transform  
        x_hat = self.synthesis_transform(y_hat)
        
        return x_hat, likelihoods, y_quantized
    
    def compress(self, x):
        """
        Compress input to bitstream
        Args:
            x: Input features [B, C_mix, H, W]
        Returns:
            bitstream: Compressed representation
        """
        # Analysis transform
        y = self.analysis_transform(x)
        
        # Quantize for compression
        y_quantized = self.quantizer.quantize(y)
        
        # Entropy compress
        strings = self.entropy_bottleneck.compress(y_quantized)
        
        return {
            'strings': strings,
            'shape': y_quantized.shape,
        }
    
    def decompress(self, bitstream):
        """
        Decompress bitstream to features
        Args:
            bitstream: Compressed representation
        Returns:
            x_hat: Reconstructed features
        """
        # Entropy decompress
        y_hat = self.entropy_bottleneck.decompress(bitstream['strings'], bitstream['shape'])
        
        # Synthesis transform
        x_hat = self.synthesis_transform(y_hat)
        
        return x_hat
    
    def compute_rate_distortion_loss(self, x, x_hat, likelihoods, original_shape):
        """
        Compute RD loss: λ·L_rec + BPP
        Args:
            x: Original features
            x_hat: Reconstructed features  
            likelihoods: Bit probabilities
            original_shape: Original image dimensions for proper BPP calculation
        Returns:
            rd_loss: Rate-distortion loss
            distortion: Reconstruction loss
            rate: Rate in BPP
        """
        # Distortion (MSE reconstruction loss)
        distortion = F.mse_loss(x_hat, x)
        
        # Rate (bits per pixel) - FIXED: Use original image dimensions
        batch_size = original_shape[0]
        num_pixels = original_shape[2] * original_shape[3]  # H * W of ORIGINAL images
        
        # Compute rate từ likelihoods
        log_likelihoods = torch.log(likelihoods.clamp(min=1e-10))
        rate = -log_likelihoods.sum() / (batch_size * num_pixels * math.log(2))
        
        # Rate-distortion loss
        rd_loss = self.lambda_rd * distortion + rate
        
        return rd_loss, distortion, rate


class MultiLambdaCompressorVNVC(nn.Module):
    """
    Multi-lambda compressor hỗ trợ λ ∈ {64, 128, 256, 512, 1024, 2048, 4096}
    """
    
    def __init__(self, input_channels=128, latent_channels=192):
        super().__init__()
        
        # Multiple compressors cho different lambda values - UPDATED to include 128
        self.compressors = nn.ModuleDict({
            '64': CompressorVNVC(input_channels, latent_channels, lambda_rd=64),
            '128': CompressorVNVC(input_channels, latent_channels, lambda_rd=128),
            '256': CompressorVNVC(input_channels, latent_channels, lambda_rd=256),
            '512': CompressorVNVC(input_channels, latent_channels, lambda_rd=512), 
            '1024': CompressorVNVC(input_channels, latent_channels, lambda_rd=1024),
            '2048': CompressorVNVC(input_channels, latent_channels, lambda_rd=2048),
            '4096': CompressorVNVC(input_channels, latent_channels, lambda_rd=4096)
        })
        
        self.current_lambda = 128  # Default - match training
        
    def set_lambda(self, lambda_value):
        """Set current lambda value"""
        if str(lambda_value) not in self.compressors:
            raise ValueError(f"Lambda {lambda_value} not supported. Use one of {list(self.compressors.keys())}")
        self.current_lambda = lambda_value
        
    def forward(self, x):
        """Forward using current lambda"""
        compressor = self.compressors[str(self.current_lambda)]
        return compressor(x)
    
    def compress(self, x, lambda_value=None):
        """Compress với specific lambda"""
        if lambda_value is None:
            lambda_value = self.current_lambda
        compressor = self.compressors[str(lambda_value)]
        return compressor.compress(x)
    
    def decompress(self, bitstream, lambda_value=None):
        """Decompress với specific lambda"""
        if lambda_value is None:
            lambda_value = self.current_lambda
        compressor = self.compressors[str(lambda_value)]
        return compressor.decompress(bitstream)
    
    def compute_rate_distortion_loss(self, x, x_hat, likelihoods, original_shape):
        """Compute RD loss với current lambda"""
        compressor = self.compressors[str(self.current_lambda)]
        
        # Distortion (MSE reconstruction loss)
        distortion = F.mse_loss(x_hat, x)
        
        # Rate (bits per pixel) - FIXED: Use original image dimensions
        batch_size = original_shape[0]
        num_pixels = original_shape[2] * original_shape[3]  # H * W of ORIGINAL images
        
        # Compute rate từ likelihoods
        log_likelihoods = torch.log(likelihoods.clamp(min=1e-10))
        rate = -log_likelihoods.sum() / (batch_size * num_pixels * math.log(2))
        
        # Rate-distortion loss
        rd_loss = self.current_lambda * distortion + rate
        
        return rd_loss, distortion, rate


def test_compressor_vnvc():
    """Unit test for CompressorVNVC"""
    
    # Test single compressor
    compressor = CompressorVNVC(input_channels=128, latent_channels=192, lambda_rd=256)
    
    # Test forward pass
    x = torch.randn(2, 128, 64, 64)
    x_hat, likelihoods, y_quantized = compressor(x)
    
    # Check shapes
    assert x_hat.shape == x.shape, f"Reconstruction shape mismatch: {x_hat.shape} vs {x.shape}"
    assert y_quantized.shape[0] == x.shape[0], f"Latent batch size mismatch"
    assert likelihoods.shape == y_quantized.shape, f"Likelihoods shape mismatch"
    
    # Test RD loss
    rd_loss, distortion, rate = compressor.compute_rate_distortion_loss(x, x_hat, likelihoods, x.shape)
    assert rd_loss.item() > 0, f"RD loss should be positive"
    
    print(f"✓ Single compressor: RD={rd_loss.item():.4f}, MSE={distortion.item():.6f}, BPP={rate.item():.4f}")
    
    print("✓ CompressorVNVC tests passed!")
    
    # Test multi-lambda compressor
    multi_compressor = MultiLambdaCompressorVNVC(input_channels=128)
    
    # Test different lambda values
    for lambda_val in [64, 128, 256, 512, 1024]:
        multi_compressor.set_lambda(lambda_val)
        x_hat, likelihoods, y_quantized = multi_compressor(x)
        assert x_hat.shape == x.shape, f"Multi-compressor shape mismatch for λ={lambda_val}"
        
        # Test RD loss with correct signature
        rd_loss, distortion, rate = multi_compressor.compute_rate_distortion_loss(x, x_hat, likelihoods, x.shape)
        assert rd_loss.item() > 0, f"RD loss should be positive for λ={lambda_val}"
        
        print(f"✓ λ={lambda_val}: RD={rd_loss.item():.4f}, MSE={distortion.item():.6f}, BPP={rate.item():.4f}")


if __name__ == "__main__":
    test_compressor_vnvc() 