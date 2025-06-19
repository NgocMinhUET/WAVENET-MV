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
        # During forward, add uniform noise for training
        if ctx.needs_input_grad[0] and input.requires_grad:
            noise = torch.empty_like(input).uniform_(-0.5, 0.5)
            return input + noise
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
    
    def __init__(self, channels, tail_mass=1e-9, init_scale=10, filters=(3, 3, 3, 3)):
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
        
        # Add learned global scales
        scales = scales + self._scales.view(1, -1, 1, 1)
        
        # Gaussian conditional entropy model
        _, likelihoods = self.gaussian_conditional(y, scales)
        
        return y, likelihoods
    
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
        
        # Analysis transform (encoder)
        self.analysis_transform = nn.Sequential(
            conv(input_channels, latent_channels, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(latent_channels, latent_channels, kernel_size=5, stride=2),  
            nn.ReLU(inplace=True),
            conv(latent_channels, latent_channels, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(latent_channels, latent_channels, kernel_size=5, stride=2)
        )
        
        # Synthesis transform (decoder)
        self.synthesis_transform = nn.Sequential(
            deconv(latent_channels, latent_channels, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(latent_channels, latent_channels, kernel_size=5, stride=2),
            nn.ReLU(inplace=True), 
            deconv(latent_channels, latent_channels, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(latent_channels, input_channels, kernel_size=5, stride=2)
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
    
    def compute_rate_distortion_loss(self, x, x_hat, likelihoods):
        """
        Compute RD loss: λ·L_rec + BPP
        Args:
            x: Original features
            x_hat: Reconstructed features  
            likelihoods: Bit probabilities
        Returns:
            rd_loss: Rate-distortion loss
            distortion: Reconstruction loss
            rate: Rate in BPP
        """
        # Distortion (MSE reconstruction loss)
        distortion = F.mse_loss(x_hat, x)
        
        # Rate (bits per pixel)
        batch_size = x.size(0)
        num_pixels = x.size(2) * x.size(3)  # H * W
        
        # Compute rate từ likelihoods
        log_likelihoods = torch.log(likelihoods.clamp(min=1e-10))
        rate = -log_likelihoods.sum() / (batch_size * num_pixels * math.log(2))
        
        # Rate-distortion loss
        rd_loss = self.lambda_rd * distortion + rate
        
        return rd_loss, distortion, rate


class MultiLambdaCompressorVNVC(nn.Module):
    """
    Multi-lambda compressor hỗ trợ λ ∈ {256, 512, 1024}
    """
    
    def __init__(self, input_channels=128, latent_channels=192):
        super().__init__()
        
        # Multiple compressors cho different lambda values
        self.compressors = nn.ModuleDict({
            '256': CompressorVNVC(input_channels, latent_channels, lambda_rd=256),
            '512': CompressorVNVC(input_channels, latent_channels, lambda_rd=512), 
            '1024': CompressorVNVC(input_channels, latent_channels, lambda_rd=1024)
        })
        
        self.current_lambda = 256  # Default
        
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
    
    def compute_rate_distortion_loss(self, x, x_hat, likelihoods):
        """Compute RD loss với current lambda"""
        compressor = self.compressors[str(self.current_lambda)]
        
        # Distortion (MSE reconstruction loss)
        distortion = F.mse_loss(x_hat, x)
        
        # Rate (bits per pixel)
        batch_size = x.size(0)
        num_pixels = x.size(2) * x.size(3)  # H * W
        
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
    
    print("✓ CompressorVNVC tests passed!")
    
    # Test multi-lambda compressor
    multi_compressor = MultiLambdaCompressorVNVC(input_channels=128)
    
    # Test different lambda values
    for lambda_val in [256, 512, 1024]:
        multi_compressor.set_lambda(lambda_val)
        x_hat, likelihoods, y_quantized = multi_compressor(x)
        assert x_hat.shape == x.shape, f"Multi-compressor shape mismatch for λ={lambda_val}"
        
        # Test RD loss
        rd_loss, distortion, rate = multi_compressor.compute_rate_distortion_loss(x, x_hat, likelihoods)
        assert rd_loss.item() > 0, f"RD loss should be positive for λ={lambda_val}"
    
    print("✓ MultiLambdaCompressorVNVC tests passed!")


if __name__ == "__main__":
    test_compressor_vnvc() 