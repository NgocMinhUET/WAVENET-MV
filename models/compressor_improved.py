"""
Improved CompressorVNVC - Fixed Analysis Transform + Adaptive Quantizer
Based on bottleneck analysis: Analysis transform collapses range 0.22 -> 0.04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import GaussianConditional
from compressai.models.utils import conv, deconv
import math


class AdaptiveRoundWithNoise(torch.autograd.Function):
    """
    Improved quantizer with adaptive scaling based on input range
    Addresses Analysis Transform range collapse issue
    """
    
    @staticmethod
    def forward(ctx, input):
        # Calculate adaptive scale factor based on ACTUAL input range
        input_range = input.abs().max()
        
        # Adjusted thresholds based on real pipeline analysis
        if input_range < 0.02:      # Ultra tiny (like our 0.04 case)
            scale_factor = 50.0     # Aggressive scaling
        elif input_range < 0.05:    # Small 
            scale_factor = 20.0
        elif input_range < 0.1:     # Medium small
            scale_factor = 10.0
        elif input_range < 0.5:     # Medium
            scale_factor = 4.0
        else:                       # Large
            scale_factor = 1.0
            
        # Scale up to quantizable range
        scaled_input = input * scale_factor
        
        # Quantize
        if ctx.needs_input_grad[0] and input.requires_grad:
            noise = torch.empty_like(scaled_input).uniform_(-0.5, 0.5)
            quantized = torch.round(scaled_input + noise)
        else:
            quantized = torch.round(scaled_input)
            
        # Scale back down
        return quantized / scale_factor
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ImprovedQuantizerVNVC(nn.Module):
    """Improved quantizer with adaptive scaling"""
    
    def __init__(self, min_scale=1.0, max_scale=50.0):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    def forward(self, x):
        return AdaptiveRoundWithNoise.apply(x)
        
    def quantize(self, x):
        """Inference quantization with same adaptive logic"""
        input_range = x.abs().max()
        
        if input_range < 0.02:
            scale_factor = 50.0
        elif input_range < 0.05:
            scale_factor = 20.0
        elif input_range < 0.1:
            scale_factor = 10.0
        elif input_range < 0.5:
            scale_factor = 4.0
        else:
            scale_factor = 1.0
            
        scaled = x * scale_factor
        quantized = torch.round(scaled)
        return quantized / scale_factor


class ImprovedAnalysisTransform(nn.Module):
    """
    Improved Analysis Transform with normalization to prevent range collapse
    Based on analysis: Current transform collapses 0.22 -> 0.04
    """
    
    def __init__(self, input_channels=128, latent_channels=192):
        super().__init__()
        
        # Layer 1: Conv + GroupNorm + ReLU
        self.conv1 = conv(input_channels, latent_channels, kernel_size=5, stride=2)
        self.norm1 = nn.GroupNorm(8, latent_channels)  # 8 groups
        
        # Layer 2: Conv + GroupNorm + ReLU  
        self.conv2 = conv(latent_channels, latent_channels, kernel_size=5, stride=2)
        self.norm2 = nn.GroupNorm(8, latent_channels)
        
        # Layer 3: Conv + GroupNorm (reduced from 4->3 layers)
        self.conv3 = conv(latent_channels, latent_channels, kernel_size=3, stride=1)
        self.norm3 = nn.GroupNorm(8, latent_channels)
        
        # Skip connection để preserve information
        self.skip_conv = conv(input_channels, latent_channels, kernel_size=1, stride=4)  # 1x1 conv to match dimensions
        
    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv3(out)
        out = self.norm3(out)
        
        # Skip connection - downsample input to match output size
        skip = self.skip_conv(x)
        
        # Residual connection
        out = out + skip
        out = F.relu(out, inplace=True)
        
        return out


class ImprovedSynthesisTransform(nn.Module):
    """
    Matching synthesis transform for improved analysis
    """
    
    def __init__(self, latent_channels=192, output_channels=128):
        super().__init__()
        
        # Layer 1: Deconv + GroupNorm + ReLU
        self.deconv1 = deconv(latent_channels, latent_channels, kernel_size=3, stride=1)
        self.norm1 = nn.GroupNorm(8, latent_channels)
        
        # Layer 2: Deconv + GroupNorm + ReLU
        self.deconv2 = deconv(latent_channels, latent_channels, kernel_size=5, stride=2)
        self.norm2 = nn.GroupNorm(8, latent_channels)
        
        # Layer 3: Deconv (output layer)
        self.deconv3 = deconv(latent_channels, output_channels, kernel_size=5, stride=2)
        
        # Skip connection
        self.skip_deconv = deconv(latent_channels, output_channels, kernel_size=1, stride=4)
        
    def forward(self, x):
        # Store input for skip connection
        skip_input = x
        
        # Main path
        out = self.deconv1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)
        
        out = self.deconv2(out)
        out = self.norm2(out)
        out = F.relu(out, inplace=True)
        
        out = self.deconv3(out)
        
        # Skip connection
        skip = self.skip_deconv(skip_input)
        
        # Residual connection
        out = out + skip
        
        return out


class ImprovedCompressorVNVC(nn.Module):
    """
    Improved Compressor với fixed analysis transform và adaptive quantizer
    Addresses range collapse issue: 0.22 -> 0.04 -> all zeros
    """
    
    def __init__(self, input_channels=128, latent_channels=192, lambda_rd=256):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.lambda_rd = lambda_rd
        
        # Improved transforms
        self.analysis_transform = ImprovedAnalysisTransform(input_channels, latent_channels)
        self.synthesis_transform = ImprovedSynthesisTransform(latent_channels, input_channels)
        
        # Improved quantizer
        self.quantizer = ImprovedQuantizerVNVC()
        
        # Enhanced entropy bottleneck
        try:
            from models.compressor_vnvc import EntropyBottleneck
        except ModuleNotFoundError:
            from compressor_vnvc import EntropyBottleneck
        self.entropy_bottleneck = EntropyBottleneck(latent_channels, init_scale=0.2)
        
    def to(self, device):
        """Move all parameters to device"""
        super().to(device)
        if hasattr(self, 'analysis_transform'):
            self.analysis_transform.to(device)
        if hasattr(self, 'synthesis_transform'):
            self.synthesis_transform.to(device)
        if hasattr(self, 'quantizer'):
            self.quantizer.to(device)
        if hasattr(self, 'entropy_bottleneck'):
            self.entropy_bottleneck.to(device)
        return self
        
    def forward(self, x):
        """
        Forward pass với improved components
        """
        # Analysis transform với range preservation
        y = self.analysis_transform(x)
        
        # Adaptive quantization
        y_quantized = self.quantizer(y)
        
        # Entropy bottleneck
        y_hat, likelihoods = self.entropy_bottleneck(y_quantized)
        
        # Synthesis transform
        x_hat = self.synthesis_transform(y_hat)
        
        return x_hat, likelihoods, y_quantized
    
    def compress(self, x):
        """Compression với improved pipeline"""
        y = self.analysis_transform(x)
        y_quantized = self.quantizer.quantize(y)
        strings = self.entropy_bottleneck.compress(y_quantized)
        return strings
    
    def decompress(self, strings, shape):
        """Decompression"""
        y_hat = self.entropy_bottleneck.decompress(strings, shape)
        x_hat = self.synthesis_transform(y_hat)
        return x_hat
    
    def compute_rate_distortion_loss(self, x, x_hat, likelihoods, original_shape):
        """Same RD loss as original"""
        # Reconstruction loss
        mse = F.mse_loss(x_hat, x)
        
        # Rate loss
        H, W = original_shape
        N = x.size(0)
        num_pixels = N * H * W
        
        # BPP calculation với fixed dimensions
        bpp = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        bpp = torch.clamp(bpp, 0.1, 10.0)  # Reasonable range
        
        # Rate-distortion tradeoff
        rd_loss = self.lambda_rd * mse + bpp
        
        return rd_loss, mse, bpp


class ImprovedMultiLambdaCompressorVNVC(nn.Module):
    """
    Multi-lambda version of improved compressor
    Maintains compatibility with existing training scripts
    """
    
    def __init__(self, input_channels=128, latent_channels=192):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        
        # Create compressor instances for different lambdas
        self.compressors = nn.ModuleDict({
            '64': ImprovedCompressorVNVC(input_channels, latent_channels, 64),
            '128': ImprovedCompressorVNVC(input_channels, latent_channels, 128),
            '256': ImprovedCompressorVNVC(input_channels, latent_channels, 256),
            '512': ImprovedCompressorVNVC(input_channels, latent_channels, 512),
            '1024': ImprovedCompressorVNVC(input_channels, latent_channels, 1024),
            '2048': ImprovedCompressorVNVC(input_channels, latent_channels, 2048),
            '4096': ImprovedCompressorVNVC(input_channels, latent_channels, 4096)
        })
        
        self.current_lambda = 128
        
    def to(self, device):
        """Move all parameters to device"""
        super().to(device)
        if hasattr(self, 'compressors'):
            for lambda_key, compressor in self.compressors.items():
                if compressor is not None:
                    compressor.to(device)
        return self
        
    def set_lambda(self, lambda_value):
        """Set current lambda value"""
        self.current_lambda = lambda_value
        
    def forward(self, x):
        """Forward pass using current lambda"""
        compressor = self.compressors[str(self.current_lambda)]
        return compressor(x)
        
    def compress(self, x, lambda_value=None):
        """Compress using specified lambda"""
        if lambda_value is None:
            lambda_value = self.current_lambda
        compressor = self.compressors[str(lambda_value)]
        return compressor.compress(x)
        
    def decompress(self, bitstream, lambda_value=None):
        """Decompress using specified lambda"""
        if lambda_value is None:
            lambda_value = self.current_lambda
        compressor = self.compressors[str(lambda_value)]
        return compressor.decompress(bitstream)
        
    def compute_rate_distortion_loss(self, x, x_hat, likelihoods, original_shape):
        """Compute RD loss using current lambda"""
        compressor = self.compressors[str(self.current_lambda)]
        return compressor.compute_rate_distortion_loss(x, x_hat, likelihoods, original_shape)
        
    def update(self):
        """Update all entropy models"""
        for lambda_key, compressor in self.compressors.items():
            try:
                if hasattr(compressor, 'entropy_bottleneck') and hasattr(compressor.entropy_bottleneck, 'gaussian_conditional'):
                    compressor.entropy_bottleneck.gaussian_conditional.update()
            except Exception as e:
                print(f"Warning: Failed to update entropy model for lambda={lambda_key}: {e}")


def test_improved_compressor():
    """Test improved compressor với realistic inputs"""
    
    print("🧪 TESTING IMPROVED COMPRESSOR")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create improved compressor
    improved_comp = ImprovedCompressorVNVC(input_channels=128, latent_channels=192, lambda_rd=128).to(device)
    improved_comp.eval()
    
    # Test với range collapse scenario
    test_input = torch.randn(1, 128, 64, 64).to(device) * 0.2  # Similar to AdaMixNet output
    print(f"Input range: [{test_input.min():.6f}, {test_input.max():.6f}]")
    
    with torch.no_grad():
        # Analysis transform
        analysis_out = improved_comp.analysis_transform(test_input)
        print(f"After Improved Analysis: [{analysis_out.min():.6f}, {analysis_out.max():.6f}]")
        
        # Full forward
        x_hat, likelihoods, y_quantized = improved_comp(test_input)
        
        print(f"Quantized range: [{y_quantized.min():.6f}, {y_quantized.max():.6f}]")
        print(f"Quantized unique: {torch.unique(y_quantized).numel()}")
        print(f"Non-zero ratio: {(y_quantized != 0).float().mean():.4f}")
        
        # Reconstruction quality
        mse = F.mse_loss(x_hat, test_input)
        print(f"Reconstruction MSE: {mse.item():.6f}")
        
        if (y_quantized == 0).all():
            print("❌ Still all zeros - need further fixes")
        elif torch.unique(y_quantized).numel() > 10:
            print("✅ Good quantization diversity!")
        else:
            print("⚠️ Limited diversity but improved")

if __name__ == "__main__":
    test_improved_compressor() 