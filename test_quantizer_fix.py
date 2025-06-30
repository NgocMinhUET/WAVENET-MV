#!/usr/bin/env python3
"""
Test script để verify RoundWithNoise fix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

def test_quantizer():
    """Test RoundWithNoise fix"""
    print("🔍 TESTING: RoundWithNoise Fix")
    print("-" * 40)
    
    # Test input with requires_grad=True
    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    print(f"Input: range=[{x.min():.4f}, {x.max():.4f}]")
    
    # Apply quantizer
    quantized = RoundWithNoise.apply(x)
    print(f"Quantized: range=[{quantized.min():.4f}, {quantized.max():.4f}]")
    
    # Check if properly quantized (should have no fractional part)
    fractional_part = torch.abs(quantized - torch.round(quantized))
    print(f"Fractional part: max={fractional_part.max():.8f}")
    
    if fractional_part.max() < 1e-6:
        print("✅ FIXED: RoundWithNoise now properly rounds!")
    else:
        print("❌ STILL BROKEN: RoundWithNoise not rounding properly!")
    
    # Test MSE calculation
    mse = F.mse_loss(quantized, x).item()
    print(f"MSE with input: {mse:.6f}")
    
    if mse > 0.01:  # Should have some reconstruction error due to quantization
        print("✅ MSE > 0: Quantization creates proper reconstruction loss")
    else:
        print("❌ MSE ≈ 0: Still acting like identity function")
    
    print()

def test_compressor_simulation():
    """Simulate CompressorVNVC behavior with fixed quantizer"""
    print("🔍 TESTING: Simulated Compressor Behavior")
    print("-" * 40)
    
    # Simulate mixed features from AdaMixNet
    mixed_features = torch.randn(1, 128, 32, 32, requires_grad=True)
    print(f"Mixed features: {mixed_features.shape}")
    
    # Simulate compression pipeline
    # 1. Analysis transform (simple downsampling)
    y = F.avg_pool2d(mixed_features, kernel_size=2, stride=2)  # [1, 128, 16, 16]
    print(f"After analysis: {y.shape}")
    
    # 2. Quantization (FIXED)
    y_quantized = RoundWithNoise.apply(y)
    
    # 3. Synthesis transform (simple upsampling)
    x_hat = F.interpolate(y_quantized, size=mixed_features.shape[2:], mode='bilinear', align_corners=False)
    print(f"After synthesis: {x_hat.shape}")
    
    # 4. Calculate MSE
    mse_loss = F.mse_loss(x_hat, mixed_features).item()
    print(f"MSE Loss: {mse_loss:.8f}")
    
    if mse_loss > 1e-4:
        print("✅ GOOD: MSE Loss > 0, compressor creates reconstruction error")
    else:
        print("❌ BAD: MSE Loss ≈ 0, still acting like identity")
    
    print()

if __name__ == "__main__":
    print("=" * 50)
    print("🚨 TESTING QUANTIZER FIX")
    print("=" * 50)
    
    test_quantizer()
    test_compressor_simulation()
    
    print("=" * 50)
    print("🎯 TEST COMPLETED")
    print("=" * 50) 