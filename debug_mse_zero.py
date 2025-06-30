#!/usr/bin/env python3
"""
Debug script để tìm nguyên nhân MSE về 0 trong Stage 2 training
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import math
import numpy as np
from pathlib import Path

# Add parent directory
sys.path.append('.')

from models.wavelet_transform_cnn import WaveletTransformCNN
from models.adamixnet import AdaMixNet
from models.compressor_vnvc import CompressorVNVC

# Simulated components
class SimpleRoundWithNoise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if ctx.needs_input_grad[0] and input.requires_grad:
            noise = torch.empty_like(input).uniform_(-0.5, 0.5)
            return torch.round(input + noise)
        else:
            return torch.round(input)
    
    @staticmethod  
    def backward(ctx, grad_output):
        return grad_output

def debug_quantization_levels():
    """Debug quantization behavior"""
    print("🔍 DEBUGGING: Quantization Levels")
    print("-" * 50)
    
    # Test with different input ranges
    test_ranges = [
        ("Small values", torch.randn(4, 64, 32, 32) * 0.1),  # [-0.3, 0.3]
        ("Medium values", torch.randn(4, 64, 32, 32) * 1.0),  # [-3, 3]  
        ("Large values", torch.randn(4, 64, 32, 32) * 10.0),  # [-30, 30]
    ]
    
    for name, x in test_ranges:
        x.requires_grad_(True)
        
        # Apply quantization
        x_quantized = SimpleRoundWithNoise.apply(x)
        
        # Calculate MSE
        mse = F.mse_loss(x_quantized, x).item()
        
        # Calculate quantization error stats
        diff = torch.abs(x_quantized - x)
        
        print(f"{name}:")
        print(f"  Input range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"  Quantized range: [{x_quantized.min():.4f}, {x_quantized.max():.4f}]")
        print(f"  MSE: {mse:.8f}")
        print(f"  Diff mean: {diff.mean():.6f}, max: {diff.max():.6f}")
        print()

def debug_analysis_synthesis_pipeline():
    """Debug analysis-synthesis pipeline"""
    print("🔍 DEBUGGING: Analysis-Synthesis Pipeline")
    print("-" * 50)
    
    # Simulate pipeline components
    def simple_analysis(x):
        # Simulate downsampling by 16x (4 conv layers with stride 2)
        return F.avg_pool2d(x, kernel_size=16, stride=16)
    
    def simple_synthesis(y):
        # Simulate upsampling back to original size
        return F.interpolate(y, scale_factor=16, mode='bilinear', align_corners=False)
    
    # Test input
    x = torch.randn(1, 128, 256, 256, requires_grad=True) * 0.2  # Similar to mixed_features
    print(f"Input: {x.shape}, range: [{x.min():.4f}, {x.max():.4f}]")
    
    # Analysis
    y = simple_analysis(x)
    print(f"After analysis: {y.shape}, range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Quantization
    y_quantized = SimpleRoundWithNoise.apply(y)
    print(f"After quantization: {y_quantized.shape}, range: [{y_quantized.min():.4f}, {y_quantized.max():.4f}]")
    
    # Synthesis
    x_hat = simple_synthesis(y_quantized)
    print(f"After synthesis: {x_hat.shape}, range: [{x_hat.min():.4f}, {x_hat.max():.4f}]")
    
    # Calculate MSE
    mse = F.mse_loss(x_hat, x).item()
    print(f"Pipeline MSE: {mse:.8f}")
    
    if mse < 1e-6:
        print("❌ PROBLEM: Pipeline MSE too small!")
    else:
        print("✅ Pipeline MSE reasonable")
    
    print()

def debug_learning_rate_effect():
    """Debug effect of learning rate on MSE reduction"""
    print("🔍 DEBUGGING: Learning Rate Effect")
    print("-" * 50)
    
    # Simulate scenario: model learns to reduce MSE too quickly
    initial_mse = 0.001649  # From epoch 1
    learning_rates = [1e-4, 2e-4, 1.99e-4, 1.97e-4, 1.95e-4]
    
    print("Simulated MSE reduction pattern:")
    current_mse = initial_mse
    
    for epoch, lr in enumerate(learning_rates, 1):
        # Simulate aggressive MSE reduction
        reduction_factor = 0.1 + (lr / 2e-4) * 0.05  # Higher LR = more reduction
        current_mse *= reduction_factor
        
        print(f"Epoch {epoch}: LR={lr:.2e}, MSE={current_mse:.8f}")
        
        if current_mse < 1e-8:
            print("❌ MSE collapsed to near zero!")
            break
    
    print()

def debug_entropy_bottleneck_simulation():
    """Debug entropy bottleneck behavior"""
    print("🔍 DEBUGGING: Entropy Bottleneck Simulation")
    print("-" * 50)
    
    # Simulate GaussianConditional behavior
    def simulate_gaussian_conditional(y, scales):
        """Simplified simulation of GaussianConditional"""
        # In training, GaussianConditional might act nearly like identity
        # if scales are not properly learned
        
        # Add small noise proportional to scales
        noise = torch.randn_like(y) * scales.mean() * 0.01
        y_hat = y + noise
        
        # Simulate likelihoods (higher for smaller deviations)
        diff = torch.abs(y_hat - y)
        likelihoods = torch.exp(-diff / scales.mean())
        
        return y_hat, likelihoods
    
    # Test input
    y = torch.randn(1, 192, 16, 16, requires_grad=True)
    scales = torch.ones_like(y) * 1.0  # Fixed scale
    
    print(f"Input y: shape={y.shape}, range=[{y.min():.4f}, {y.max():.4f}]")
    
    # Apply entropy bottleneck
    y_hat, likelihoods = simulate_gaussian_conditional(y, scales)
    
    print(f"Output y_hat: shape={y_hat.shape}, range=[{y_hat.min():.4f}, {y_hat.max():.4f}]")
    print(f"Likelihoods: shape={likelihoods.shape}, range=[{likelihoods.min():.6f}, {likelihoods.max():.6f}]")
    
    # Check if acting as identity
    diff = torch.abs(y_hat - y)
    print(f"Difference: mean={diff.mean():.8f}, max={diff.max():.8f}")
    
    if diff.max() < 1e-4:
        print("❌ PROBLEM: EntropyBottleneck acting as identity!")
    else:
        print("✅ EntropyBottleneck adds proper distortion")
    
    print()

def debug_gradient_flow():
    """Debug gradient flow behavior"""
    print("🔍 DEBUGGING: Gradient Flow")
    print("-" * 50)
    
    # Simulate the training scenario
    mixed_features = torch.randn(1, 128, 256, 256, requires_grad=True) * 0.2
    
    # Simulate forward pass
    y = F.avg_pool2d(mixed_features, kernel_size=16, stride=16)  # Analysis
    y_quantized = SimpleRoundWithNoise.apply(y)  # Quantization
    x_hat = F.interpolate(y_quantized, size=mixed_features.shape[2:], mode='bilinear', align_corners=False)  # Synthesis
    
    # Calculate MSE
    mse_loss = F.mse_loss(x_hat, mixed_features)
    
    print(f"MSE Loss: {mse_loss.item():.8f}")
    
    # Backward pass
    mse_loss.backward()
    
    # Check gradients
    if mixed_features.grad is not None:
        grad_norm = mixed_features.grad.norm().item()
        print(f"Gradient norm: {grad_norm:.8f}")
        
        if grad_norm < 1e-8:
            print("❌ PROBLEM: Gradients too small!")
        else:
            print("✅ Gradients reasonable")
    else:
        print("❌ PROBLEM: No gradients!")
    
    print()

def debug_compression_fixes():
    """Debug the 5 compression fixes"""
    print("🔧 DEBUGGING: Compression System Fixes")
    print("-" * 60)
    
    # Import fixed models
    sys.path.append(str(Path(__file__).parent))
    
    from models.compressor_vnvc import CompressorVNVC
    from models.adamixnet import AdaMixNet
    
    # Test setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulate original images (as reference for BPP calculation)
    original_images = torch.randn(2, 3, 256, 256).to(device)
    
    # Simulate mixed features from AdaMixNet
    mixed_features = torch.randn(2, 128, 64, 64).to(device) * 0.5  # Moderate range
    
    print(f"Original images: {original_images.shape}")
    print(f"Mixed features: {mixed_features.shape}")
    
    # Test CompressorVNVC with fixes
    compressor = CompressorVNVC(input_channels=128, latent_channels=192, lambda_rd=128).to(device)
    
    # Forward pass
    x_hat, likelihoods, y_quantized = compressor(mixed_features)
    
    print(f"\nAfter compression:")
    print(f"Compressed features: {x_hat.shape}, range: [{x_hat.min():.4f}, {x_hat.max():.4f}]")
    print(f"Likelihoods: {likelihoods.shape}, range: [{likelihoods.min():.6f}, {likelihoods.max():.6f}]")
    
    # Check if acting as identity
    diff = torch.abs(x_hat - mixed_features)
    print(f"Reconstruction diff: mean={diff.mean():.8f}, max={diff.max():.8f}")
    
    if diff.max() < 1e-6:
        print("❌ STILL IDENTITY: Compressor still acting as identity function!")
        return False
    else:
        print("✅ COMPRESSION WORKING: Compressor applying meaningful compression!")
    
    # Test BPP calculation with fixed dimensions
    batch_size = original_images.size(0)
    num_pixels = original_images.size(2) * original_images.size(3)  # Original image H*W
    
    log_likelihoods = torch.log(likelihoods.clamp(min=1e-10))
    total_bits = -log_likelihoods.sum() / math.log(2)
    bpp = total_bits / (batch_size * num_pixels)
    
    print(f"\nBPP Calculation (FIXED):")
    print(f"Total bits: {total_bits:.2f}")
    print(f"Num pixels: {num_pixels} (from original {original_images.shape[2]}x{original_images.shape[3]})")
    print(f"BPP: {bpp:.4f}")
    
    # Test MSE without floor
    mse_loss = F.mse_loss(x_hat, mixed_features)
    print(f"MSE Loss (no floor): {mse_loss:.8f}")
    
    # Test rate-distortion loss
    rd_loss, distortion, rate = compressor.compute_rate_distortion_loss(
        mixed_features, x_hat, likelihoods, original_images.shape
    )
    
    print(f"\nRate-Distortion Components:")
    print(f"Distortion (MSE): {distortion:.8f}")
    print(f"Rate (BPP): {rate:.4f}")
    print(f"λ*MSE: {128 * distortion:.4f}")
    print(f"Total RD Loss: {rd_loss:.4f}")
    print(f"MSE component ratio: {(128 * distortion / rd_loss * 100):.1f}%")
    print(f"BPP component ratio: {(rate / rd_loss * 100):.1f}%")
    
    # Check if MSE component is meaningful
    if distortion < 1e-7:
        print("❌ MSE TOO SMALL: MSE still collapsing!")
        return False
    elif 128 * distortion < rate * 0.01:  # MSE component < 1% of BPP
        print("❌ MSE IGNORED: MSE component still being ignored by optimizer!")
        return False
    else:
        print("✅ BALANCED LOSS: MSE and BPP components are balanced!")
        return True

def main():
    """Main debugging function"""
    print("=" * 60)
    print("🚨 DEBUGGING MSE → 0 ISSUE IN STAGE 2")
    print("=" * 60)
    print()
    
    debug_quantization_levels()
    debug_analysis_synthesis_pipeline()
    debug_learning_rate_effect()
    debug_entropy_bottleneck_simulation()
    debug_gradient_flow()
    
    # NEW: Test compression fixes
    print()
    print("=" * 60)
    print("🔧 TESTING COMPRESSION FIXES")
    print("=" * 60)
    fixes_working = debug_compression_fixes()
    
    print("=" * 60)
    print("🎯 DEBUGGING COMPLETED")
    print("=" * 60)
    print()
    
    if fixes_working:
        print("🎉 SUCCESS: All compression fixes appear to be working!")
        print("💡 RECOMMENDED ACTION:")
        print("1. 🚀 Run Stage 2 training với λ=128")
        print("2. 📊 Monitor MSE values - should be 0.001-0.1 range")
        print("3. 📊 Monitor BPP values - should be 1-10 range")
        print("4. 🔍 Check debug output trong first epoch")
    else:
        print("❌ ISSUES REMAINING: Some fixes may not be working")
        print("💡 DEBUGGING STEPS:")
        print("1. 🔧 Check if CompressorVNVC still acting as identity")
        print("2. 🔧 Verify EntropyBottleneck scale parameters")
        print("3. 🔧 Check BPP calculation dimensions")
    
    print()
    print("💡 POTENTIAL SOLUTIONS:")
    print("1. 🔧 Increase quantization strength")
    print("2. 🔧 Adjust entropy bottleneck scales")
    print("3. 🔧 Reduce learning rate")
    print("4. 🔧 Add regularization to prevent perfect reconstruction")
    print("5. 🔧 Check if CompressorVNVC analysis/synthesis transforms are too powerful")

if __name__ == "__main__":
    main() 