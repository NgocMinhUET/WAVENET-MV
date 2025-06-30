#!/usr/bin/env python3
"""
Debug script cho CompressorVNVC MSE = 0 issue
Tìm ra chính xác vấn đề trong Stage 2 training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.compressor_vnvc import CompressorVNVC, RoundWithNoise
from models.adamixnet import AdaMixNet
from models.wavelet_transform_cnn import WaveletTransformCNN

def debug_round_with_noise():
    """Debug RoundWithNoise behavior"""
    print("🔍 DEBUGGING: RoundWithNoise")
    print("-" * 50)
    
    # Create test input
    x = torch.randn(2, 64, 32, 32, requires_grad=True)
    print(f"Input: shape={x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")
    
    # Test RoundWithNoise
    quantized = RoundWithNoise.apply(x)
    print(f"Quantized: shape={quantized.shape}, range=[{quantized.min():.4f}, {quantized.max():.4f}]")
    
    # Check difference
    diff = torch.abs(quantized - x)
    print(f"Difference: mean={diff.mean():.6f}, max={diff.max():.6f}")
    
    # Check if it's actually quantized
    fractional_part = torch.abs(quantized - torch.round(quantized))
    print(f"Fractional part: mean={fractional_part.mean():.6f}, max={fractional_part.max():.6f}")
    
    if fractional_part.max() > 0.1:
        print("❌ BUG: RoundWithNoise không round trong training!")
    else:
        print("✅ RoundWithNoise hoạt động đúng")
    
    print()

def debug_compressor_forward():
    """Debug CompressorVNVC forward pass"""
    print("🔍 DEBUGGING: CompressorVNVC Forward Pass")
    print("-" * 50)
    
    # Create compressor
    compressor = CompressorVNVC(input_channels=128, latent_channels=192, lambda_rd=512)
    compressor.train()
    
    # Create test input
    x = torch.randn(2, 128, 64, 64)
    print(f"Input: shape={x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")
    
    # Forward pass
    with torch.no_grad():
        x_hat, likelihoods, y_quantized = compressor(x)
    
    print(f"Output: shape={x_hat.shape}, range=[{x_hat.min():.4f}, {x_hat.max():.4f}]")
    print(f"Likelihoods: shape={likelihoods.shape}, range=[{likelihoods.min():.6f}, {likelihoods.max():.6f}]")
    print(f"Y_quantized: shape={y_quantized.shape}, range=[{y_quantized.min():.4f}, {y_quantized.max():.4f}]")
    
    # Check shape match
    if x_hat.shape != x.shape:
        print(f"❌ SHAPE MISMATCH: {x_hat.shape} vs {x.shape}")
    else:
        print("✅ Shape match OK")
    
    # Check MSE
    mse = F.mse_loss(x_hat, x).item()
    print(f"MSE Loss: {mse:.8f}")
    
    # Check if acting as identity
    diff = torch.abs(x_hat - x)
    print(f"Difference: mean={diff.mean():.8f}, max={diff.max():.8f}")
    
    if diff.max() < 1e-6:
        print("❌ BUG: CompressorVNVC acting as identity function!")
    else:
        print("✅ CompressorVNVC có reconstruction loss")
    
    # Check BPP calculation
    batch_size = likelihoods.size(0)
    num_pixels = likelihoods.size(2) * likelihoods.size(3)
    log_likelihoods = torch.log(likelihoods.clamp(min=1e-10))
    bpp = -log_likelihoods.sum() / (batch_size * num_pixels * math.log(2))
    print(f"BPP: {bpp:.6f}")
    
    print()

def debug_analysis_synthesis():
    """Debug Analysis/Synthesis transforms"""
    print("🔍 DEBUGGING: Analysis/Synthesis Transforms")
    print("-" * 50)
    
    compressor = CompressorVNVC(input_channels=128, latent_channels=192, lambda_rd=512)
    
    # Test input
    x = torch.randn(1, 128, 64, 64)
    print(f"Input: {x.shape}")
    
    # Analysis transform
    y = compressor.analysis_transform(x)
    print(f"After analysis: {y.shape}")
    
    # Synthesis transform
    x_recon = compressor.synthesis_transform(y)
    print(f"After synthesis: {x_recon.shape}")
    
    # Check perfect reconstruction without quantization
    mse_perfect = F.mse_loss(x_recon, x).item()
    print(f"MSE (perfect): {mse_perfect:.8f}")
    
    if mse_perfect > 0.1:
        print("❌ Analysis/Synthesis transforms có lỗi!")
    else:
        print("✅ Analysis/Synthesis transforms OK")
    
    print()

def debug_entropy_bottleneck():
    """Debug Entropy Bottleneck"""
    print("🔍 DEBUGGING: Entropy Bottleneck")
    print("-" * 50)
    
    from models.compressor_vnvc import EntropyBottleneck
    
    # Create entropy bottleneck
    entropy_bottleneck = EntropyBottleneck(channels=192)
    
    # Test input
    y = torch.randn(1, 192, 4, 4)
    print(f"Input: shape={y.shape}, range=[{y.min():.4f}, {y.max():.4f}]")
    
    # Forward pass
    y_hat, likelihoods = entropy_bottleneck(y)
    print(f"Output: shape={y_hat.shape}, range=[{y_hat.min():.4f}, {y_hat.max():.4f}]")
    print(f"Likelihoods: shape={likelihoods.shape}, range=[{likelihoods.min():.6f}, {likelihoods.max():.6f}]")
    
    # Check if identity
    diff = torch.abs(y_hat - y)
    print(f"Difference: mean={diff.mean():.8f}, max={diff.max():.8f}")
    
    if diff.max() < 1e-6:
        print("❌ BUG: EntropyBottleneck acting as identity!")
    else:
        print("✅ EntropyBottleneck hoạt động đúng")
    
    print()

def debug_full_pipeline():
    """Debug toàn bộ pipeline Stage 2"""
    print("🔍 DEBUGGING: Full Stage 2 Pipeline")
    print("-" * 50)
    
    # Create models
    wavelet_model = WaveletTransformCNN(input_channels=3, feature_channels=64, wavelet_channels=64)
    adamix_model = AdaMixNet(input_channels=256, C_prime=64, C_mix=128, N=4)
    compressor_model = CompressorVNVC(input_channels=128, latent_channels=192, lambda_rd=512)
    
    # Set to eval/train mode
    wavelet_model.eval()
    adamix_model.train()
    compressor_model.train()
    
    # Test input
    images = torch.randn(1, 3, 256, 256)
    print(f"Input images: {images.shape}")
    
    # Forward pass
    with torch.no_grad():
        wavelet_coeffs = wavelet_model(images)
    print(f"Wavelet coeffs: {wavelet_coeffs.shape}")
    
    mixed_features = adamix_model(wavelet_coeffs)
    print(f"Mixed features: {mixed_features.shape}")
    
    x_hat, likelihoods, y_quantized = compressor_model(mixed_features)
    print(f"Compressed features: {x_hat.shape}")
    
    # Calculate losses
    mse_loss = F.mse_loss(x_hat, mixed_features).item()
    
    batch_size = likelihoods.size(0)
    num_pixels = likelihoods.size(2) * likelihoods.size(3)
    log_likelihoods = torch.log(likelihoods.clamp(min=1e-10))
    bpp = -log_likelihoods.sum() / (batch_size * num_pixels * math.log(2))
    
    lambda_rd = 512
    total_loss = lambda_rd * mse_loss + bpp.item()
    
    print(f"MSE Loss: {mse_loss:.8f}")
    print(f"BPP: {bpp:.6f}")
    print(f"Total Loss: {total_loss:.6f}")
    
    if mse_loss < 1e-6:
        print("❌ CONFIRMED BUG: MSE Loss = 0 trong full pipeline!")
    else:
        print("✅ MSE Loss hoạt động đúng")
    
    print()

def main():
    """Main debugging function"""
    print("=" * 60)
    print("🚨 DEBUGGING COMPRESSOR VNVC MSE = 0 ISSUE")
    print("=" * 60)
    print()
    
    # Run all debug tests
    debug_round_with_noise()
    debug_compressor_forward()
    debug_analysis_synthesis()
    debug_entropy_bottleneck()
    debug_full_pipeline()
    
    print("=" * 60)
    print("🎯 DEBUG COMPLETED - CHECK RESULTS ABOVE")
    print("=" * 60)

if __name__ == "__main__":
    main() 