#!/usr/bin/env python3
"""
Debug Script cho MSE = 0 issue trong Stage 2 Training
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# Add parent directory
sys.path.append('.')

from models.wavelet_transform_cnn import WaveletTransformCNN
from models.adamixnet import AdaMixNet
from models.compressor_vnvc import CompressorVNVC

def debug_mse_zero():
    """Debug tại sao MSE = 0"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 DEBUG MSE = 0 trên {device}")
    print("="*60)
    
    # Setup models giống như training
    print("📦 Setting up models...")
    
    # WaveletCNN (frozen)
    wavelet_model = WaveletTransformCNN(
        input_channels=3,
        feature_channels=64,
        wavelet_channels=64
    ).to(device).eval()
    
    # AdaMixNet
    adamix_model = AdaMixNet(
        input_channels=256,  # 4*64
        C_prime=64,
        C_mix=128,
        N=4
    ).to(device)
    
    # CompressorVNVC
    compressor_model = CompressorVNVC(
        input_channels=128,
        latent_channels=192,
        lambda_rd=512
    ).to(device)
    
    print("✅ Models created")
    
    # Test với dummy data giống training
    print("\n🧪 Testing với dummy data...")
    batch_size = 8  # Giống training
    image_size = 256
    
    images = torch.randn(batch_size, 3, image_size, image_size).to(device)
    print(f"Input images: {images.shape}")
    
    with torch.no_grad():
        with autocast():
            # Stage 1: Wavelet transform
            wavelet_coeffs = wavelet_model(images)
            print(f"Wavelet coeffs: {wavelet_coeffs.shape}")
            
            # Stage 2a: AdaMixNet
            mixed_features = adamix_model(wavelet_coeffs)
            print(f"Mixed features: {mixed_features.shape}")
            
            # Stage 2b: CompressorVNVC - ĐÂY LÀ CHỖ CÓ VẤN ĐỀ
            print("\n🔬 Testing CompressorVNVC...")
            x_hat, likelihoods, y_quantized = compressor_model(mixed_features)
            compressed_features = x_hat
            
            print(f"📊 SHAPES COMPARISON:")
            print(f"  Mixed features: {mixed_features.shape}")
            print(f"  Compressed (x_hat): {compressed_features.shape}")
            print(f"  Likelihoods: {likelihoods.shape}")
            print(f"  Quantized latents: {y_quantized.shape}")
            
            # Kiểm tra shape match
            shapes_match = mixed_features.shape == compressed_features.shape
            print(f"\n✅ Shapes match: {shapes_match}")
            
            if shapes_match:
                # Tính MSE
                mse_criterion = nn.MSELoss()
                mse_loss = mse_criterion(compressed_features, mixed_features)
                print(f"🎯 MSE Loss: {mse_loss.item():.8f}")
                
                # Kiểm tra tensor statistics
                print(f"\n📈 TENSOR STATISTICS:")
                print(f"  Mixed features:")
                print(f"    - Min: {mixed_features.min().item():.6f}")
                print(f"    - Max: {mixed_features.max().item():.6f}")
                print(f"    - Mean: {mixed_features.mean().item():.6f}")
                print(f"    - Std: {mixed_features.std().item():.6f}")
                
                print(f"  Compressed features:")
                print(f"    - Min: {compressed_features.min().item():.6f}")
                print(f"    - Max: {compressed_features.max().item():.6f}")
                print(f"    - Mean: {compressed_features.mean().item():.6f}")
                print(f"    - Std: {compressed_features.std().item():.6f}")
                
                # Kiểm tra difference
                diff = torch.abs(compressed_features - mixed_features)
                print(f"\n🔍 DIFFERENCE ANALYSIS:")
                print(f"  Absolute difference:")
                print(f"    - Mean: {diff.mean().item():.8f}")
                print(f"    - Max: {diff.max().item():.8f}")
                print(f"    - Min: {diff.min().item():.8f}")
                print(f"    - Std: {diff.std().item():.8f}")
                
                # Phân tích nguyên nhân
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                
                print(f"\n🚨 DIAGNOSIS:")
                if max_diff < 1e-6:
                    print("❌ BUG FOUND: Tensors are nearly IDENTICAL!")
                    print("   -> CompressorVNVC đang hoạt động như IDENTITY FUNCTION")
                    print("   -> Không có compression/decompression thực sự")
                elif max_diff < 1e-3:
                    print("⚠️  WARNING: Very small differences")
                    print("   -> Compression ratio quá thấp")
                else:
                    print("✅ Normal differences - compression working")
                
                # Kiểm tra gradients
                if compressed_features.requires_grad:
                    print(f"  Gradients enabled: {compressed_features.requires_grad}")
                else:
                    print(f"  Gradients disabled: {compressed_features.requires_grad}")
                
            else:
                print(f"🚨 SHAPE MISMATCH DETECTED!")
                print(f"   Mixed: {mixed_features.shape}")
                print(f"   Compressed: {compressed_features.shape}")
                
                # Try resize
                try:
                    compressed_resized = F.interpolate(
                        compressed_features,
                        size=mixed_features.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    print(f"✅ Resized to: {compressed_resized.shape}")
                    
                    mse_criterion = nn.MSELoss()
                    mse_resized = mse_criterion(compressed_resized, mixed_features)
                    print(f"🎯 MSE after resize: {mse_resized.item():.8f}")
                    
                except Exception as e:
                    print(f"❌ Resize failed: {e}")
    
    print("\n" + "="*60)
    print("🏁 DEBUG COMPLETED")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("1. Nếu MSE ≈ 0 và tensors identical:")
    print("   -> Fix CompressorVNVC architecture")
    print("   -> Check analysis/synthesis transforms")
    print("2. Nếu có shape mismatch:")
    print("   -> Add resize logic trong training")
    print("3. Nếu gradients disabled:")
    print("   -> Check model.train() vs model.eval()")

if __name__ == '__main__':
    debug_mse_zero() 