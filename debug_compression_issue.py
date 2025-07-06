#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import json
import os
from models.compressor_vnvc import CompressorVNVC
from models.wavelet_transform_cnn import WaveletTransformCNN
from models.adamixnet import AdaMixNet

def debug_compression_pipeline():
    print("🔍 DEBUG COMPRESSION PIPELINE")
    print("=" * 50)
    
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Stage 1: Wavelet - load checkpoint với format đúng
    wavelet = WaveletTransformCNN(input_channels=3, feature_channels=64, wavelet_channels=64)
    wavelet_checkpoint = torch.load('checkpoints/stage1_wavelet_coco_best.pth', map_location=device)
    if 'model_state_dict' in wavelet_checkpoint:
        wavelet.load_state_dict(wavelet_checkpoint['model_state_dict'])
    else:
        wavelet.load_state_dict(wavelet_checkpoint)
    wavelet.eval()
    
    # Stage 2: Compressor - load checkpoint với format đúng
    compressor = CompressorVNVC(input_channels=256, latent_channels=192, lambda_rd=128)  # 4*64=256 channels
    compressor_checkpoint = torch.load('checkpoints/stage2_compressor_coco_lambda128_best.pth', map_location=device)
    if 'compressor_state_dict' in compressor_checkpoint:
        compressor.load_state_dict(compressor_checkpoint['compressor_state_dict'])
    elif 'model_state_dict' in compressor_checkpoint:
        compressor.load_state_dict(compressor_checkpoint['model_state_dict'])
    else:
        compressor.load_state_dict(compressor_checkpoint)
    compressor.eval()
    
    print("✅ Models loaded successfully")
    
    # Test with dummy input
    batch_size = 2
    height, width = 256, 256
    x = torch.randn(batch_size, 3, height, width).to(device)
    
    print(f"\n📊 INPUT ANALYSIS:")
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Input mean: {x.mean():.4f}")
    print(f"Input std: {x.std():.4f}")
    
    # Stage 1: Wavelet transform
    with torch.no_grad():
        wavelet_features = wavelet(x)
        print(f"\n🌊 WAVELET FEATURES:")
        print(f"Wavelet output shape: {wavelet_features.shape}")
        print(f"Wavelet range: [{wavelet_features.min():.4f}, {wavelet_features.max():.4f}]")
        print(f"Wavelet mean: {wavelet_features.mean():.4f}")
        print(f"Wavelet std: {wavelet_features.std():.4f}")
        
        # Check for NaN/Inf
        if torch.isnan(wavelet_features).any():
            print("❌ NaN detected in wavelet features!")
        if torch.isinf(wavelet_features).any():
            print("❌ Inf detected in wavelet features!")
    
    # Stage 2: Compressor
    with torch.no_grad():
        # Analysis transform
        y = compressor.analysis_transform(wavelet_features)
        print(f"\n🔍 ANALYSIS TRANSFORM:")
        print(f"Analysis output shape: {y.shape}")
        print(f"Analysis range: [{y.min():.4f}, {y.max():.4f}]")
        print(f"Analysis mean: {y.mean():.4f}")
        print(f"Analysis std: {y.std():.4f}")
        
        # Quantization
        y_quantized = compressor.quantizer(y)
        print(f"\n🔢 QUANTIZATION:")
        print(f"Quantized shape: {y_quantized.shape}")
        print(f"Quantized range: [{y_quantized.min():.4f}, {y_quantized.max():.4f}]")
        print(f"Quantized mean: {y_quantized.mean():.4f}")
        print(f"Quantized std: {y_quantized.std():.4f}")
        
        # Check unique values
        unique_vals = torch.unique(y_quantized)
        print(f"Unique values count: {len(unique_vals)}")
        print(f"First 10 unique values: {unique_vals[:10].tolist()}")
        
        # Synthesis transform
        x_hat = compressor.synthesis_transform(y_quantized)
        print(f"\n🔄 SYNTHESIS TRANSFORM:")
        print(f"Synthesis output shape: {x_hat.shape}")
        print(f"Synthesis range: [{x_hat.min():.4f}, {x_hat.max():.4f}]")
        print(f"Synthesis mean: {x_hat.mean():.4f}")
        print(f"Synthesis std: {x_hat.std():.4f}")
        
        # Calculate metrics
        mse = nn.MSELoss()(x_hat, wavelet_features)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        print(f"\n📈 METRICS:")
        print(f"MSE: {mse:.6f}")
        print(f"PSNR: {psnr:.2f} dB")
        
        # Calculate BPP (simplified)
        # BPP = bits_per_symbol * num_symbols / (H * W)
        bits_per_symbol = 8  # Assume 8 bits per feature
        num_symbols = y_quantized.numel()
        total_pixels = batch_size * height * width
        bpp = (bits_per_symbol * num_symbols) / total_pixels
        
        print(f"BPP (simplified): {bpp:.3f}")
        
        # Check entropy model
        try:
            # Use entropy_bottleneck instead of entropy_model
            y_hat, likelihoods = compressor.entropy_bottleneck(y_quantized)
            print(f"\n🎲 ENTROPY MODEL:")
            print(f"Likelihoods shape: {likelihoods.shape}")
            print(f"Likelihoods range: [{likelihoods.min():.6f}, {likelihoods.max():.6f}]")
            
            # Calculate actual BPP from entropy
            log_likelihoods = torch.log(likelihoods + 1e-10)
            total_bits = -torch.sum(log_likelihoods) / torch.log(torch.tensor(2.0))
            actual_bpp = total_bits / total_pixels
            print(f"Actual BPP from entropy: {actual_bpp:.3f}")
            
        except Exception as e:
            print(f"❌ Entropy model error: {e}")

if __name__ == "__main__":
    debug_compression_pipeline() 