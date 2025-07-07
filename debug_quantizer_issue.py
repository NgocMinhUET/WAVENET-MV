#!/usr/bin/env python3
"""
Debug Quantizer Issue
Vấn đề: Y quantized = 0 → Quantizer không hoạt động
Giải pháp: Kiểm tra và fix quantizer
"""

import torch
import torch.nn as nn
import numpy as np
import os
from models.compressor_vnvc import CompressorVNVC, QuantizerVNVC, RoundWithNoise
from models.wavelet_transform_cnn import WaveletTransformCNN
from models.adamixnet import AdaMixNet

def debug_quantizer():
    print("🔍 DEBUG QUANTIZER ISSUE")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Test QuantizerVNVC directly
    print("\n🔢 TESTING QUANTIZERVNVC:")
    quantizer = QuantizerVNVC(scale_factor=4.0)
    quantizer.to(device)
    
    # Test with different inputs
    test_inputs = [
        torch.randn(2, 3, 64, 64).to(device),  # Random values
        torch.ones(2, 3, 64, 64).to(device) * 0.5,  # Constant 0.5
        torch.ones(2, 3, 64, 64).to(device) * 0.1,  # Small values
        torch.ones(2, 3, 64, 64).to(device) * 0.01,  # Very small values
    ]
    
    for i, test_input in enumerate(test_inputs):
        print(f"\nTest {i+1}:")
        print(f"  Input range: [{test_input.min():.4f}, {test_input.max():.4f}]")
        print(f"  Input mean: {test_input.mean():.4f}")
        
        # Test quantizer
        with torch.no_grad():
            quantized = quantizer(test_input)
            print(f"  Quantized range: [{quantized.min():.4f}, {quantized.max():.4f}]")
            print(f"  Quantized mean: {quantized.mean():.4f}")
            print(f"  Non-zero ratio: {(quantized != 0).float().mean():.4f}")
            
            # Check unique values
            unique_vals = torch.unique(quantized)
            print(f"  Unique values: {len(unique_vals)}")
            print(f"  First 5 unique: {unique_vals[:5].tolist()}")
    
    # 2. Test RoundWithNoise function
    print("\n🎲 TESTING ROUNDWITHNOISE:")
    
    # Test in training mode
    test_input = torch.randn(2, 3, 64, 64).to(device)
    test_input.requires_grad_(True)
    
    print(f"Training mode test:")
    print(f"  Input range: [{test_input.min():.4f}, {test_input.max():.4f}]")
    
    # Forward pass
    quantized = RoundWithNoise.apply(test_input, 4.0)
    print(f"  Quantized range: [{quantized.min():.4f}, {quantized.max():.4f}]")
    print(f"  Non-zero ratio: {(quantized != 0).float().mean():.4f}")
    
    # Test in eval mode
    test_input.requires_grad_(False)
    quantized_eval = RoundWithNoise.apply(test_input, 4.0)
    print(f"Eval mode test:")
    print(f"  Quantized range: [{quantized_eval.min():.4f}, {quantized_eval.max():.4f}]")
    print(f"  Non-zero ratio: {(quantized_eval != 0).float().mean():.4f}")
    
    # 3. Test with actual compressor
    print("\n🔧 TESTING WITH ACTUAL COMPRESSOR:")
    
    try:
        # Load compressor
        compressor = CompressorVNVC(input_channels=128, latent_channels=192, lambda_rd=128)
        compressor.to(device)
        
        # Test with dummy input
        dummy_input = torch.randn(2, 128, 64, 64).to(device)
        
        print(f"Dummy input range: [{dummy_input.min():.4f}, {dummy_input.max():.4f}]")
        
        with torch.no_grad():
            # Analysis transform
            y = compressor.analysis_transform(dummy_input)
            print(f"Analysis output range: [{y.min():.4f}, {y.max():.4f}]")
            
            # Quantization
            y_quantized = compressor.quantizer(y)
            print(f"Quantized range: [{y_quantized.min():.4f}, {y_quantized.max():.4f}]")
            print(f"Non-zero ratio: {(y_quantized != 0).float().mean():.4f}")
            
            # Check if quantizer is the issue
            if y_quantized.abs().max() < 1e-6:
                print("❌ QUANTIZER ISSUE CONFIRMED: All values quantized to 0")
            else:
                print("✅ Quantizer seems to be working")
                
    except Exception as e:
        print(f"❌ Error testing compressor: {e}")
    
    # 4. Root Cause Analysis
    print("\n🔍 ROOT CAUSE ANALYSIS:")
    print("Y quantized = 0 suggests:")
    print("1. ❌ Quantizer scale_factor too small")
    print("2. ❌ Input values too small for quantization")
    print("3. ❌ RoundWithNoise implementation bug")
    print("4. ❌ Scale factor not applied correctly")
    
    # 5. Suggested Fixes
    print("\n🛠️ SUGGESTED FIXES:")
    print("1. Increase quantizer scale_factor:")
    print("   - Current: 4.0")
    print("   - Try: 10.0, 20.0, 100.0")
    
    print("\n2. Check input normalization:")
    print("   - Ensure input values are in reasonable range")
    print("   - Check if wavelet/compressor outputs are too small")
    
    print("\n3. Fix RoundWithNoise implementation:")
    print("   - Ensure scale_factor is applied correctly")
    print("   - Check forward/backward passes")
    
    print("\n4. Alternative quantization:")
    print("   - Use straight-through estimator")
    print("   - Use different quantization scheme")
    
    return True

def test_quantizer_fixes():
    """Test different quantizer fixes"""
    print("\n🧪 TESTING QUANTIZER FIXES:")
    print("=" * 30)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different scale factors
    scale_factors = [1.0, 4.0, 10.0, 20.0, 100.0]
    test_input = torch.randn(2, 3, 64, 64).to(device) * 0.1  # Small values like in real data
    
    print(f"Test input range: [{test_input.min():.4f}, {test_input.max():.4f}]")
    
    for scale in scale_factors:
        print(f"\nScale factor: {scale}")
        
        # Test RoundWithNoise
        quantized = RoundWithNoise.apply(test_input, scale)
        non_zero_ratio = (quantized != 0).float().mean()
        unique_count = len(torch.unique(quantized))
        
        print(f"  Non-zero ratio: {non_zero_ratio:.4f}")
        print(f"  Unique values: {unique_count}")
        print(f"  Range: [{quantized.min():.4f}, {quantized.max():.4f}]")
        
        if non_zero_ratio > 0.1:
            print(f"  ✅ Scale {scale} works!")
        else:
            print(f"  ❌ Scale {scale} still produces mostly zeros")

if __name__ == "__main__":
    debug_quantizer()
    test_quantizer_fixes()
    
    print("\n" + "=" * 50)
    print("🔍 QUANTIZER DEBUG COMPLETE")
    print("\n💡 RECOMMENDATION:")
    print("The quantizer is producing all zeros, which explains:")
    print("1. BPP = 48.0 (cannot compress zeros efficiently)")
    print("2. PSNR = 6.87 dB (reconstruction from zeros)")
    print("3. Lambda not affecting results (always zeros)")
    print("\nNext steps:")
    print("1. Increase quantizer scale_factor")
    print("2. Check input normalization")
    print("3. Retrain Stage 2 with fixed quantizer") 