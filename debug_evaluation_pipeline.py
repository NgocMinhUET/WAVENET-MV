"""
Debug script cho evaluation pipeline issues
- PSNR = -1.56dB (should be 20-40dB)
- MS-SSIM = 0.0015 (should be 0.7-0.99)
- BPP = 96.0 (should be 0.5-5.0)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import models
from models.wavelet_transform_cnn import WaveletTransformCNN
from models.adamixnet import AdaMixNet
from models.compressor_vnvc import MultiLambdaCompressorVNVC

def debug_evaluation_pipeline():
    """Debug each component of evaluation pipeline"""
    
    print("🔍 DEBUGGING EVALUATION PIPELINE")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Create test input
    batch_size = 2
    channels = 3  
    height, width = 256, 256
    test_input = torch.randn(batch_size, channels, height, width).to(device)
    test_input = torch.clamp(test_input, 0, 1)  # Simulate normalized images
    
    print(f"\n📊 TEST INPUT:")
    print(f"Shape: {test_input.shape}")
    print(f"Range: [{test_input.min():.4f}, {test_input.max():.4f}]")
    print(f"Mean: {test_input.mean():.4f}, Std: {test_input.std():.4f}")
    
    # 2. Initialize models
    print(f"\n🔧 INITIALIZING MODELS...")
    
    wavelet_cnn = WaveletTransformCNN(
        input_channels=3,
        feature_channels=64, 
        wavelet_channels=64
    ).to(device)
    
    adamixnet = AdaMixNet(
        input_channels=256,  # 4 * 64
        C_prime=64,
        C_mix=128
    ).to(device)
    
    compressor = MultiLambdaCompressorVNVC(
        input_channels=128,
        latent_channels=192
    ).to(device)
    
    # Set to eval mode
    wavelet_cnn.eval()
    adamixnet.eval() 
    compressor.eval()
    compressor.set_lambda(128)
    
    print("✓ Models initialized")
    
    # 3. Test each step
    with torch.no_grad():
        print(f"\n🔍 STEP-BY-STEP DEBUGGING:")
        
        # Step 1: Wavelet transform
        print(f"\n1️⃣ WAVELET TRANSFORM:")
        wavelet_coeffs = wavelet_cnn(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {wavelet_coeffs.shape}")
        print(f"Output range: [{wavelet_coeffs.min():.4f}, {wavelet_coeffs.max():.4f}]")
        print(f"Output mean: {wavelet_coeffs.mean():.4f}, std: {wavelet_coeffs.std():.4f}")
        
        # Step 2: AdaMixNet
        print(f"\n2️⃣ ADAMIXNET:")
        mixed_features = adamixnet(wavelet_coeffs)
        print(f"Input shape: {wavelet_coeffs.shape}")
        print(f"Output shape: {mixed_features.shape}")
        print(f"Output range: [{mixed_features.min():.4f}, {mixed_features.max():.4f}]")
        print(f"Output mean: {mixed_features.mean():.4f}, std: {mixed_features.std():.4f}")
        
        # Step 3: Compressor forward
        print(f"\n3️⃣ COMPRESSOR:")
        x_hat, likelihoods, y_quantized = compressor(mixed_features)
        print(f"Input shape: {mixed_features.shape}")
        print(f"x_hat shape: {x_hat.shape}")
        print(f"x_hat range: [{x_hat.min():.4f}, {x_hat.max():.4f}]")
        print(f"x_hat mean: {x_hat.mean():.4f}, std: {x_hat.std():.4f}")
        print(f"y_quantized shape: {y_quantized.shape}")
        print(f"y_quantized range: [{y_quantized.min():.4f}, {y_quantized.max():.4f}]")
        
        # Step 4: Inverse AdaMixNet
        print(f"\n4️⃣ INVERSE ADAMIXNET:")
        inverse_adamix = torch.nn.Conv2d(128, 256, 1).to(device)
        recovered_coeffs = inverse_adamix(x_hat)
        print(f"Input shape: {x_hat.shape}")
        print(f"Output shape: {recovered_coeffs.shape}")
        print(f"Output range: [{recovered_coeffs.min():.4f}, {recovered_coeffs.max():.4f}]")
        print(f"Output mean: {recovered_coeffs.mean():.4f}, std: {recovered_coeffs.std():.4f}")
        
        # Step 5: Inverse Wavelet
        print(f"\n5️⃣ INVERSE WAVELET:")
        try:
            reconstructed = wavelet_cnn.inverse_transform(recovered_coeffs)
            print(f"Input shape: {recovered_coeffs.shape}")
            print(f"Output shape: {reconstructed.shape}")
            print(f"Output range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
            print(f"Output mean: {reconstructed.mean():.4f}, std: {reconstructed.std():.4f}")
        except Exception as e:
            print(f"❌ Error in inverse wavelet: {e}")
            reconstructed = recovered_coeffs[:, :3]  # Take first 3 channels as fallback
            print(f"Using fallback reconstruction: {reconstructed.shape}")
        
        # Step 6: Size matching
        print(f"\n6️⃣ SIZE MATCHING:")
        if reconstructed.shape != test_input.shape:
            print(f"Resizing from {reconstructed.shape} to {test_input.shape}")
            reconstructed = F.interpolate(
                reconstructed, 
                size=test_input.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        print(f"Final shape: {reconstructed.shape}")
        print(f"Final range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
        
        # Step 7: Metrics calculation
        print(f"\n7️⃣ METRICS CALCULATION:")
        
        # PSNR calculation debug
        mse = F.mse_loss(reconstructed, test_input)
        if mse.item() > 0:
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        else:
            psnr = torch.tensor(float('inf'))
        print(f"MSE: {mse.item():.6f}")
        print(f"PSNR: {psnr.item():.2f} dB")
        
        # BPP calculation debug
        B, C, H_feat, W_feat = y_quantized.shape
        bits_per_feature = 8
        total_feature_bits = C * H_feat * W_feat * bits_per_feature
        H_img, W_img = test_input.shape[2:]
        total_pixels = H_img * W_img
        bpp = total_feature_bits / total_pixels
        
        print(f"Feature shape: {y_quantized.shape}")
        print(f"Feature bits: {total_feature_bits}")
        print(f"Image pixels: {total_pixels}")
        print(f"BPP: {bpp:.4f}")
        
        # Summary
        print(f"\n📋 SUMMARY:")
        print(f"✓ Pipeline completed")
        print(f"✓ PSNR: {psnr.item():.2f} dB")
        print(f"✓ MSE: {mse.item():.6f}")
        print(f"✓ BPP: {bpp:.4f}")
        
        # Check for common issues
        print(f"\n🚨 ISSUE ANALYSIS:")
        if psnr.item() < 10:
            print("❌ PSNR too low - possible reconstruction issues")
        if mse.item() > 1.0:
            print("❌ MSE too high - poor reconstruction quality")
        if bpp > 10:
            print("❌ BPP too high - inefficient compression")
        if torch.isnan(reconstructed).any():
            print("❌ NaN values in reconstruction")
        if torch.isinf(reconstructed).any():
            print("❌ Inf values in reconstruction")
            
        print(f"\n✅ Debug completed")

if __name__ == "__main__":
    debug_evaluation_pipeline() 