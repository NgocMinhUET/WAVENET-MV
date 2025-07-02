"""
Debug WAVENET-MV Evaluation Zero Results
Tìm nguyên nhân tại sao evaluation trả về 0.00 cho tất cả metrics
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory
sys.path.append(str(Path(__file__).parent))

from models.wavelet_transform_cnn import WaveletTransformCNN
from models.adamixnet import AdaMixNet
from models.compressor_vnvc import MultiLambdaCompressorVNVC
from datasets.dataset_loaders import COCODatasetLoader

def debug_model_loading():
    """Debug model loading and basic functionality"""
    print("🔍 DEBUGGING MODEL LOADING...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Check checkpoint exists
    checkpoint_path = "checkpoints/stage3_ai_heads_coco_best.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"✅ Checkpoint exists: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"✅ Checkpoint loaded successfully")
        print(f"Keys in checkpoint: {list(checkpoint.keys())}")
        
        # Check individual state dicts
        for key in ['wavelet_state_dict', 'adamixnet_state_dict', 'compressor_state_dict']:
            if key in checkpoint:
                print(f"✅ {key} found with {len(checkpoint[key])} parameters")
            else:
                print(f"❌ {key} missing from checkpoint")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

def debug_model_initialization():
    """Debug model initialization"""
    print("\n🔍 DEBUGGING MODEL INITIALIZATION...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Initialize models
        wavelet_cnn = WaveletTransformCNN(
            input_channels=3,
            feature_channels=64,
            wavelet_channels=64
        ).to(device)
        print(f"✅ WaveletTransformCNN initialized")
        
        adamixnet = AdaMixNet(
            input_channels=256,  # 4 * 64
            C_prime=64,
            C_mix=128
        ).to(device)
        print(f"✅ AdaMixNet initialized")
        
        compressor = MultiLambdaCompressorVNVC(
            input_channels=128,
            latent_channels=192
        ).to(device)
        print(f"✅ MultiLambdaCompressorVNVC initialized")
        
        return wavelet_cnn, adamixnet, compressor
        
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        return None, None, None

def debug_forward_pass():
    """Debug forward pass with dummy data"""
    print("\n🔍 DEBUGGING FORWARD PASS...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get models
    wavelet_cnn, adamixnet, compressor = debug_model_initialization()
    if wavelet_cnn is None:
        return False
    
    try:
        # Create dummy input
        dummy_input = torch.randn(2, 3, 256, 256).to(device)
        print(f"✅ Dummy input created: {dummy_input.shape}")
        
        # Forward through wavelet
        with torch.no_grad():
            wavelet_coeffs = wavelet_cnn(dummy_input)
            print(f"✅ Wavelet forward: {wavelet_coeffs.shape}")
            print(f"   Range: [{wavelet_coeffs.min():.3f}, {wavelet_coeffs.max():.3f}]")
            
            # Forward through AdaMixNet
            mixed_features = adamixnet(wavelet_coeffs)
            print(f"✅ AdaMixNet forward: {mixed_features.shape}")
            print(f"   Range: [{mixed_features.min():.3f}, {mixed_features.max():.3f}]")
            
            # Set lambda and forward through compressor
            compressor.set_lambda(256)
            x_hat, likelihoods, y_quantized = compressor(mixed_features)
            print(f"✅ Compressor forward:")
            print(f"   x_hat: {x_hat.shape}, range: [{x_hat.min():.3f}, {x_hat.max():.3f}]")
            print(f"   y_quantized: {y_quantized.shape}, range: [{y_quantized.min():.3f}, {y_quantized.max():.3f}]")
            print(f"   likelihoods: {likelihoods.shape}, range: [{likelihoods.min():.6f}, {likelihoods.max():.6f}]")
            
            # Check for all zeros
            if torch.all(x_hat == 0):
                print("❌ x_hat is all zeros!")
            if torch.all(y_quantized == 0):
                print("❌ y_quantized is all zeros!")
            if torch.all(likelihoods == 0):
                print("❌ likelihoods is all zeros!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_metrics_calculation():
    """Debug metrics calculation"""
    print("\n🔍 DEBUGGING METRICS CALCULATION...")
    
    # Test PSNR calculation
    try:
        from evaluation.codec_metrics import calculate_psnr, calculate_ms_ssim
        
        # Create test images
        img1 = torch.randn(1, 3, 64, 64)
        img2 = img1 + 0.1 * torch.randn_like(img1)  # Add some noise
        
        psnr = calculate_psnr(img1, img2)
        print(f"✅ PSNR calculation works: {psnr:.2f} dB")
        
        ms_ssim = calculate_ms_ssim(img1, img2)
        print(f"✅ MS-SSIM calculation works: {ms_ssim:.4f}")
        
        # Test with identical images
        psnr_identical = calculate_psnr(img1, img1)
        print(f"✅ PSNR for identical images: {psnr_identical}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in metrics calculation: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_dataset_loading():
    """Debug dataset loading"""
    print("\n🔍 DEBUGGING DATASET LOADING...")
    
    try:
        dataset = COCODatasetLoader(
            data_dir='datasets/COCO',
            subset='val',
            image_size=256,
            augmentation=False
        )
        print(f"✅ Dataset loaded: {len(dataset)} images")
        
        # Test loading one sample
        sample = dataset[0]
        print(f"✅ Sample loaded:")
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        print(f"   Image dtype: {sample['image'].dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_bpp_calculation():
    """Debug BPP calculation specifically"""
    print("\n🔍 DEBUGGING BPP CALCULATION...")
    
    try:
        from evaluation.codec_metrics import estimate_bpp_from_features
        
        # Test BPP calculation
        dummy_features = torch.randn(2, 192, 16, 16)  # Typical compressed features
        image_shape = (256, 256)
        
        bpp = estimate_bpp_from_features(dummy_features, image_shape)
        print(f"✅ BPP calculation: {bpp:.4f}")
        
        if bpp == 0.0:
            print("❌ BPP is zero - this is the problem!")
        else:
            print(f"✅ BPP is non-zero: {bpp:.4f}")
        
        return bpp != 0.0
        
    except Exception as e:
        print(f"❌ Error in BPP calculation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debug tests"""
    print("🔍 DEBUGGING WAVENET-MV EVALUATION ZERO RESULTS")
    print("="*60)
    
    tests = [
        ("Model Loading", debug_model_loading),
        ("Model Initialization", debug_model_initialization),
        ("Forward Pass", debug_forward_pass),
        ("Metrics Calculation", debug_metrics_calculation),
        ("Dataset Loading", debug_dataset_loading),
        ("BPP Calculation", debug_bpp_calculation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            if test_name == "Model Initialization":
                # Skip this as it's called by forward pass
                continue
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "="*60)
    print("DEBUG SUMMARY:")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    # Identify likely root cause
    if not results.get("Model Loading", False):
        print("\n🎯 ROOT CAUSE: Checkpoint loading failed")
    elif not results.get("Forward Pass", False):
        print("\n🎯 ROOT CAUSE: Model forward pass failed")
    elif not results.get("BPP Calculation", False):
        print("\n🎯 ROOT CAUSE: BPP calculation returns zero")
    elif not results.get("Metrics Calculation", False):
        print("\n🎯 ROOT CAUSE: Metrics calculation failed")
    else:
        print("\n🤔 ROOT CAUSE: Unknown - all individual tests passed")
    
    print("\n📋 NEXT STEPS:")
    print("1. Fix the identified root cause")
    print("2. Re-run generate_paper_results.py")
    print("3. Verify non-zero metrics")

if __name__ == '__main__':
    main() 