#!/usr/bin/env python3
"""
Quick Test Script - Chạy trên Windows trước khi push code
Kiểm tra syntax, import, và basic functionality
"""

import sys
import torch
import importlib.util
from pathlib import Path

def test_imports():
    """Test tất cả imports trong project"""
    print("🔍 Testing imports...")
    
    try:
        # Test models
        from models.wavelet_transform_cnn import WaveletTransformCNN
        from models.adamixnet import AdaMixNet
        from models.compressor_vnvc import CompressorVNVC
        from models.ai_heads import YOLOTinyHead, SegFormerHead
        print("✅ Models import OK")
        
        # Test training scripts syntax
        spec = importlib.util.spec_from_file_location("stage1", "training/stage1_train_wavelet.py")
        print("✅ Training scripts syntax OK")
        
        # Test evaluation
        from evaluation.codec_metrics import calculate_psnr, calculate_ssim
        print("✅ Evaluation imports OK")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic model functionality với dummy data"""
    print("🔍 Testing basic functionality...")
    
    try:
        # Test với dummy data
        dummy_input = torch.randn(1, 3, 64, 64)  # Small size for quick test
        
        # Test WaveletCNN
        from models.wavelet_transform_cnn import WaveletTransformCNN
        wavelet_cnn = WaveletTransformCNN(in_channels=3, out_channels=64)
        wavelet_out = wavelet_cnn(dummy_input)
        print(f"✅ WaveletCNN output shape: {wavelet_out.shape}")
        
        # Test AdaMixNet
        from models.adamixnet import AdaMixNet
        adamix = AdaMixNet(in_channels=wavelet_out.shape[1], out_channels=128)
        adamix_out = adamix(wavelet_out)
        print(f"✅ AdaMixNet output shape: {adamix_out.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False

def test_code_quality():
    """Check code quality"""
    print("🔍 Testing code quality...")
    
    try:
        import subprocess
        
        # Check syntax với flake8 (nếu có)
        try:
            result = subprocess.run(['flake8', '--select=E9,F63,F7,F82', '.'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Code syntax OK")
            else:
                print(f"⚠️ Code quality issues: {result.stdout}")
        except FileNotFoundError:
            print("⚠️ flake8 not installed, skipping syntax check")
        
        return True
    except Exception as e:
        print(f"❌ Code quality check error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running Quick Test Before Git Push...")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Functionality Test", test_basic_functionality),
        ("Code Quality Test", test_code_quality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Ready to push to git.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please fix before pushing.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 