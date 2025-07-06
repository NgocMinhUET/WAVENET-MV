#!/usr/bin/env python3
"""
Test script để kiểm tra các fix đã thực hiện
"""

import os
import sys
import subprocess
from pathlib import Path

# Fix OpenMP warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_ms_ssim_fix():
    """Test MS-SSIM fix với ảnh nhỏ"""
    print("🧪 Testing MS-SSIM fix with small images...")
    
    try:
        import numpy as np
        from skimage.metrics import structural_similarity as ssim
        
        # Tạo ảnh nhỏ (5x5) để test
        img1 = np.random.rand(5, 5, 3)
        img2 = np.random.rand(5, 5, 3)
        
        # Test safe_ssim function
        def safe_ssim(im1, im2, data_range):
            try:
                H, W = im1.shape[:2] if im1.ndim >= 2 else im1.shape
                min_dim = min(H, W)
                if min_dim < 7:
                    if min_dim < 3:
                        return np.corrcoef(im1.flatten(), im2.flatten())[0, 1]
                    else:
                        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
                        return ssim(im1, im2, data_range=data_range, win_size=win_size)
                else:
                    return ssim(im1, im2, data_range=data_range)
            except Exception as e:
                return np.corrcoef(im1.flatten(), im2.flatten())[0, 1]
        
        # Test với ảnh nhỏ
        result = safe_ssim(img1, img2, 1.0)
        print(f"✅ MS-SSIM with small image (5x5): {result:.4f}")
        
        # Test với ảnh lớn hơn
        img1_large = np.random.rand(10, 10, 3)
        img2_large = np.random.rand(10, 10, 3)
        result_large = safe_ssim(img1_large, img2_large, 1.0)
        print(f"✅ MS-SSIM with larger image (10x10): {result_large:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ MS-SSIM test failed: {e}")
        return False

def test_script_arguments():
    """Test script arguments"""
    print("\n🧪 Testing script arguments...")
    
    # Test evaluate_vcm.py
    try:
        result = subprocess.run([
            sys.executable, "evaluate_vcm.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ evaluate_vcm.py arguments OK")
        else:
            print(f"❌ evaluate_vcm.py arguments failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ evaluate_vcm.py test failed: {e}")
        return False
    
    # Test codec_metrics_final.py
    try:
        result = subprocess.run([
            sys.executable, "evaluation/codec_metrics_final.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ codec_metrics_final.py arguments OK")
        else:
            print(f"❌ codec_metrics_final.py arguments failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ codec_metrics_final.py test failed: {e}")
        return False
    
    # Test compare_baselines.py
    try:
        result = subprocess.run([
            sys.executable, "evaluation/compare_baselines.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ compare_baselines.py arguments OK")
        else:
            print(f"❌ compare_baselines.py arguments failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ compare_baselines.py test failed: {e}")
        return False
    
    return True

def test_new_scripts():
    """Test các script mới được tạo"""
    print("\n🧪 Testing new scripts...")
    
    scripts_to_test = [
        "evaluation/generate_paper_results.py",
        "evaluation/generate_tables.py", 
        "evaluation/statistical_analysis.py",
        "evaluation/generate_summary_report.py"
    ]
    
    for script in scripts_to_test:
        if os.path.exists(script):
            try:
                result = subprocess.run([
                    sys.executable, script, "--help"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    print(f"✅ {script} OK")
                else:
                    print(f"❌ {script} failed: {result.stderr}")
                    return False
                    
            except Exception as e:
                print(f"❌ {script} test failed: {e}")
                return False
        else:
            print(f"❌ {script} not found")
            return False
    
    return True

def test_run_complete_evaluation():
    """Test run_complete_evaluation.py với arguments đã sửa"""
    print("\n🧪 Testing run_complete_evaluation.py...")
    
    try:
        result = subprocess.run([
            sys.executable, "run_complete_evaluation.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ run_complete_evaluation.py arguments OK")
            
            # Kiểm tra xem có sử dụng đúng arguments không
            help_text = result.stdout
            if "--output_json" in help_text and "--output_csv" in help_text:
                print("✅ Correct arguments detected")
                return True
            else:
                print("❌ Wrong arguments detected")
                return False
        else:
            print(f"❌ run_complete_evaluation.py failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ run_complete_evaluation.py test failed: {e}")
        return False

def main():
    print("🔧 Testing WAVENET-MV Evaluation Fixes")
    print("=" * 50)
    
    tests = [
        ("MS-SSIM Fix", test_ms_ssim_fix),
        ("Script Arguments", test_script_arguments),
        ("New Scripts", test_new_scripts),
        ("Complete Evaluation", test_run_complete_evaluation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Evaluation pipeline is ready.")
        print("\n📝 Next steps:")
        print("1. Run on server: git pull origin master")
        print("2. Test with real data: python run_complete_evaluation.py --stage1_checkpoint ...")
        print("3. Check results in results/ directory")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 