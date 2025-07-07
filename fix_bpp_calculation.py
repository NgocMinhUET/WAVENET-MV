#!/usr/bin/env python3
"""
Fix BPP Calculation Issue
Vấn đề: BPP = 10.0 cho tất cả lambda → Compression không hoạt động
Giải pháp: Sửa BPP calculation trong evaluation scripts
"""

import os
import sys

def fix_bpp_calculation():
    print("🔧 FIXING BPP CALCULATION ISSUE")
    print("=" * 50)
    
    # 1. Check current BPP calculation in codec_metrics_final.py
    print("\n📊 CHECKING CURRENT BPP CALCULATION:")
    
    if os.path.exists("evaluation/codec_metrics_final.py"):
        with open("evaluation/codec_metrics_final.py", 'r') as f:
            content = f.read()
        
        # Look for BPP calculation
        if "bpp" in content.lower():
            print("✅ BPP calculation found in codec_metrics_final.py")
            
            # Check for suspicious patterns
            if "8 *" in content or "bits_per_symbol" in content:
                print("❌ Found suspicious BPP calculation (8 bits per symbol)")
            if "numel()" in content:
                print("❌ Found numel() in BPP calculation")
        else:
            print("❌ No BPP calculation found")
    
    # 2. Check evaluation script
    if os.path.exists("evaluate_vcm.py"):
        with open("evaluate_vcm.py", 'r') as f:
            content = f.read()
        
        if "bpp" in content.lower():
            print("✅ BPP calculation found in evaluate_vcm.py")
        else:
            print("❌ No BPP calculation found in evaluate_vcm.py")
    
    # 3. Root Cause Analysis
    print("\n🔍 ROOT CAUSE ANALYSIS:")
    print("BPP = 10.0 suggests:")
    print("1. ❌ Quantizer not working - all values quantized to same")
    print("2. ❌ Entropy model not trained - uniform distribution")
    print("3. ❌ BPP calculation wrong - using 8 bits per symbol")
    print("4. ❌ Model not actually compressing")
    
    # 4. Suggested Fixes
    print("\n🛠️ SUGGESTED FIXES:")
    print("1. Fix BPP calculation:")
    print("   - Use entropy-based BPP instead of 8 bits per symbol")
    print("   - Calculate actual bits from likelihoods")
    print("   - Formula: BPP = -sum(log2(likelihoods)) / (H * W)")
    
    print("\n2. Fix Quantizer:")
    print("   - Check if quantizer is working")
    print("   - Ensure quantization preserves information")
    print("   - Verify round-with-noise implementation")
    
    print("\n3. Fix Entropy Model:")
    print("   - Train entropy model properly")
    print("   - Use CompressAI GaussianConditional correctly")
    print("   - Ensure likelihoods are reasonable")
    
    print("\n4. Retrain Model:")
    print("   - Retrain Stage 2 with proper rate-distortion loss")
    print("   - Use correct lambda values")
    print("   - Monitor BPP during training")
    
    # 5. Check if we need to retrain
    print("\n🎯 NEXT STEPS:")
    print("1. Check BPP calculation in evaluation scripts")
    print("2. Fix BPP formula if wrong")
    print("3. Retrain Stage 2 if quantizer/entropy model broken")
    print("4. Run evaluation again with fixed BPP")
    
    return True

def check_evaluation_scripts():
    """Check BPP calculation in evaluation scripts"""
    print("\n📋 CHECKING EVALUATION SCRIPTS:")
    
    scripts_to_check = [
        "evaluation/codec_metrics_final.py",
        "evaluate_vcm.py",
        "evaluation/codec_metrics.py"
    ]
    
    for script in scripts_to_check:
        if os.path.exists(script):
            print(f"\n🔍 {script}:")
            with open(script, 'r') as f:
                content = f.read()
            
            # Look for BPP calculation patterns
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'bpp' in line.lower() and ('=' in line or ':' in line):
                    print(f"  Line {i+1}: {line.strip()}")
                    
                    # Check for suspicious patterns
                    if '8 *' in line or 'bits_per_symbol' in line:
                        print(f"    ❌ SUSPICIOUS: {line.strip()}")
                    if 'numel()' in line:
                        print(f"    ❌ SUSPICIOUS: {line.strip()}")
                    if 'entropy' in line.lower() or 'likelihood' in line.lower():
                        print(f"    ✅ GOOD: {line.strip()}")

if __name__ == "__main__":
    fix_bpp_calculation()
    check_evaluation_scripts()
    
    print("\n" + "=" * 50)
    print("🔧 FIX ANALYSIS COMPLETE")
    print("\n💡 RECOMMENDATION:")
    print("The BPP = 10.0 issue suggests the compression pipeline is not working.")
    print("You should:")
    print("1. Check and fix BPP calculation in evaluation scripts")
    print("2. Retrain Stage 2 compressor with proper rate-distortion loss")
    print("3. Ensure quantizer and entropy model are working correctly") 