#!/usr/bin/env python3
import pandas as pd
import json
import numpy as np
import os

def debug_evaluation_results():
    print("🔍 DEBUG EVALUATION RESULTS")
    print("=" * 50)
    
    # 1. Analyze Codec Metrics
    print("\n📊 CODEC METRICS ANALYSIS:")
    if os.path.exists("results/codec_metrics_final.csv"):
        df = pd.read_csv("results/codec_metrics_final.csv")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Lambda values: {sorted(df['lambda'].unique())}")
        
        print("\n📈 DETAILED STATISTICS:")
        print(f"PSNR range: {df['psnr_db'].min():.2f} - {df['psnr_db'].max():.2f} dB")
        print(f"BPP range: {df['bpp'].min():.3f} - {df['bpp'].max():.3f}")
        print(f"MS-SSIM range: {df['ms_ssim'].min():.4f} - {df['ms_ssim'].max():.4f}")
        
        print("\n📊 BY LAMBDA:")
        for lambda_val in sorted(df['lambda'].unique()):
            lambda_data = df[df['lambda'] == lambda_val]
            print(f"λ={lambda_val}: PSNR={lambda_data['psnr_db'].mean():.2f}dB, BPP={lambda_data['bpp'].mean():.3f}, MS-SSIM={lambda_data['ms_ssim'].mean():.4f}")
        
        # Check for suspicious values
        print("\n🚨 SUSPICIOUS VALUES:")
        if df['bpp'].mean() > 5.0:
            print(f"❌ BPP too high: {df['bpp'].mean():.3f} (should be 0.1-2.0)")
        if df['psnr_db'].mean() < 20.0:
            print(f"❌ PSNR too low: {df['psnr_db'].mean():.2f} dB (should be 25-40 dB)")
        if df['ms_ssim'].mean() < 0.5:
            print(f"❌ MS-SSIM too low: {df['ms_ssim'].mean():.4f} (should be 0.8-0.99)")
        
        # Check lambda variation
        lambda_psnr = df.groupby('lambda')['psnr_db'].mean()
        lambda_bpp = df.groupby('lambda')['bpp'].mean()
        print(f"\n📊 LAMBDA VARIATION:")
        print(f"PSNR variation: {lambda_psnr.max() - lambda_psnr.min():.2f} dB")
        print(f"BPP variation: {lambda_bpp.max() - lambda_bpp.min():.3f}")
        
        if lambda_bpp.max() - lambda_bpp.min() < 0.1:
            print("❌ BPP doesn't vary with lambda - compression not working!")
        
    else:
        print("❌ Codec metrics file not found!")
    
    # 2. Analyze VCM Results
    print("\n🎯 VCM RESULTS ANALYSIS:")
    if os.path.exists("results/vcm_results.json"):
        with open("results/vcm_results.json", 'r') as f:
            vcm_data = json.load(f)
        
        print(f"Keys: {list(vcm_data.keys())}")
        
        if 'metadata' in vcm_data:
            print(f"\n📋 METADATA:")
            for key, value in vcm_data['metadata'].items():
                print(f"  {key}: {value}")
        
        if 'results' in vcm_data:
            results = vcm_data['results']
            print(f"\n📊 RESULTS:")
            
            if 'detection' in results:
                det = results['detection']
                print(f"Detection: {det}")
                if det.get('detection_rate', 0) == 1.0:
                    print("❌ Detection rate = 1.0 - suspicious!")
            
            if 'segmentation' in results:
                seg = results['segmentation']
                print(f"Segmentation: {seg}")
                if seg.get('avg_iou', 0) > 0.99:
                    print("❌ IoU > 0.99 - suspicious!")
                if seg.get('foreground_ratio', 0) == 1.0:
                    print("❌ Foreground ratio = 1.0 - suspicious!")
    
    # 3. Check Baseline Comparison
    print("\n📊 BASELINE COMPARISON:")
    if os.path.exists("results/baseline_comparison.csv"):
        baseline_df = pd.read_csv("results/baseline_comparison.csv")
        print(f"Shape: {baseline_df.shape}")
        print(f"Methods: {baseline_df['method'].unique()}")
        
        print("\n📈 BASELINE STATISTICS:")
        for method in baseline_df['method'].unique():
            method_data = baseline_df[baseline_df['method'] == method]
            print(f"{method}: PSNR={method_data['psnr'].mean():.2f}dB, BPP={method_data['bpp'].mean():.3f}, MS-SSIM={method_data['ms_ssim'].mean():.4f}")
    
    # 4. Root Cause Analysis
    print("\n🔍 ROOT CAUSE ANALYSIS:")
    print("Based on the results, possible issues:")
    
    if os.path.exists("results/codec_metrics_final.csv"):
        df = pd.read_csv("results/codec_metrics_final.csv")
        
        if df['bpp'].mean() > 5.0:
            print("1. ❌ BPP too high (10.0) - Possible causes:")
            print("   - Quantizer not working (all values quantized to same)")
            print("   - Entropy model not trained properly")
            print("   - BPP calculation formula wrong")
            print("   - Model not actually compressing")
        
        if df['psnr_db'].mean() < 20.0:
            print("2. ❌ PSNR too low (6.93 dB) - Possible causes:")
            print("   - Reconstruction quality very poor")
            print("   - Model not trained properly")
            print("   - Input/output mismatch")
            print("   - Quantization destroying information")
        
        if df.groupby('lambda')['bpp'].mean().max() - df.groupby('lambda')['bpp'].mean().min() < 0.1:
            print("3. ❌ Lambda doesn't affect BPP - Possible causes:")
            print("   - Lambda not used in training")
            print("   - Rate-distortion loss not working")
            print("   - Model ignoring lambda parameter")
    
    print("\n" + "=" * 50)
    print("🔍 DEBUG COMPLETE")

if __name__ == "__main__":
    debug_evaluation_results() 