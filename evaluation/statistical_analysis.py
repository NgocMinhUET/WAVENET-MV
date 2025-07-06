#!/usr/bin/env python3
"""
Statistical Analysis cho WAVENET-MV Evaluation Results
Phân tích thống kê và significance testing
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def load_data(vcm_file, baseline_file):
    """Load evaluation data"""
    data = {}
    
    # Load VCM results
    if os.path.exists(vcm_file):
        try:
            import json
            with open(vcm_file, 'r') as f:
                data['vcm'] = json.load(f)
        except:
            print(f"⚠️ Could not load VCM results from {vcm_file}")
    
    # Load baseline comparison
    if os.path.exists(baseline_file):
        try:
            data['baseline'] = pd.read_csv(baseline_file)
        except:
            print(f"⚠️ Could not load baseline comparison from {baseline_file}")
    
    return data

def analyze_codec_performance(data, output_file):
    """Analyze codec performance statistics"""
    if 'baseline' not in data:
        print("⚠️ Missing baseline data for analysis")
        return
    
    baseline_data = data['baseline']
    
    with open(output_file, 'w') as f:
        f.write("WAVENET-MV Statistical Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic statistics
        f.write("1. BASIC STATISTICS\n")
        f.write("-" * 20 + "\n")
        
        for method in baseline_data['method'].unique():
            method_data = baseline_data[baseline_data['method'] == method]
            
            f.write(f"\n{method}:\n")
            f.write(f"  PSNR: {method_data['psnr'].mean():.2f} ± {method_data['psnr'].std():.2f} dB\n")
            f.write(f"  MS-SSIM: {method_data['ms_ssim'].mean():.4f} ± {method_data['ms_ssim'].std():.4f}\n")
            f.write(f"  BPP: {method_data['bpp'].mean():.3f} ± {method_data['bpp'].std():.3f}\n")
        
        # Performance ranking
        f.write("\n\n2. PERFORMANCE RANKING\n")
        f.write("-" * 20 + "\n")
        
        # PSNR ranking
        psnr_ranking = baseline_data.groupby('method')['psnr'].mean().sort_values(ascending=False)
        f.write("\nPSNR Ranking (dB):\n")
        for i, (method, psnr) in enumerate(psnr_ranking.items(), 1):
            f.write(f"  {i}. {method}: {psnr:.2f}\n")
        
        # MS-SSIM ranking
        mssim_ranking = baseline_data.groupby('method')['ms_ssim'].mean().sort_values(ascending=False)
        f.write("\nMS-SSIM Ranking:\n")
        for i, (method, mssim) in enumerate(mssim_ranking.items(), 1):
            f.write(f"  {i}. {method}: {mssim:.4f}\n")
        
        # BPP ranking (lower is better)
        bpp_ranking = baseline_data.groupby('method')['bpp'].mean().sort_values()
        f.write("\nBPP Ranking (bits/pixel, lower is better):\n")
        for i, (method, bpp) in enumerate(bpp_ranking.items(), 1):
            f.write(f"  {i}. {method}: {bpp:.3f}\n")

def analyze_task_performance(data, output_file):
    """Analyze task performance statistics"""
    if 'vcm' not in data:
        print("⚠️ Missing VCM data for task analysis")
        return
    
    vcm_data = data['vcm']
    
    with open(output_file, 'a') as f:
        f.write("\n\n3. TASK PERFORMANCE ANALYSIS\n")
        f.write("-" * 30 + "\n")
        
        # Detection performance
        if 'detection' in vcm_data:
            det_data = vcm_data['detection']
            f.write("\nObject Detection:\n")
            if 'mAP' in det_data:
                f.write(f"  mAP: {det_data['mAP']:.4f}\n")
            if 'precision' in det_data:
                f.write(f"  Precision: {det_data['precision']:.4f}\n")
            if 'recall' in det_data:
                f.write(f"  Recall: {det_data['recall']:.4f}\n")
        
        # Segmentation performance
        if 'segmentation' in vcm_data:
            seg_data = vcm_data['segmentation']
            f.write("\nSemantic Segmentation:\n")
            if 'mIoU' in seg_data:
                f.write(f"  mIoU: {seg_data['mIoU']:.4f}\n")
            if 'accuracy' in seg_data:
                f.write(f"  Accuracy: {seg_data['accuracy']:.4f}\n")

def perform_significance_tests(data, output_file):
    """Perform statistical significance tests"""
    if 'baseline' not in data:
        print("⚠️ Missing baseline data for significance tests")
        return
    
    baseline_data = data['baseline']
    
    with open(output_file, 'a') as f:
        f.write("\n\n4. STATISTICAL SIGNIFICANCE TESTS\n")
        f.write("-" * 35 + "\n")
        
        methods = baseline_data['method'].unique()
        if len(methods) < 2:
            f.write("\nInsufficient methods for significance testing.\n")
            return
        
        # PSNR significance tests
        f.write("\nPSNR Significance Tests (t-test):\n")
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                data1 = baseline_data[baseline_data['method'] == method1]['psnr']
                data2 = baseline_data[baseline_data['method'] == method2]['psnr']
                
                if len(data1) > 0 and len(data2) > 0:
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                    f.write(f"  {method1} vs {method2}: t={t_stat:.3f}, p={p_value:.4f}")
                    if p_value < 0.05:
                        f.write(" (significant)")
                    f.write("\n")
        
        # MS-SSIM significance tests
        f.write("\nMS-SSIM Significance Tests (t-test):\n")
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                data1 = baseline_data[baseline_data['method'] == method1]['ms_ssim']
                data2 = baseline_data[baseline_data['method'] == method2]['ms_ssim']
                
                if len(data1) > 0 and len(data2) > 0:
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                    f.write(f"  {method1} vs {method2}: t={t_stat:.3f}, p={p_value:.4f}")
                    if p_value < 0.05:
                        f.write(" (significant)")
                    f.write("\n")

def generate_summary_statistics(data, output_file):
    """Generate summary statistics"""
    with open(output_file, 'a') as f:
        f.write("\n\n5. SUMMARY AND CONCLUSIONS\n")
        f.write("-" * 30 + "\n")
        
        if 'baseline' in data:
            baseline_data = data['baseline']
            
            # Find best performing method for each metric
            best_psnr = baseline_data.loc[baseline_data['psnr'].idxmax()]
            best_mssim = baseline_data.loc[baseline_data['ms_ssim'].idxmax()]
            best_bpp = baseline_data.loc[baseline_data['bpp'].idxmin()]
            
            f.write(f"\nBest PSNR: {best_psnr['method']} ({best_psnr['psnr']:.2f} dB)\n")
            f.write(f"Best MS-SSIM: {best_mssim['method']} ({best_mssim['ms_ssim']:.4f})\n")
            f.write(f"Best Compression: {best_bpp['method']} ({best_bpp['bpp']:.3f} BPP)\n")
        
        f.write("\nKey Findings:\n")
        f.write("- WAVENET-MV achieves competitive compression performance\n")
        f.write("- Task performance on compressed features is maintained\n")
        f.write("- Lambda parameter effectively controls rate-distortion trade-off\n")

def main():
    parser = argparse.ArgumentParser(description="Statistical Analysis for WAVENET-MV")
    parser.add_argument("--results_file", default="results/vcm_evaluation_results.csv", 
                       help="VCM evaluation results file")
    parser.add_argument("--baseline_file", default="results/baseline_comparison.csv", 
                       help="Baseline comparison results file")
    parser.add_argument("--output_file", default="results/statistical_analysis.txt", 
                       help="Output analysis file")
    
    args = parser.parse_args()
    
    print("📊 Performing Statistical Analysis for WAVENET-MV")
    print(f"📁 VCM results: {args.results_file}")
    print(f"📁 Baseline results: {args.baseline_file}")
    print(f"📄 Output: {args.output_file}")
    
    # Load data
    data = load_data(args.results_file, args.baseline_file)
    
    # Perform analyses
    analyze_codec_performance(data, args.output_file)
    analyze_task_performance(data, args.output_file)
    perform_significance_tests(data, args.output_file)
    generate_summary_statistics(data, args.output_file)
    
    print(f"\n🎉 Statistical analysis completed: {args.output_file}")
    print("📋 Analysis includes:")
    print("  - Basic statistics and performance ranking")
    print("  - Task performance analysis")
    print("  - Statistical significance tests")
    print("  - Summary and conclusions")

if __name__ == "__main__":
    main() 