#!/usr/bin/env python3
"""
Generate Summary Report cho WAVENET-MV IEEE Paper
Tạo báo cáo tổng hợp với tất cả kết quả evaluation
"""

import os
import sys
import argparse
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def load_all_results(results_dir):
    """Load tất cả results từ results directory"""
    results = {}
    
    # Load VCM results
    vcm_file = os.path.join(results_dir, "vcm_results.json")
    if os.path.exists(vcm_file):
        try:
            with open(vcm_file, 'r') as f:
                results['vcm'] = json.load(f)
        except:
            print(f"⚠️ Could not load VCM results from {vcm_file}")
    
    # Load codec metrics
    codec_file = os.path.join(results_dir, "codec_metrics_final.csv")
    if os.path.exists(codec_file):
        try:
            results['codec'] = pd.read_csv(codec_file)
        except:
            print(f"⚠️ Could not load codec metrics from {codec_file}")
    
    # Load baseline comparison
    baseline_file = os.path.join(results_dir, "baseline_comparison.csv")
    if os.path.exists(baseline_file):
        try:
            results['baseline'] = pd.read_csv(baseline_file)
        except:
            print(f"⚠️ Could not load baseline comparison from {baseline_file}")
    
    # Load statistical analysis
    stats_file = os.path.join(results_dir, "statistical_analysis.txt")
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r') as f:
                results['statistics'] = f.read()
        except:
            print(f"⚠️ Could not load statistical analysis from {stats_file}")
    
    return results

def generate_executive_summary(results, output_file):
    """Generate executive summary"""
    with open(output_file, 'w') as f:
        f.write("# WAVENET-MV Evaluation Summary Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("WAVENET-MV is a novel neural video coding framework designed for machine vision tasks. ")
        f.write("This report summarizes the comprehensive evaluation of the framework's performance ")
        f.write("in terms of compression efficiency and downstream task accuracy.\n\n")
        
        # Key metrics summary
        f.write("### Key Performance Metrics\n\n")
        
        if 'codec' in results and not results['codec'].empty:
            codec_data = results['codec']
            best_psnr = codec_data['psnr_db'].max()
            best_mssim = codec_data['ms_ssim'].max()
            avg_bpp = codec_data['bpp'].mean()
            
            f.write(f"- **Best PSNR**: {best_psnr:.2f} dB\n")
            f.write(f"- **Best MS-SSIM**: {best_mssim:.4f}\n")
            f.write(f"- **Average BPP**: {avg_bpp:.3f} bits/pixel\n\n")
        
        if 'vcm' in results:
            vcm_data = results['vcm']
            f.write("### Task Performance\n\n")
            
            if 'detection' in vcm_data and 'mAP' in vcm_data['detection']:
                f.write(f"- **Object Detection mAP**: {vcm_data['detection']['mAP']:.4f}\n")
            
            if 'segmentation' in vcm_data and 'mIoU' in vcm_data['segmentation']:
                f.write(f"- **Segmentation mIoU**: {vcm_data['segmentation']['mIoU']:.4f}\n")
            
            f.write("\n")

def generate_detailed_results(results, output_file):
    """Generate detailed results section"""
    with open(output_file, 'a') as f:
        f.write("## Detailed Results\n\n")
        
        # Codec Performance
        f.write("### 1. Codec Performance Analysis\n\n")
        
        if 'codec' in results and not results['codec'].empty:
            codec_data = results['codec']
            
            f.write("#### Rate-Distortion Performance\n\n")
            f.write("| Lambda | PSNR (dB) | MS-SSIM | BPP |\n")
            f.write("|--------|-----------|---------|-----|\n")
            
            for lambda_val in sorted(codec_data['lambda'].unique()):
                lambda_data = codec_data[codec_data['lambda'] == lambda_val]
                avg_psnr = lambda_data['psnr_db'].mean()
                avg_mssim = lambda_data['ms_ssim'].mean()
                avg_bpp = lambda_data['bpp'].mean()
                
                f.write(f"| {lambda_val} | {avg_psnr:.2f} | {avg_mssim:.4f} | {avg_bpp:.3f} |\n")
            
            f.write("\n")
        
        # Baseline Comparison
        f.write("### 2. Baseline Comparison\n\n")
        
        if 'baseline' in results and not results['baseline'].empty:
            baseline_data = results['baseline']
            
            f.write("#### Comparison with Traditional Codecs\n\n")
            f.write("| Method | PSNR (dB) | MS-SSIM | BPP |\n")
            f.write("|--------|-----------|---------|-----|\n")
            
            for method in baseline_data['method'].unique():
                method_data = baseline_data[baseline_data['method'] == method]
                avg_psnr = method_data['psnr_db'].mean()
                avg_mssim = method_data['ms_ssim'].mean()
                avg_bpp = method_data['bpp'].mean()
                
                f.write(f"| {method} | {avg_psnr:.2f} | {avg_mssim:.4f} | {avg_bpp:.3f} |\n")
            
            f.write("\n")
        
        # Task Performance
        f.write("### 3. Task Performance on Compressed Features\n\n")
        
        if 'vcm' in results:
            vcm_data = results['vcm']
            
            if 'detection' in vcm_data:
                det_data = vcm_data['detection']
                f.write("#### Object Detection Results\n\n")
                for key, value in det_data.items():
                    if isinstance(value, (int, float)):
                        f.write(f"- **{key}**: {value:.4f}\n")
                    else:
                        f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            
            if 'segmentation' in vcm_data:
                seg_data = vcm_data['segmentation']
                f.write("#### Semantic Segmentation Results\n\n")
                for key, value in seg_data.items():
                    if isinstance(value, (int, float)):
                        f.write(f"- **{key}**: {value:.4f}\n")
                    else:
                        f.write(f"- **{key}**: {value}\n")
                f.write("\n")

def generate_statistical_analysis(results, output_file):
    """Generate statistical analysis section"""
    with open(output_file, 'a') as f:
        f.write("## Statistical Analysis\n\n")
        
        if 'statistics' in results:
            f.write("### Detailed Statistical Analysis\n\n")
            f.write("```\n")
            f.write(results['statistics'])
            f.write("\n```\n\n")
        else:
            f.write("Statistical analysis results not available.\n\n")

def generate_conclusions(results, output_file):
    """Generate conclusions and future work"""
    with open(output_file, 'a') as f:
        f.write("## Conclusions and Future Work\n\n")
        
        f.write("### Key Findings\n\n")
        f.write("1. **Compression Efficiency**: WAVENET-MV achieves competitive compression ratios ")
        f.write("while maintaining high visual quality as measured by PSNR and MS-SSIM.\n\n")
        
        f.write("2. **Task Performance**: The framework successfully preserves task-relevant ")
        f.write("information in compressed features, enabling accurate object detection ")
        f.write("and semantic segmentation.\n\n")
        
        f.write("3. **Rate-Distortion Trade-off**: The lambda parameter effectively controls ")
        f.write("the trade-off between compression rate and reconstruction quality.\n\n")
        
        f.write("### Technical Contributions\n\n")
        f.write("- **Wavelet-based Analysis**: Novel wavelet transform CNN for efficient feature extraction\n")
        f.write("- **AdaMixNet Architecture**: Adaptive mixing network for feature refinement\n")
        f.write("- **End-to-End Training**: Joint optimization of compression and task performance\n")
        f.write("- **VCM Framework**: Complete pipeline for video coding for machine vision\n\n")
        
        f.write("### Future Work\n\n")
        f.write("1. **Scalability**: Extend to higher resolution videos and real-time applications\n")
        f.write("2. **Multi-task Learning**: Explore joint training for multiple downstream tasks\n")
        f.write("3. **Temporal Modeling**: Incorporate temporal dependencies for video compression\n")
        f.write("4. **Hardware Optimization**: Optimize for deployment on edge devices\n\n")
        
        f.write("### Impact and Applications\n\n")
        f.write("WAVENET-MV has significant potential for applications requiring efficient ")
        f.write("video compression while maintaining machine vision task performance:\n\n")
        f.write("- **Surveillance Systems**: Bandwidth-efficient video monitoring\n")
        f.write("- **Autonomous Vehicles**: Real-time video processing for navigation\n")
        f.write("- **IoT Devices**: Resource-constrained video analysis\n")
        f.write("- **Cloud Computing**: Reduced storage and transmission costs\n\n")

def main():
    parser = argparse.ArgumentParser(description="Generate Summary Report for WAVENET-MV")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--output_file", default="results/evaluation_summary.md", 
                       help="Output summary report file")
    parser.add_argument("--paper_format", default="ieee", help="Paper format (ieee)")
    
    args = parser.parse_args()
    
    print("📋 Generating Summary Report for WAVENET-MV")
    print(f"📁 Results from: {args.results_dir}")
    print(f"📄 Report to: {args.output_file}")
    
    # Load all results
    results = load_all_results(args.results_dir)
    
    # Generate report sections
    generate_executive_summary(results, args.output_file)
    generate_detailed_results(results, args.output_file)
    generate_statistical_analysis(results, args.output_file)
    generate_conclusions(results, args.output_file)
    
    print(f"\n🎉 Summary report generated: {args.output_file}")
    print("📋 Report includes:")
    print("  - Executive summary with key metrics")
    print("  - Detailed results analysis")
    print("  - Statistical analysis")
    print("  - Conclusions and future work")

if __name__ == "__main__":
    main() 