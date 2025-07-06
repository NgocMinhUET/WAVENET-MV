#!/usr/bin/env python3
"""
Generate Paper Figures cho WAVENET-MV IEEE Paper
Tạo rate-distortion curves, task performance curves, và ablation study plots
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style cho IEEE paper
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(results_dir):
    """Load tất cả results từ CSV files"""
    results = {}
    
    # Load VCM results
    vcm_file = os.path.join(results_dir, "vcm_results.json")
    if os.path.exists(vcm_file):
        try:
            import json
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
    
    return results

def generate_rd_curves(results, output_dir):
    """Generate Rate-Distortion curves"""
    if 'codec' not in results or 'baseline' not in results:
        print("⚠️ Missing data for RD curves")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # WAVENET-MV RD curves
    codec_data = results['codec']
    if not codec_data.empty:
        for lambda_val in sorted(codec_data['lambda'].unique()):
            lambda_data = codec_data[codec_data['lambda'] == lambda_val]
            ax1.plot(lambda_data['bpp'], lambda_data['psnr'], 
                    marker='o', label=f'WAVENET-MV (λ={lambda_val})', linewidth=2)
            ax2.plot(lambda_data['bpp'], lambda_data['ms_ssim'], 
                    marker='o', label=f'WAVENET-MV (λ={lambda_val})', linewidth=2)
    
    # Baseline methods
    baseline_data = results['baseline']
    if not baseline_data.empty:
        for method in baseline_data['method'].unique():
            method_data = baseline_data[baseline_data['method'] == method]
            ax1.plot(method_data['bpp'], method_data['psnr'], 
                    marker='s', linestyle='--', label=f'{method}', linewidth=2)
            ax2.plot(method_data['bpp'], method_data['ms_ssim'], 
                    marker='s', linestyle='--', label=f'{method}', linewidth=2)
    
    # Formatting
    ax1.set_xlabel('Bits per Pixel (BPP)')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Rate-Distortion: PSNR vs BPP')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Bits per Pixel (BPP)')
    ax2.set_ylabel('MS-SSIM')
    ax2.set_title('Rate-Distortion: MS-SSIM vs BPP')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rate_distortion_comparison.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'rate_distortion_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Generated Rate-Distortion curves")

def generate_task_curves(results, output_dir):
    """Generate Task Performance curves"""
    if 'vcm' not in results:
        print("⚠️ Missing VCM data for task curves")
        return
    
    vcm_data = results['vcm']
    
    # Create task performance plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Detection performance
    if 'detection' in vcm_data:
        det_data = vcm_data['detection']
        if 'mAP' in det_data:
            ax1.bar(['WAVENET-MV'], [det_data['mAP']], 
                   color='skyblue', alpha=0.7, label='Detection mAP')
            ax1.set_ylabel('mAP')
            ax1.set_title('Object Detection Performance')
            ax1.grid(True, alpha=0.3)
    
    # Segmentation performance
    if 'segmentation' in vcm_data:
        seg_data = vcm_data['segmentation']
        if 'mIoU' in seg_data:
            ax2.bar(['WAVENET-MV'], [seg_data['mIoU']], 
                   color='lightgreen', alpha=0.7, label='Segmentation mIoU')
            ax2.set_ylabel('mIoU')
            ax2.set_title('Semantic Segmentation Performance')
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task_performance.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'task_performance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Generated Task Performance curves")

def generate_ablation_study(results, output_dir):
    """Generate Ablation Study plots"""
    if 'codec' not in results:
        print("⚠️ Missing codec data for ablation study")
        return
    
    codec_data = results['codec']
    
    # Lambda ablation study
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if not codec_data.empty:
        lambda_vals = sorted(codec_data['lambda'].unique())
        avg_psnr = [codec_data[codec_data['lambda'] == l]['psnr'].mean() for l in lambda_vals]
        avg_bpp = [codec_data[codec_data['lambda'] == l]['bpp'].mean() for l in lambda_vals]
        
        ax1.plot(lambda_vals, avg_psnr, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Lambda (λ)')
        ax1.set_ylabel('Average PSNR (dB)')
        ax1.set_title('PSNR vs Lambda')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(lambda_vals, avg_bpp, marker='s', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Lambda (λ)')
        ax2.set_ylabel('Average BPP')
        ax2.set_title('BPP vs Lambda')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_study.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'ablation_study.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Generated Ablation Study plots")

def main():
    parser = argparse.ArgumentParser(description="Generate Paper Figures for WAVENET-MV")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--output_dir", default="fig", help="Output directory for figures")
    parser.add_argument("--paper_format", default="ieee", help="Paper format (ieee)")
    parser.add_argument("--generate_rd_curves", action="store_true", help="Generate RD curves")
    parser.add_argument("--generate_task_curves", action="store_true", help="Generate task curves")
    parser.add_argument("--generate_ablation_study", action="store_true", help="Generate ablation study")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🎨 Generating Paper Figures for WAVENET-MV")
    print(f"📁 Results from: {args.results_dir}")
    print(f"📊 Figures to: {args.output_dir}")
    
    # Load results
    results = load_results(args.results_dir)
    
    # Generate figures
    if args.generate_rd_curves:
        generate_rd_curves(results, args.output_dir)
    
    if args.generate_task_curves:
        generate_task_curves(results, args.output_dir)
    
    if args.generate_ablation_study:
        generate_ablation_study(results, args.output_dir)
    
    # Generate all if no specific option given
    if not any([args.generate_rd_curves, args.generate_task_curves, args.generate_ablation_study]):
        generate_rd_curves(results, args.output_dir)
        generate_task_curves(results, args.output_dir)
        generate_ablation_study(results, args.output_dir)
    
    print(f"\n🎉 Paper figures generated in: {args.output_dir}")
    print("📋 Available figures:")
    for fig_file in os.listdir(args.output_dir):
        if fig_file.endswith(('.pdf', '.png')):
            print(f"  - {fig_file}")

if __name__ == "__main__":
    main() 