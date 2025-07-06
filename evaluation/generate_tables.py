#!/usr/bin/env python3
"""
Generate LaTeX Tables cho WAVENET-MV IEEE Paper
Tạo tables cho codec metrics, task performance, và baseline comparison
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

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

def generate_codec_table(results, output_file):
    """Generate LaTeX table for codec metrics"""
    print("📊 Generating codec metrics table...")
    
    # Load results - handle both dict and DataFrame
    if isinstance(results, dict):
        if 'codec' in results and results['codec'] is not None:
            results_df = results['codec']
        else:
            print("⚠️ No codec data found in results")
            return
    elif isinstance(results, str):
        results_df = pd.read_csv(results)
    else:
        results_df = results
    
    # Check available columns
    print(f"Available columns: {list(results_df.columns)}")
    
    # Use correct column names
    psnr_col = 'psnr_db' if 'psnr_db' in results_df.columns else 'psnr'
    bpp_col = 'bpp' if 'bpp' in results_df.columns else 'bits_per_pixel'
    ms_ssim_col = 'ms_ssim' if 'ms_ssim' in results_df.columns else 'ms_ssim_db'
    
    # Calculate averages for each lambda
    table_data = []
    for lambda_val in sorted(results_df['lambda'].unique()):
        lambda_data = results_df[results_df['lambda'] == lambda_val]
        
        avg_psnr = lambda_data[psnr_col].mean()
        avg_bpp = lambda_data[bpp_col].mean()
        avg_ms_ssim = lambda_data[ms_ssim_col].mean()
        
        table_data.append({
            'lambda': lambda_val,
            'psnr': f"{avg_psnr:.2f}",
            'bpp': f"{avg_bpp:.2f}",
            'ms_ssim': f"{avg_ms_ssim:.3f}"
        })
    
    # Generate LaTeX table
    latex_table = r"""
\begin{table}[t]
\centering
\caption{Codec Performance Metrics}
\label{tab:codec_metrics}
\begin{tabular}{cccc}
\toprule
$\lambda$ & PSNR (dB) & BPP & MS-SSIM \\
\midrule
"""
    
    for row in table_data:
        latex_table += f"{row['lambda']} & {row['psnr']} & {row['bpp']} & {row['ms_ssim']} \\\\\n"
    
    latex_table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    # Save table
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"✓ Saved codec table to {output_file}")
    return output_file

def generate_baseline_table(results, output_file):
    """Generate baseline comparison table"""
    if 'baseline' not in results:
        print("⚠️ Missing baseline data for table")
        return
    
    baseline_data = results['baseline']
    
    # Create LaTeX table
    table_content = []
    table_content.append("\\begin{table}[t]")
    table_content.append("\\centering")
    table_content.append("\\caption{Baseline Methods Comparison}")
    table_content.append("\\label{tab:baseline_comparison}")
    table_content.append("\\begin{tabular}{lccc}")
    table_content.append("\\hline")
    table_content.append("Method & PSNR (dB) & MS-SSIM & BPP \\\\")
    table_content.append("\\hline")
    
    if not baseline_data.empty:
        for method in baseline_data['method'].unique():
            method_data = baseline_data[baseline_data['method'] == method]
            avg_psnr = method_data['psnr'].mean()
            avg_ms_ssim = method_data['ms_ssim'].mean()
            avg_bpp = method_data['bpp'].mean()
            
            table_content.append(f"{method} & {avg_psnr:.2f} & {avg_ms_ssim:.4f} & {avg_bpp:.3f} \\\\")
    
    table_content.append("\\hline")
    table_content.append("\\end{tabular}")
    table_content.append("\\end{table}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(table_content))
    
    print(f"✅ Generated baseline table: {output_file}")

def generate_task_table(results, output_file):
    """Generate task performance table"""
    if 'vcm' not in results:
        print("⚠️ Missing VCM data for task table")
        return
    
    vcm_data = results['vcm']
    
    # Create LaTeX table
    table_content = []
    table_content.append("\\begin{table}[t]")
    table_content.append("\\centering")
    table_content.append("\\caption{Task Performance on Compressed Features}")
    table_content.append("\\label{tab:task_performance}")
    table_content.append("\\begin{tabular}{lc}")
    table_content.append("\\hline")
    table_content.append("Task & Metric \\\\")
    table_content.append("\\hline")
    
    # Detection performance
    if 'detection' in vcm_data:
        det_data = vcm_data['detection']
        if 'mAP' in det_data:
            table_content.append(f"Object Detection & mAP: {det_data['mAP']:.4f} \\\\")
    
    # Segmentation performance
    if 'segmentation' in vcm_data:
        seg_data = vcm_data['segmentation']
        if 'mIoU' in seg_data:
            table_content.append(f"Semantic Segmentation & mIoU: {seg_data['mIoU']:.4f} \\\\")
    
    table_content.append("\\hline")
    table_content.append("\\end{tabular}")
    table_content.append("\\end{table}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(table_content))
    
    print(f"✅ Generated task table: {output_file}")

def generate_ablation_table(results, output_file):
    """Generate ablation study table"""
    if 'codec' not in results:
        print("⚠️ Missing codec data for ablation table")
        return
    
    codec_data = results['codec']
    
    # Create LaTeX table
    table_content = []
    table_content.append("\\begin{table}[t]")
    table_content.append("\\centering")
    table_content.append("\\caption{Lambda Ablation Study}")
    table_content.append("\\label{tab:lambda_ablation}")
    table_content.append("\\begin{tabular}{lccc}")
    table_content.append("\\hline")
    table_content.append("Lambda (λ) & PSNR (dB) & MS-SSIM & BPP \\\\")
    table_content.append("\\hline")
    
    if not codec_data.empty:
        for lambda_val in sorted(codec_data['lambda'].unique()):
            lambda_data = codec_data[codec_data['lambda'] == lambda_val]
            avg_psnr = lambda_data['psnr'].mean()
            avg_ms_ssim = lambda_data['ms_ssim'].mean()
            avg_bpp = lambda_data['bpp'].mean()
            
            table_content.append(f"{lambda_val} & {avg_psnr:.2f} & {avg_ms_ssim:.4f} & {avg_bpp:.3f} \\\\")
    
    table_content.append("\\hline")
    table_content.append("\\end{tabular}")
    table_content.append("\\end{table}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(table_content))
    
    print(f"✅ Generated ablation table: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX Tables for WAVENET-MV")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--output_file", default="tables/paper_tables.tex", help="Output LaTeX file")
    parser.add_argument("--format", default="ieee", help="Paper format (ieee)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print("📋 Generating LaTeX Tables for WAVENET-MV")
    print(f"📁 Results from: {args.results_dir}")
    print(f"📄 Tables to: {args.output_file}")
    
    # Load results
    results = load_results(args.results_dir)
    
    # Generate all tables
    all_tables = []
    
    # Codec performance table
    codec_table = []
    generate_codec_table(results, "tables/codec_table.tex")
    with open("tables/codec_table.tex", 'r') as f:
        all_tables.append(f.read())
    
    # Baseline comparison table
    baseline_table = []
    generate_baseline_table(results, "tables/baseline_table.tex")
    with open("tables/baseline_table.tex", 'r') as f:
        all_tables.append(f.read())
    
    # Task performance table
    task_table = []
    generate_task_table(results, "tables/task_table.tex")
    with open("tables/task_table.tex", 'r') as f:
        all_tables.append(f.read())
    
    # Ablation study table
    ablation_table = []
    generate_ablation_table(results, "tables/ablation_table.tex")
    with open("tables/ablation_table.tex", 'r') as f:
        all_tables.append(f.read())
    
    # Combine all tables
    with open(args.output_file, 'w') as f:
        f.write("% WAVENET-MV Paper Tables\n")
        f.write("% Generated automatically\n\n")
        f.write('\n\n'.join(all_tables))
    
    print(f"\n🎉 All tables generated in: {args.output_file}")
    print("📋 Individual tables:")
    print("  - tables/codec_table.tex")
    print("  - tables/baseline_table.tex")
    print("  - tables/task_table.tex")
    print("  - tables/ablation_table.tex")

if __name__ == "__main__":
    main() 