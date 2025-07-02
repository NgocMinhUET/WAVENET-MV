"""
Generate Paper Results Script
Tạo tất cả kết quả cần thiết cho bài báo WAVENET-MV:
1. WAVENET-MV evaluation trên multiple lambda values
2. Baseline comparison (JPEG, WebP, PNG)
3. Rate-Distortion curves
4. Performance tables
5. Visualization figures
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_command(cmd, description):
    """Run command with error handling"""
    print(f"\n🔄 {description}...")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def generate_wavenet_results(args):
    """Generate WAVENET-MV evaluation results"""
    print("\n" + "="*50)
    print("STEP 1: WAVENET-MV EVALUATION")
    print("="*50)
    
    # Multiple lambda evaluation
    lambdas = [64, 128, 256, 512, 1024]
    
    cmd = f"""python evaluation/codec_metrics.py \
        --checkpoint {args.checkpoint} \
        --dataset {args.dataset} \
        --data_dir {args.data_dir} \
        --split {args.split} \
        --lambdas {' '.join(map(str, lambdas))} \
        --batch_size {args.batch_size} \
        --max_samples {args.max_samples} \
        --output_csv results/wavenet_mv_full_evaluation.csv \
        --skip_entropy_update"""
    
    return run_command(cmd, "WAVENET-MV Full Evaluation")

def generate_baseline_comparison(args):
    """Generate baseline comparison"""
    print("\n" + "="*50)
    print("STEP 2: BASELINE COMPARISON")
    print("="*50)
    
    cmd = f"""python evaluation/compare_baselines.py \
        --dataset {args.dataset} \
        --data_dir {args.data_dir} \
        --split {args.split} \
        --image_size {args.image_size} \
        --max_samples {args.max_samples} \
        --wavenet_checkpoint {args.checkpoint} \
        --lambdas 256 512 1024 \
        --batch_size {args.batch_size} \
        --output_csv results/baseline_comparison_full.csv"""
    
    return run_command(cmd, "Baseline Comparison")

def create_paper_tables(args):
    """Create formatted tables for paper"""
    print("\n" + "="*50)
    print("STEP 3: CREATING PAPER TABLES")
    print("="*50)
    
    # Load results
    try:
        wavenet_df = pd.read_csv('results/wavenet_mv_full_evaluation.csv')
        baseline_df = pd.read_csv('results/baseline_comparison_full.csv')
        
        print("✅ Results loaded successfully")
        
        # Create comparison table
        comparison_data = []
        
        # Add WAVENET-MV results
        for _, row in wavenet_df.iterrows():
            comparison_data.append({
                'Method': f'WAVENET-MV (λ={row["lambda"]})',
                'PSNR (dB)': f'{row["psnr_db"]:.2f}',
                'MS-SSIM': f'{row["ms_ssim"]:.4f}',
                'BPP': f'{row["bpp"]:.3f}'
            })
        
        # Add baseline results
        for _, row in baseline_df.iterrows():
            if row['method'] == 'JPEG':
                method_name = f'JPEG (Q={row["quality"]})'
            elif row['method'] == 'WebP':
                method_name = f'WebP (Q={row["quality"]})'
            else:
                method_name = row['method']
            
            comparison_data.append({
                'Method': method_name,
                'PSNR (dB)': f'{row["psnr_db"]:.2f}',
                'MS-SSIM': f'{row["ms_ssim"]:.4f}',
                'BPP': f'{row["bpp"]:.3f}'
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save formatted table
        comparison_df.to_csv('results/paper_comparison_table.csv', index=False)
        
        # Create LaTeX table
        latex_table = comparison_df.to_latex(index=False, float_format="%.3f")
        with open('results/paper_comparison_table.tex', 'w') as f:
            f.write(latex_table)
        
        print("✅ Paper tables created:")
        print("  - results/paper_comparison_table.csv")
        print("  - results/paper_comparison_table.tex")
        
        # Print summary
        print("\n📊 PERFORMANCE SUMMARY:")
        print(comparison_df.to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create paper tables: {e}")
        return False

def create_paper_figures(args):
    """Create figures for paper"""
    print("\n" + "="*50)
    print("STEP 4: CREATING PAPER FIGURES")
    print("="*50)
    
    try:
        # Load data
        wavenet_df = pd.read_csv('results/wavenet_mv_full_evaluation.csv')
        baseline_df = pd.read_csv('results/baseline_comparison_full.csv')
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create Rate-Distortion curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # PSNR vs BPP
        # WAVENET-MV
        wavenet_sorted = wavenet_df.sort_values('bpp')
        ax1.plot(wavenet_sorted['bpp'], wavenet_sorted['psnr_db'], 
                'o-', label='WAVENET-MV', linewidth=3, markersize=8, color='red')
        
        # Baselines
        methods = baseline_df['method'].unique()
        colors = ['blue', 'green', 'orange', 'purple']
        
        for i, method in enumerate(methods):
            method_data = baseline_df[baseline_df['method'] == method].sort_values('bpp')
            ax1.plot(method_data['bpp'], method_data['psnr_db'], 
                    'o-', label=method, linewidth=2, markersize=6, color=colors[i % len(colors)])
        
        ax1.set_xlabel('Rate (BPP)', fontsize=14)
        ax1.set_ylabel('PSNR (dB)', fontsize=14)
        ax1.set_title('Rate-Distortion: PSNR vs BPP', fontsize=16)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # MS-SSIM vs BPP
        ax2.plot(wavenet_sorted['bpp'], wavenet_sorted['ms_ssim'], 
                'o-', label='WAVENET-MV', linewidth=3, markersize=8, color='red')
        
        for i, method in enumerate(methods):
            method_data = baseline_df[baseline_df['method'] == method].sort_values('bpp')
            ax2.plot(method_data['bpp'], method_data['ms_ssim'], 
                    'o-', label=method, linewidth=2, markersize=6, color=colors[i % len(colors)])
        
        ax2.set_xlabel('Rate (BPP)', fontsize=14)
        ax2.set_ylabel('MS-SSIM', fontsize=14)
        ax2.set_title('Rate-Distortion: MS-SSIM vs BPP', fontsize=16)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/paper_rd_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig('results/paper_rd_curves.pdf', bbox_inches='tight')
        
        print("✅ R-D curves saved:")
        print("  - results/paper_rd_curves.png")
        print("  - results/paper_rd_curves.pdf")
        
        # Create performance bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Select representative points for comparison
        comparison_points = []
        
        # WAVENET-MV at λ=256
        wv_256 = wavenet_df[wavenet_df['lambda'] == 256].iloc[0]
        comparison_points.append({
            'Method': 'WAVENET-MV\n(λ=256)',
            'PSNR': wv_256['psnr_db'],
            'BPP': wv_256['bpp']
        })
        
        # JPEG at similar BPP
        jpeg_data = baseline_df[baseline_df['method'] == 'JPEG']
        if not jpeg_data.empty:
            # Find JPEG quality closest to WAVENET-MV BPP
            closest_jpeg = jpeg_data.iloc[(jpeg_data['bpp'] - wv_256['bpp']).abs().argsort()[:1]]
            comparison_points.append({
                'Method': f'JPEG\n(Q={closest_jpeg.iloc[0]["quality"]})',
                'PSNR': closest_jpeg.iloc[0]['psnr_db'],
                'BPP': closest_jpeg.iloc[0]['bpp']
            })
        
        # WebP at similar BPP
        webp_data = baseline_df[baseline_df['method'] == 'WebP']
        if not webp_data.empty:
            closest_webp = webp_data.iloc[(webp_data['bpp'] - wv_256['bpp']).abs().argsort()[:1]]
            comparison_points.append({
                'Method': f'WebP\n(Q={closest_webp.iloc[0]["quality"]})',
                'PSNR': closest_webp.iloc[0]['psnr_db'],
                'BPP': closest_webp.iloc[0]['bpp']
            })
        
        # Create bar chart
        comp_df = pd.DataFrame(comparison_points)
        x = np.arange(len(comp_df))
        
        bars = ax.bar(x, comp_df['PSNR'], color=['red', 'blue', 'green'], alpha=0.7)
        ax.set_xlabel('Method', fontsize=14)
        ax.set_ylabel('PSNR (dB)', fontsize=14)
        ax.set_title('PSNR Comparison at Similar Bit Rates', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(comp_df['Method'], fontsize=12)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.1f}dB\n({comp_df.iloc[i]["BPP"]:.2f} BPP)',
                   ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('results/paper_psnr_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('results/paper_psnr_comparison.pdf', bbox_inches='tight')
        
        print("✅ PSNR comparison chart saved:")
        print("  - results/paper_psnr_comparison.png")
        print("  - results/paper_psnr_comparison.pdf")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create paper figures: {e}")
        return False

def create_paper_summary(args):
    """Create paper summary"""
    print("\n" + "="*50)
    print("STEP 5: CREATING PAPER SUMMARY")
    print("="*50)
    
    try:
        # Load results
        wavenet_df = pd.read_csv('results/wavenet_mv_full_evaluation.csv')
        baseline_df = pd.read_csv('results/baseline_comparison_full.csv')
        
        # Calculate key statistics
        best_wavenet = wavenet_df.loc[wavenet_df['psnr_db'].idxmax()]
        best_jpeg = baseline_df[baseline_df['method'] == 'JPEG'].loc[
            baseline_df[baseline_df['method'] == 'JPEG']['psnr_db'].idxmax()
        ]
        
        summary = f"""
# WAVENET-MV Paper Results Summary

## Dataset
- Dataset: {args.dataset.upper()}
- Split: {args.split}
- Images: {args.max_samples if args.max_samples else 'All'}
- Image Size: {args.image_size}x{args.image_size}

## Best Performance
### WAVENET-MV (λ={best_wavenet['lambda']})
- PSNR: {best_wavenet['psnr_db']:.2f} dB
- MS-SSIM: {best_wavenet['ms_ssim']:.4f}
- BPP: {best_wavenet['bpp']:.3f}

### Best JPEG (Q={best_jpeg['quality']})
- PSNR: {best_jpeg['psnr_db']:.2f} dB
- MS-SSIM: {best_jpeg['ms_ssim']:.4f}
- BPP: {best_jpeg['bpp']:.3f}

## Performance Gain
- PSNR Improvement: {best_wavenet['psnr_db'] - best_jpeg['psnr_db']:.2f} dB
- MS-SSIM Improvement: {best_wavenet['ms_ssim'] - best_jpeg['ms_ssim']:.4f}

## Files Generated
1. results/wavenet_mv_full_evaluation.csv - WAVENET-MV detailed results
2. results/baseline_comparison_full.csv - Baseline comparison results
3. results/paper_comparison_table.csv - Formatted comparison table
4. results/paper_comparison_table.tex - LaTeX table for paper
5. results/paper_rd_curves.png/pdf - Rate-Distortion curves
6. results/paper_psnr_comparison.png/pdf - PSNR comparison chart

## Usage in Paper
- Use paper_rd_curves.pdf for Rate-Distortion analysis
- Use paper_comparison_table.tex for performance table
- Use paper_psnr_comparison.pdf for visual comparison
- Cite PSNR improvement of {best_wavenet['psnr_db'] - best_jpeg['psnr_db']:.2f} dB over JPEG
"""
        
        # Save summary
        with open('results/paper_summary.md', 'w') as f:
            f.write(summary)
        
        print("✅ Paper summary created: results/paper_summary.md")
        print(summary)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create paper summary: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate Paper Results')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='WAVENET-MV checkpoint path')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='coco',
                       help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='datasets/COCO',
                       help='Dataset directory')
    parser.add_argument('--split', type=str, default='val',
                       help='Dataset split')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='Maximum samples for evaluation')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    
    # Control arguments
    parser.add_argument('--skip_wavenet', action='store_true',
                       help='Skip WAVENET-MV evaluation')
    parser.add_argument('--skip_baseline', action='store_true',
                       help='Skip baseline comparison')
    parser.add_argument('--skip_figures', action='store_true',
                       help='Skip figure generation')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    print("🚀 GENERATING PAPER RESULTS FOR WAVENET-MV")
    print("="*60)
    
    success_count = 0
    total_steps = 5
    
    # Step 1: WAVENET-MV evaluation
    if not args.skip_wavenet:
        if generate_wavenet_results(args):
            success_count += 1
    else:
        print("⏭️ Skipping WAVENET-MV evaluation")
        success_count += 1
    
    # Step 2: Baseline comparison
    if not args.skip_baseline:
        if generate_baseline_comparison(args):
            success_count += 1
    else:
        print("⏭️ Skipping baseline comparison")
        success_count += 1
    
    # Step 3: Create tables
    if create_paper_tables(args):
        success_count += 1
    
    # Step 4: Create figures
    if not args.skip_figures:
        if create_paper_figures(args):
            success_count += 1
    else:
        print("⏭️ Skipping figure generation")
        success_count += 1
    
    # Step 5: Create summary
    if create_paper_summary(args):
        success_count += 1
    
    # Final summary
    print("\n" + "="*60)
    print(f"PAPER RESULTS GENERATION COMPLETE: {success_count}/{total_steps} steps successful")
    print("="*60)
    
    if success_count == total_steps:
        print("🎉 All results generated successfully!")
        print("📁 Check the 'results/' directory for all output files")
        print("📊 Ready for paper writing!")
    else:
        print(f"⚠️ {total_steps - success_count} steps failed - check logs above")


if __name__ == '__main__':
    main() 