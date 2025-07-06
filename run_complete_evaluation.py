#!/usr/bin/env python3
"""
Complete Evaluation Pipeline cho WAVENET-MV Paper
Chạy tất cả evaluation và generate figures/tables
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run command với error handling"""
    print(f"\n🚀 {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Complete WAVENET-MV Evaluation Pipeline")
    parser.add_argument("--stage1_checkpoint", required=True, help="Stage 1 checkpoint path")
    parser.add_argument("--stage2_checkpoint", required=True, help="Stage 2 checkpoint path") 
    parser.add_argument("--stage3_checkpoint", required=True, help="Stage 3 checkpoint path")
    parser.add_argument("--dataset", default="coco", help="Dataset name")
    parser.add_argument("--data_dir", default="datasets/COCO", help="Dataset directory")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples for evaluation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("fig", exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    
    print("🎯 WAVENET-MV Complete Evaluation Pipeline")
    print("=" * 50)
    
    # 1. VCM Evaluation (AI Tasks)
    print("\n📊 Step 1: VCM Evaluation")
    vcm_cmd = f"""python evaluate_vcm.py \
        --stage1_checkpoint {args.stage1_checkpoint} \
        --stage2_checkpoint {args.stage2_checkpoint} \
        --stage3_checkpoint {args.stage3_checkpoint} \
        --dataset {args.dataset} \
        --data_dir {args.data_dir} \
        --enable_detection \
        --enable_segmentation \
        --batch_size {args.batch_size} \
        --max_samples {args.max_samples} \
        --output_dir {args.output_dir}"""
    
    if not run_command(vcm_cmd, "VCM Evaluation"):
        print("⚠️ VCM evaluation failed, but continuing...")
    
    # 2. Codec Metrics Evaluation (PSNR, MS-SSIM, BPP)
    print("\n📊 Step 2: Codec Metrics Evaluation")
    codec_cmd = f"""python evaluation/codec_metrics_final.py \
        --stage1_checkpoint {args.stage1_checkpoint} \
        --stage2_checkpoint {args.stage2_checkpoint} \
        --dataset {args.dataset} \
        --data_dir {args.data_dir} \
        --split val \
        --batch_size {args.batch_size} \
        --max_samples {args.max_samples} \
        --lambdas 64 128 256 512 1024 \
        --output_dir {args.output_dir}"""
    
    if not run_command(codec_cmd, "Codec Metrics Evaluation"):
        print("⚠️ Codec metrics evaluation failed, but continuing...")
    
    # 3. Baseline Comparison
    print("\n📊 Step 3: Baseline Comparison")
    baseline_cmd = f"""python evaluation/compare_baselines.py \
        --dataset {args.dataset} \
        --data_dir {args.data_dir} \
        --split val \
        --max_samples {args.max_samples} \
        --methods JPEG WebP PNG \
        --qualities 10 30 50 70 90 \
        --output_dir {args.output_dir}"""
    
    if not run_command(baseline_cmd, "Baseline Comparison"):
        print("⚠️ Baseline comparison failed, but continuing...")
    
    # 4. Generate Paper Figures
    print("\n📊 Step 4: Generate Paper Figures")
    figures_cmd = f"""python evaluation/generate_paper_results.py \
        --results_dir {args.output_dir} \
        --output_dir fig \
        --paper_format ieee \
        --generate_rd_curves \
        --generate_task_curves \
        --generate_ablation_study"""
    
    if not run_command(figures_cmd, "Generate Paper Figures"):
        print("⚠️ Figure generation failed, but continuing...")
    
    # 5. Generate Tables
    print("\n📊 Step 5: Generate Tables")
    tables_cmd = f"""python evaluation/generate_tables.py \
        --results_dir {args.output_dir} \
        --output_file tables/paper_tables.tex \
        --format ieee"""
    
    if not run_command(tables_cmd, "Generate Tables"):
        print("⚠️ Table generation failed, but continuing...")
    
    # 6. Statistical Analysis
    print("\n📊 Step 6: Statistical Analysis")
    stats_cmd = f"""python evaluation/statistical_analysis.py \
        --results_file {args.output_dir}/vcm_evaluation_results.csv \
        --baseline_file {args.output_dir}/baseline_comparison_results.csv \
        --output_file {args.output_dir}/statistical_analysis.txt"""
    
    if not run_command(stats_cmd, "Statistical Analysis"):
        print("⚠️ Statistical analysis failed, but continuing...")
    
    # 7. Generate Summary Report
    print("\n📊 Step 7: Generate Summary Report")
    summary_cmd = f"""python evaluation/generate_summary_report.py \
        --results_dir {args.output_dir} \
        --output_file {args.output_dir}/evaluation_summary.md \
        --paper_format ieee"""
    
    if not run_command(summary_cmd, "Generate Summary Report"):
        print("⚠️ Summary report generation failed")
    
    print("\n" + "=" * 50)
    print("🎉 Complete Evaluation Pipeline Finished!")
    print(f"📁 Results saved in: {args.output_dir}")
    print(f"📊 Figures saved in: fig/")
    print(f"📋 Tables saved in: tables/")
    print("\n📝 Next steps for paper:")
    print("1. Review generated figures in fig/")
    print("2. Check tables in tables/paper_tables.tex")
    print("3. Read evaluation summary in results/evaluation_summary.md")
    print("4. Include statistical analysis in results/statistical_analysis.txt")

if __name__ == "__main__":
    main() 