#!/usr/bin/env python3
"""
Evaluate Video Coding for Machine (VCM) Performance
- Object Detection trên compressed features
- Semantic Segmentation trên compressed features
- So sánh với pixel-domain baselines
"""

import os
import sys
import argparse
from pathlib import Path

# Fix OpenMP warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import VCM metrics directly
from evaluation.vcm_metrics import VCMEvaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate VCM Performance')
    
    # Model checkpoints
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                       help='Path to Stage 1 checkpoint (wavelet)')
    parser.add_argument('--stage2_checkpoint', type=str, required=True,
                       help='Path to Stage 2 checkpoint (compressor)')
    parser.add_argument('--stage3_checkpoint', type=str, default=None,
                       help='Path to Stage 3 checkpoint (AI heads)')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='coco',
                       choices=['coco', 'davis'], help='Dataset to evaluate')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='val',
                       help='Dataset split to evaluate')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    
    # Evaluation settings
    parser.add_argument('--enable_detection', action='store_true',
                       help='Enable object detection evaluation')
    parser.add_argument('--enable_segmentation', action='store_true',
                       help='Enable semantic segmentation evaluation')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--lambda_rd', type=int, default=128,
                       help='Rate-distortion lambda')
    
    # Output
    parser.add_argument('--output_json', type=str, default='results/vcm_results.json',
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.enable_detection and not args.enable_segmentation:
        print("⚠️ Warning: No evaluation tasks enabled!")
        print("Use --enable_detection and/or --enable_segmentation")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    
    print("🚀 Starting VCM Evaluation...")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🎯 Tasks: Detection={args.enable_detection}, Segmentation={args.enable_segmentation}")
    print(f"📁 Output: {args.output_json}")
    
    # Run VCM evaluation
    try:
        evaluator = VCMEvaluator(args)
        evaluator.evaluate_all()
        evaluator.save_results()
        print("✅ VCM evaluation completed successfully")
    except Exception as e:
        print(f"❌ VCM evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main() 