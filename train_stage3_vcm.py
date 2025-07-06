#!/usr/bin/env python3
"""
Train Stage 3: AI Heads for Video Coding for Machine (VCM)
- Train YOLO-tiny head trên compressed features
- Train SegFormer-lite head trên compressed features
- Frozen compression pipeline (Stage 1 + Stage 2)
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from training.stage3_train_ai import main as stage3_main


def main():
    parser = argparse.ArgumentParser(description='Train Stage 3: AI Heads for VCM')
    
    # Model checkpoints
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                       help='Path to Stage 1 checkpoint (wavelet)')
    parser.add_argument('--stage2_checkpoint', type=str, required=True,
                       help='Path to Stage 2 checkpoint (compressor)')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='coco',
                       choices=['coco', 'davis'], help='Dataset to use')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    
    # Training settings
    parser.add_argument('--enable_detection', action='store_true',
                       help='Enable YOLO detection training')
    parser.add_argument('--enable_segmentation', action='store_true',
                       help='Enable SegFormer segmentation training')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--lambda_rd', type=int, default=128,
                       help='Rate-distortion lambda')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--experiment_name', type=str, default='stage3_vcm',
                       help='Experiment name for logging')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.enable_detection and not args.enable_segmentation:
        print("⚠️ Warning: No training tasks enabled!")
        print("Use --enable_detection and/or --enable_segmentation")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set environment variables for stage3_main
    import sys
    
    # Modify sys.argv to pass arguments to stage3_main
    original_argv = sys.argv.copy()
    
    # Build command line arguments for stage3_main
    stage3_argv = [
        'stage3_train_ai.py',
        '--stage1_checkpoint', args.stage1_checkpoint,
        '--stage2_checkpoint', args.stage2_checkpoint,
        '--dataset', args.dataset,
        '--data_dir', args.data_dir,
        '--image_size', str(args.image_size),
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--learning_rate', str(args.lr),
        '--lambda_rd', str(args.lambda_rd),
        '--num_workers', '4'
    ]
    
    # Add task flags
    if args.enable_detection:
        stage3_argv.append('--enable_detection')
    if args.enable_segmentation:
        stage3_argv.append('--enable_segmentation')
    
    # Replace sys.argv temporarily
    sys.argv = stage3_argv
    
    print("🚀 Starting Stage 3 Training for VCM...")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🎯 Tasks: Detection={args.enable_detection}, Segmentation={args.enable_segmentation}")
    print(f"📁 Output: {args.output_dir}")
    
    try:
        # Run Stage 3 training
        stage3_main()
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == '__main__':
    main() 