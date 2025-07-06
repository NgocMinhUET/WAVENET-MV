#!/usr/bin/env python3
"""
Train Stage 3 Complete: AI Heads cho cả Detection và Segmentation
- Train YOLO-tiny head cho object detection
- Train SegFormer-lite head cho semantic segmentation
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
    parser = argparse.ArgumentParser(description='Train Stage 3 Complete: AI Heads for VCM')
    
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
    parser.add_argument('--experiment_name', type=str, default='stage3_complete',
                       help='Experiment name for logging')
    
    args = parser.parse_args()
    
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
        '--num_workers', '4',
        '--enable_detection',  # Enable both tasks
        '--enable_segmentation'  # Enable both tasks
    ]
    
    # Replace sys.argv temporarily
    sys.argv = stage3_argv
    
    print("🚀 Starting Stage 3 Complete Training for VCM...")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🎯 Tasks: Detection=True, Segmentation=True")
    print(f"📁 Output: {args.output_dir}")
    print(f"⏱️ Epochs: {args.epochs}")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"📚 Learning Rate: {args.lr}")
    
    try:
        # Run Stage 3 training
        stage3_main()
        print("✅ Stage 3 Complete Training finished successfully!")
    except Exception as e:
        print(f"❌ Stage 3 Complete Training failed: {e}")
        raise
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == '__main__':
    main() 