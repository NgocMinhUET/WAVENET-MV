#!/bin/bash

# Comprehensive Evaluation Script for WAVENET-MV
# So sánh với JPEG, JPEG 2000, neural codec và ablation study

echo "🚀 Starting WAVENET-MV Comprehensive Evaluation"
echo "=================================================="

# Set paths
DATA_DIR="datasets/COCO"
STAGE1_CHECKPOINT="checkpoints/stage1_wavelet_coco_best.pth"
STAGE2_CHECKPOINT="checkpoints/stage2_compressor_coco_best.pth"
STAGE3_CHECKPOINT="checkpoints/stage3_ai_coco_best.pth"
OUTPUT_DIR="results"
MAX_SAMPLES=200

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p fig

echo "📊 Dataset: COCO validation set"
echo "🖼️ Max samples: $MAX_SAMPLES"
echo "📁 Output: $OUTPUT_DIR"

# Run comprehensive comparison
python evaluation/comprehensive_comparison.py \
    --dataset coco \
    --data_dir $DATA_DIR \
    --split val \
    --image_size 256 \
    --max_samples $MAX_SAMPLES \
    --stage1_checkpoint $STAGE1_CHECKPOINT \
    --stage2_checkpoint $STAGE2_CHECKPOINT \
    --stage3_checkpoint $STAGE3_CHECKPOINT \
    --lambdas 128 256 512 1024 \
    --batch_size 4 \
    --enable_ablation \
    --enable_vision_tasks \
    --output_json $OUTPUT_DIR/comprehensive_comparison.json

echo ""
echo "✅ Comprehensive evaluation completed!"
echo "📋 Results saved to: $OUTPUT_DIR/comprehensive_comparison.json"
echo "📊 Figures saved to: fig/"
echo "📈 Individual CSV files:"
echo "   - $OUTPUT_DIR/comprehensive_comparison_compression.csv"
echo "   - $OUTPUT_DIR/comprehensive_comparison_vision_tasks.csv"
echo "   - $OUTPUT_DIR/comprehensive_comparison_ablation.csv" 