#!/bin/bash
# Run Full Lambda Evaluation for WAVENET-MV
# Script này sẽ chạy evaluation với tất cả λ values: [64, 128, 256, 512, 1024]

echo "🚀 RUNNING FULL LAMBDA EVALUATION FOR WAVENET-MV"
echo "=============================================="

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ This script requires CUDA GPU"
    echo "Run this on Ubuntu server (Máy B)"
    exit 1
fi

# Create results directory
mkdir -p results

# Lambda values to evaluate
LAMBDA_VALUES=(64 128 256 512 1024)

# Find best checkpoint
CHECKPOINT_CANDIDATES=(
    "checkpoints/stage2_compressor_coco_lambda256_best.pth"
    "checkpoints/stage2_compressor_coco_best.pth"
    "checkpoints/stage2_compressor_lambda256_best.pth"
)

CHECKPOINT=""
for candidate in "${CHECKPOINT_CANDIDATES[@]}"; do
    if [ -f "$candidate" ]; then
        CHECKPOINT="$candidate"
        echo "✅ Using checkpoint: $CHECKPOINT"
        break
    fi
done

if [ -z "$CHECKPOINT" ]; then
    echo "❌ No suitable checkpoint found!"
    exit 1
fi

# Run evaluation for each lambda
echo "🔄 Running evaluation for all lambda values..."

for lambda in "${LAMBDA_VALUES[@]}"; do
    echo ""
    echo "📊 Evaluating λ=$lambda..."
    
    OUTPUT_CSV="results/wavenet_mv_lambda${lambda}_evaluation.csv"
    
    python evaluation/codec_metrics.py \
        --checkpoint "$CHECKPOINT" \
        --dataset coco \
        --data_dir datasets/COCO \
        --split val \
        --lambdas $lambda \
        --batch_size 4 \
        --max_samples 500 \
        --output_csv "$OUTPUT_CSV" \
        --skip_entropy_update
    
    if [ $? -eq 0 ]; then
        echo "✅ λ=$lambda evaluation completed"
        echo "   Results saved to: $OUTPUT_CSV"
    else
        echo "❌ λ=$lambda evaluation failed"
    fi
done

# Combine all results
echo ""
echo "🔄 Combining results..."

# Create header
echo "lambda,psnr_db,ms_ssim,bpp,num_samples,dataset,split,image_size,model" > results/wavenet_mv_full_evaluation.csv

# Combine individual results
for lambda in "${LAMBDA_VALUES[@]}"; do
    LAMBDA_CSV="results/wavenet_mv_lambda${lambda}_evaluation.csv"
    if [ -f "$LAMBDA_CSV" ]; then
        # Skip header, append data
        tail -n +2 "$LAMBDA_CSV" >> results/wavenet_mv_full_evaluation.csv
    fi
done

echo "✅ Combined results saved to: results/wavenet_mv_full_evaluation.csv"

# Generate paper results
echo ""
echo "🔄 Generating paper results..."

python generate_paper_results.py \
    --checkpoint "$CHECKPOINT" \
    --dataset coco \
    --data_dir datasets/COCO \
    --split val \
    --max_samples 500 \
    --batch_size 4 \
    --skip_wavenet  # Skip since we already ran evaluation

if [ $? -eq 0 ]; then
    echo "✅ Paper results generated successfully"
    
    # Show summary
    echo ""
    echo "📊 EVALUATION SUMMARY:"
    echo "----------------------------------------"
    echo "Lambda values tested: ${LAMBDA_VALUES[*]}"
    echo "Checkpoint used: $CHECKPOINT"
    echo ""
    echo "📁 Results files:"
    ls -l results/
    
    # Show sample results
    echo ""
    echo "📋 Sample results (first 10 lines):"
    head -10 results/wavenet_mv_full_evaluation.csv
else
    echo "❌ Paper results generation failed"
fi

echo ""
echo "=============================================="
echo "🏁 FULL LAMBDA EVALUATION COMPLETED" 