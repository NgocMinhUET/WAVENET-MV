#!/bin/bash
# Run Full WAVENET-MV Training Pipeline
# Chạy toàn bộ 3 stages với tất cả lambda values

echo "🚀 RUNNING FULL WAVENET-MV TRAINING PIPELINE"
echo "============================================"

# Check if we're on the server (has CUDA)
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ This script must be run on Ubuntu server with CUDA"
    echo "Current machine doesn't have NVIDIA GPU"
    exit 1
fi



# Create directories
mkdir -p checkpoints
mkdir -p runs
mkdir -p results

# Stage 1: Train WaveletTransformCNN
echo "🔄 STAGE 1: TRAINING WAVELETTRANSFORMCNN"
echo "----------------------------------------"

python training/stage1_train_wavelet.py \
    --epochs 30 \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --dataset coco \
    --data_dir datasets/COCO \
    --resume checkpoints/stage1_wavelet_coco_best.pth

if [ $? -ne 0 ]; then
    echo "❌ Stage 1 training failed"
    exit 1
fi

echo "✅ Stage 1 completed successfully"

# Stage 2: Train CompressorVNVC with multiple lambda values
echo -e "\n🔄 STAGE 2: TRAINING COMPRESSORVNVC"
echo "----------------------------------------"

# Lambda values based on memory [[memory:645488]]
LAMBDA_VALUES=(64 128 256 512 1024 2048 4096)

for lambda in "${LAMBDA_VALUES[@]}"; do
    echo "🔄 Training with λ = $lambda"
    
    python training/stage2_train_compressor.py \
        --epochs 40 \
        --batch_size 8 \
        --learning_rate 2e-4 \
        --dataset coco \
        --data_dir datasets/COCO \
        --lambda_rd ${lambda} \
        --wavelet_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
        --resume checkpoints/stage2_compressor_coco_lambda${lambda}_best.pth
    
    if [ $? -ne 0 ]; then
        echo "❌ Stage 2 training failed for λ = $lambda"
        exit 1
    fi
    
    echo "✅ Stage 2 completed for λ = $lambda"
done

# Stage 3: Train AI Heads
echo -e "\n🔄 STAGE 3: TRAINING AI HEADS"
echo "----------------------------------------"

# Train with best lambda from Stage 2 (will be determined by evaluation)
python training/stage3_train_ai.py \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --dataset coco \
    --data_dir datasets/COCO \
    --wavelet_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
    --compressor_checkpoint checkpoints/stage2_compressor_coco_lambda256_best.pth \
    --resume checkpoints/stage3_ai_heads_coco_best.pth

if [ $? -ne 0 ]; then
    echo "❌ Stage 3 training failed"
    exit 1
fi

echo "✅ Stage 3 completed successfully"

# Run evaluation for all checkpoints
echo -e "\n🔄 RUNNING FULL EVALUATION"
echo "----------------------------------------"

python generate_paper_results.py \
    --checkpoint checkpoints/stage3_ai_heads_coco_best.pth \
    --dataset coco \
    --data_dir datasets/COCO \
    --split val \
    --max_samples 500 \
    --batch_size 4

if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed"
    exit 1
fi

echo "✅ Evaluation completed successfully"

echo -e "\n🎉 FULL TRAINING PIPELINE COMPLETED!"
echo "📊 Results saved in results/ directory"
echo "📈 TensorBoard logs in runs/ directory"
echo "💾 Checkpoints saved in checkpoints/ directory"

# Print available checkpoints
echo -e "\n📋 AVAILABLE CHECKPOINTS:"
ls -la checkpoints/

# Print sample results
echo -e "\n📊 EVALUATION RESULTS:"
head -n 5 results/wavenet_mv_full_evaluation.csv 