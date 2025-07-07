#!/bin/bash

# =============================================================================
# WAVENET-MV FULL TRAINING PIPELINE SCRIPT FOR SERVER
# =============================================================================
# Script này chạy toàn bộ quá trình training từ stage 1 đến stage 3
# và evaluation trên server
# =============================================================================

# Thiết lập môi trường
echo -e "\n🔧 SETTING UP ENVIRONMENT"
echo "----------------------------------------"

# Kiểm tra CUDA
nvidia-smi
if [ $? -ne 0 ]; then
    echo "❌ CUDA không khả dụng! Vui lòng kiểm tra cài đặt GPU."
    exit 1
fi

# Cài đặt dependencies nếu cần
pip install -r requirements.txt

# Tạo thư mục checkpoints và results nếu chưa tồn tại
mkdir -p checkpoints
mkdir -p results

# =============================================================================
# STAGE 1: TRAIN WAVELET MODEL
# =============================================================================
echo -e "\n🔄 STARTING STAGE 1: WAVELET TRAINING"
echo "----------------------------------------"

python training/stage1_train_wavelet.py \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --dataset coco \
    --data_dir datasets/COCO \
    --resume checkpoints/stage1_wavelet_coco_latest.pth

if [ $? -ne 0 ]; then
    echo "❌ Stage 1 training thất bại"
    exit 1
fi

echo "✅ Stage 1 hoàn thành thành công"

# =============================================================================
# STAGE 2: TRAIN COMPRESSOR MODEL với nhiều lambda values
# =============================================================================
echo -e "\n🔄 STARTING STAGE 2: COMPRESSOR TRAINING"
echo "----------------------------------------"

# Danh sách lambda values để train
lambda_values=(64 128 256 512 1024 2048)

for lambda in "${lambda_values[@]}"; do
    echo -e "\n🔄 Training compressor với lambda = $lambda"
    
    python training/stage2_train_compressor.py \
        --epochs 100 \
        --batch_size 8 \
        --learning_rate 1e-4 \
        --dataset coco \
        --data_dir datasets/COCO \
        --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
        --lambda_rd $lambda \
        --resume checkpoints/stage2_compressor_coco_lambda${lambda}_latest.pth
    
    if [ $? -ne 0 ]; then
        echo "❌ Stage 2 training thất bại với lambda = $lambda"
        continue
    fi
    
    echo "✅ Stage 2 hoàn thành thành công với lambda = $lambda"
done

# =============================================================================
# STAGE 3: TRAIN AI HEADS
# =============================================================================
echo -e "\n🔄 STARTING STAGE 3: AI HEADS TRAINING"
echo "----------------------------------------"

# Sử dụng lambda=256 làm default
python training/stage3_train_ai.py \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --dataset coco \
    --data_dir datasets/COCO \
    --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
    --stage2_checkpoint checkpoints/stage2_compressor_coco_lambda256_best.pth \
    --lambda_rd 256 \
    --enable_detection \
    --enable_segmentation \
    --resume checkpoints/stage3_ai_heads_coco_latest.pth

if [ $? -ne 0 ]; then
    echo "❌ Stage 3 training thất bại"
    exit 1
fi

echo "✅ Stage 3 hoàn thành thành công"

# =============================================================================
# EVALUATION: Đánh giá mô hình
# =============================================================================
echo -e "\n🔄 RUNNING COMPREHENSIVE EVALUATION"
echo "----------------------------------------"

# Tạo thư mục kết quả nếu chưa tồn tại
mkdir -p results

# Chạy đánh giá cho tất cả lambda values đã train
for lambda in "${lambda_values[@]}"; do
    echo -e "\n🔄 Evaluating model với lambda = $lambda"
    
    python evaluation/codec_metrics_final.py \
        --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
        --stage2_checkpoint checkpoints/stage2_compressor_coco_lambda${lambda}_best.pth \
        --stage3_checkpoint checkpoints/stage3_ai_heads_coco_best.pth \
        --dataset coco \
        --data_dir datasets/COCO \
        --split val \
        --max_samples 100 \
        --batch_size 4 \
        --output_file results/wavenet_mv_lambda${lambda}_evaluation.csv
    
    if [ $? -ne 0 ]; then
        echo "❌ Evaluation thất bại với lambda = $lambda"
        continue
    fi
    
    echo "✅ Evaluation hoàn thành với lambda = $lambda"
done

# =============================================================================
# GENERATE REPORTS: Tạo báo cáo tổng hợp
# =============================================================================
echo -e "\n🔄 GENERATING COMPREHENSIVE REPORTS"
echo "----------------------------------------"

# Tạo báo cáo tổng hợp từ các kết quả đánh giá
python evaluation/generate_summary_report.py \
    --input_dir results \
    --output_file results/wavenet_mv_comprehensive_results.csv

if [ $? -ne 0 ]; then
    echo "❌ Report generation thất bại"
    exit 1
fi

echo "✅ Report generation hoàn thành thành công"

# =============================================================================
# SUMMARY: Tóm tắt kết quả
# =============================================================================
echo -e "\n🎉 FULL TRAINING PIPELINE COMPLETED!"
echo "📊 Results saved in results/ directory"
echo "📈 TensorBoard logs in runs/ directory"
echo "💾 Checkpoints saved in checkpoints/ directory"

# In danh sách checkpoints
echo -e "\n📋 AVAILABLE CHECKPOINTS:"
ls -la checkpoints/

# In kết quả mẫu
echo -e "\n📊 EVALUATION RESULTS PREVIEW:"
head -n 5 results/wavenet_mv_comprehensive_results.csv

echo -e "\n✅ TRAINING PIPELINE HOÀN THÀNH!" 