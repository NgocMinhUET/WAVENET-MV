#!/bin/bash

# =============================================================================
# WAVENET-MV COMPREHENSIVE EVALUATION SCRIPT FOR SERVER
# =============================================================================
# Script này chạy đánh giá đầy đủ cho các mô hình đã được train
# Giả định rằng các checkpoints đã tồn tại
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

# Tạo thư mục results nếu chưa tồn tại
mkdir -p results

# Kiểm tra xem có checkpoints không
if [ ! -d "checkpoints" ] || [ -z "$(ls -A checkpoints/)" ]; then
    echo "❌ Không tìm thấy checkpoints! Vui lòng chạy training trước."
    exit 1
fi

# =============================================================================
# EVALUATION: Đánh giá mô hình với nhiều lambda values
# =============================================================================
echo -e "\n🔄 RUNNING COMPREHENSIVE EVALUATION"
echo "----------------------------------------"

# Danh sách lambda values để đánh giá
lambda_values=(64 128 256 512 1024 2048)

for lambda in "${lambda_values[@]}"; do
    # Kiểm tra xem checkpoint có tồn tại không
    if [ ! -f "checkpoints/stage2_compressor_coco_lambda${lambda}_best.pth" ]; then
        echo "⚠️ Không tìm thấy checkpoint cho lambda = $lambda, bỏ qua..."
        continue
    fi
    
    echo -e "\n🔄 Evaluating model với lambda = $lambda"
    
    # Đánh giá codec metrics (PSNR, MS-SSIM, BPP)
    python evaluation/codec_metrics_final.py \
        --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
        --stage2_checkpoint checkpoints/stage2_compressor_coco_lambda${lambda}_best.pth \
        --dataset coco \
        --data_dir datasets/COCO \
        --split val \
        --max_samples 100 \
        --batch_size 4 \
        --output_csv results/wavenet_mv_lambda${lambda}_codec_metrics.csv
    
    if [ $? -ne 0 ]; then
        echo "❌ Codec evaluation thất bại với lambda = $lambda"
        continue
    fi
    
    # Đánh giá AI metrics (Detection, Segmentation accuracy)
    python evaluation/vcm_metrics.py \
        --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
        --stage2_checkpoint checkpoints/stage2_compressor_coco_lambda${lambda}_best.pth \
        --stage3_checkpoint checkpoints/stage3_ai_heads_coco_best.pth \
        --dataset coco \
        --data_dir datasets/COCO \
        --split val \
        --max_samples 100 \
        --batch_size 4 \
        --enable_detection \
        --enable_segmentation \
        --lambda_rd ${lambda} \
        --output_json results/wavenet_mv_lambda${lambda}_ai_metrics.json
    
    if [ $? -ne 0 ]; then
        echo "❌ AI metrics evaluation thất bại với lambda = $lambda"
        continue
    fi
    
    echo "✅ Evaluation hoàn thành với lambda = $lambda"
done

# =============================================================================
# BASELINE COMPARISON: So sánh với các baseline
# =============================================================================
echo -e "\n🔄 RUNNING BASELINE COMPARISON"
echo "----------------------------------------"

# Chạy so sánh với các codec tiêu chuẩn (JPEG, WebP, VTM)
python evaluation/compare_baselines.py \
    --dataset coco \
    --data_dir datasets/COCO \
    --split val \
    --max_samples 50 \
    --methods JPEG WebP PNG \
    --qualities 30 50 70 90 \
    --output_csv results/baseline_comparison.csv

if [ $? -ne 0 ]; then
    echo "❌ Baseline comparison thất bại"
else
    echo "✅ Baseline comparison hoàn thành thành công"
fi

# =============================================================================
# GENERATE COMPREHENSIVE REPORTS: Tạo báo cáo tổng hợp
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

# Tạo bảng và biểu đồ so sánh
python evaluation/generate_tables.py \
    --results_dir results \
    --output_file tables/wavenet_mv_tables.csv

if [ $? -ne 0 ]; then
    echo "❌ Table generation thất bại"
    exit 1
fi

echo "✅ Report generation hoàn thành thành công"

# =============================================================================
# ABLATION STUDIES: Phân tích đóng góp của các thành phần
# =============================================================================
echo -e "\n🔄 RUNNING ABLATION STUDIES"
echo "----------------------------------------"

# Phân tích đóng góp của wavelet transform
python evaluation/statistical_analysis.py \
    --input_file results/wavenet_mv_comprehensive_results.csv \
    --output_file results/wavelet_contribution_analysis.csv \
    --analysis_type wavelet

if [ $? -ne 0 ]; then
    echo "❌ Wavelet contribution analysis thất bại"
else
    echo "✅ Wavelet contribution analysis hoàn thành thành công"
fi

# =============================================================================
# SUMMARY: Tóm tắt kết quả
# =============================================================================
echo -e "\n🎉 EVALUATION COMPLETED!"
echo "📊 Results saved in results/ directory"
echo "📈 Tables and figures saved in tables/ directory"

# In kết quả mẫu
echo -e "\n📊 EVALUATION RESULTS PREVIEW:"
head -n 5 results/wavenet_mv_comprehensive_results.csv

echo -e "\n✅ EVALUATION PIPELINE HOÀN THÀNH!" 