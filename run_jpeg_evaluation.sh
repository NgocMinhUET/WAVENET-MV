#!/bin/bash

# =============================================================================
# JPEG/JPEG2000 BASELINE EVALUATION SCRIPT FOR SERVER
# =============================================================================
# Script này chạy đánh giá baseline JPEG và JPEG2000 compression
# =============================================================================

echo -e "\n🔧 JPEG/JPEG2000 BASELINE EVALUATION"
echo "======================================="

# Kiểm tra môi trường
echo "Checking environment..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python3 không khả dụng!"
    exit 1
fi

# Kiểm tra dataset
DATASET_DIR="datasets/COCO_Official"
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ Dataset not found at $DATASET_DIR"
    echo "Please run: python datasets/setup_coco_official.py"
    exit 1
fi

if [ ! -d "$DATASET_DIR/val2017" ]; then
    echo "❌ val2017 directory not found"
    echo "Please setup COCO dataset first"
    exit 1
fi

echo "✅ Dataset found: $DATASET_DIR"

# Cài đặt và kiểm tra codec JPEG/JPEG2000
echo "Installing and testing JPEG/JPEG2000 codecs..."
python3 install_codecs.py

if [ $? -ne 0 ]; then
    echo "❌ Codec installation failed"
    exit 1
fi

# Cài đặt dependencies nếu cần
echo "Installing additional dependencies..."
pip3 install opencv-contrib-python pillow tqdm pandas scikit-image pathlib2 pillow-heif imageio

# Tạo results directory
mkdir -p results
mkdir -p results/jpeg_baseline

# =============================================================================
# RUN JPEG/JPEG2000 EVALUATION
# =============================================================================

echo -e "\n🔄 RUNNING JPEG/JPEG2000 EVALUATION"
echo "-----------------------------------"

# Chọn script evaluation tốt nhất
EVAL_SCRIPT="server_jpeg_evaluation.py"
if [ -f "improved_jpeg_evaluation.py" ]; then
    echo "📈 Using improved evaluation script with better codecs"
    EVAL_SCRIPT="improved_jpeg_evaluation.py"
fi

# Chạy evaluation với different configurations
echo "Running quick evaluation (50 images)..."
python3 $EVAL_SCRIPT \
    --data_dir "$DATASET_DIR" \
    --max_images 50 \
    --quality_levels 10 20 30 40 50 60 70 80 90 95 \
    --output_dir results/jpeg_baseline \
    --output_file jpeg_baseline_quick.csv

if [ $? -eq 0 ]; then
    echo "✅ Quick evaluation completed successfully"
else
    echo "❌ Quick evaluation failed"
    exit 1
fi

# Chạy evaluation đầy đủ
echo -e "\nRunning full evaluation (200 images)..."
python3 $EVAL_SCRIPT \
    --data_dir "$DATASET_DIR" \
    --max_images 200 \
    --quality_levels 10 20 30 40 50 60 70 80 90 95 \
    --output_dir results/jpeg_baseline \
    --output_file jpeg_baseline_full.csv

if [ $? -eq 0 ]; then
    echo "✅ Full evaluation completed successfully"
else
    echo "❌ Full evaluation failed"
    exit 1
fi

# =============================================================================
# GENERATE SUMMARY REPORT
# =============================================================================

echo -e "\n📊 GENERATING SUMMARY REPORT"
echo "----------------------------"

python3 -c "
import pandas as pd
import os

# Load results
results_dir = 'results/jpeg_baseline'
quick_file = os.path.join(results_dir, 'jpeg_baseline_quick.csv')
full_file = os.path.join(results_dir, 'jpeg_baseline_full.csv')

if os.path.exists(full_file):
    df = pd.read_csv(full_file)
    print('📊 FULL EVALUATION SUMMARY (200 images):')
elif os.path.exists(quick_file):
    df = pd.read_csv(quick_file)
    print('📊 QUICK EVALUATION SUMMARY (50 images):')
else:
    print('❌ No results found')
    exit(1)

print('=' * 60)

# Summary by codec
for codec in ['JPEG', 'JPEG2000']:
    codec_data = df[df['codec'] == codec]
    if not codec_data.empty:
        print(f'\n{codec} RESULTS:')
        print(f'  Quality levels: {sorted(codec_data[\"quality\"].unique())}')
        print(f'  PSNR range: {codec_data[\"psnr\"].min():.2f} - {codec_data[\"psnr\"].max():.2f} dB')
        print(f'  SSIM range: {codec_data[\"ssim\"].min():.4f} - {codec_data[\"ssim\"].max():.4f}')
        print(f'  BPP range:  {codec_data[\"bpp\"].min():.4f} - {codec_data[\"bpp\"].max():.4f}')
        
        # Best quality results
        best_quality = codec_data[codec_data['quality'] == codec_data['quality'].max()]
        print(f'  Best quality (Q={best_quality[\"quality\"].iloc[0]}):')
        print(f'    PSNR: {best_quality[\"psnr\"].mean():.2f} dB')
        print(f'    SSIM: {best_quality[\"ssim\"].mean():.4f}')
        print(f'    BPP:  {best_quality[\"bpp\"].mean():.4f}')

print('\n✅ Summary report generated successfully')
"

# =============================================================================
# COMPARISON WITH EXPECTED WAVENET-MV RESULTS
# =============================================================================

echo -e "\n🔍 COMPARISON WITH EXPECTED WAVENET-MV RESULTS"
echo "----------------------------------------------"

python3 -c "
import pandas as pd
import os

# Load JPEG results
results_dir = 'results/jpeg_baseline'
full_file = os.path.join(results_dir, 'jpeg_baseline_full.csv')
quick_file = os.path.join(results_dir, 'jpeg_baseline_quick.csv')

if os.path.exists(full_file):
    df = pd.read_csv(full_file)
elif os.path.exists(quick_file):
    df = pd.read_csv(quick_file)
else:
    print('❌ No JPEG results found')
    exit(1)

print('📈 WAVENET-MV EXPECTED PERFORMANCE vs JPEG/JPEG2000:')
print('=' * 60)

# Expected WAVENET-MV results (from memory)
expected_wavenet = {
    'PSNR': '28-38 dB',
    'BPP': '0.1-8.0',
    'SSIM': '0.85-0.95 (estimated)'
}

print(f'WAVENET-MV EXPECTED:')
print(f'  PSNR: {expected_wavenet[\"PSNR\"]}')
print(f'  BPP:  {expected_wavenet[\"BPP\"]}')
print(f'  SSIM: {expected_wavenet[\"SSIM\"]}')

# JPEG best case
jpeg_best = df[df['codec'] == 'JPEG'].nlargest(10, 'psnr')
print(f'\nJPEG BEST CASE (top 10):')
print(f'  PSNR: {jpeg_best[\"psnr\"].min():.2f} - {jpeg_best[\"psnr\"].max():.2f} dB')
print(f'  BPP:  {jpeg_best[\"bpp\"].min():.4f} - {jpeg_best[\"bpp\"].max():.4f}')
print(f'  SSIM: {jpeg_best[\"ssim\"].min():.4f} - {jpeg_best[\"ssim\"].max():.4f}')

# JPEG2000 best case
jp2_best = df[df['codec'] == 'JPEG2000'].nlargest(10, 'psnr')
print(f'\nJPEG2000 BEST CASE (top 10):')
print(f'  PSNR: {jp2_best[\"psnr\"].min():.2f} - {jp2_best[\"psnr\"].max():.2f} dB')
print(f'  BPP:  {jp2_best[\"bpp\"].min():.4f} - {jp2_best[\"bpp\"].max():.4f}')
print(f'  SSIM: {jp2_best[\"ssim\"].min():.4f} - {jp2_best[\"ssim\"].max():.4f}')

print('\n💡 ANALYSIS:')
print('  - WAVENET-MV should achieve competitive PSNR with lower BPP')
print('  - Focus on BPP efficiency: WAVENET-MV targets 0.1-8.0 BPP')
print('  - JPEG/JPEG2000 provide good baseline for comparison')
"

# =============================================================================
# FINAL SUMMARY
# =============================================================================

echo -e "\n🎉 JPEG/JPEG2000 BASELINE EVALUATION COMPLETED!"
echo "==============================================="
echo "📂 Results saved in: results/jpeg_baseline/"
echo "📊 CSV files:"
ls -la results/jpeg_baseline/*.csv
echo ""
echo "✅ Baseline evaluation ready for WAVENET-MV comparison"
echo "📈 Use these results to benchmark WAVENET-MV performance"
echo ""
echo "🚀 Next steps:"
echo "  1. Fix WAVENET-MV training pipeline issues"
echo "  2. Run actual WAVENET-MV training"
echo "  3. Compare results with JPEG/JPEG2000 baselines" 