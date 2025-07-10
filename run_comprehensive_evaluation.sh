#!/bin/bash

# =============================================================================
# COMPREHENSIVE EVALUATION: COMPRESSION + AI ACCURACY
# =============================================================================
# Script này chạy đánh giá toàn diện bao gồm:
# 1. JPEG/JPEG2000 compression metrics (PSNR, SSIM, BPP)
# 2. AI task performance (object detection, segmentation)
# 3. WAVENET-MV comparison (nếu có)
# =============================================================================

echo -e "\n🔧 COMPREHENSIVE COMPRESSION + AI ACCURACY EVALUATION"
echo "======================================================="

# Kiểm tra môi trường
echo "Checking environment..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python3 không khả dụng!"
    exit 1
fi

# Kiểm tra dataset
DATASET_DIR="datasets/COCO"
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

# Cài đặt dependencies
echo "Installing dependencies..."
pip3 install -q opencv-contrib-python pillow tqdm pandas scikit-image pathlib2 pillow-heif imageio
pip3 install -q ultralytics  # YOLOv8
pip3 install -q segmentation-models-pytorch  # Segmentation models
pip3 install -q torch torchvision  # PyTorch

# Kiểm tra GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Tạo results directory
mkdir -p results
mkdir -p results/comprehensive_evaluation

# =============================================================================
# PHASE 1: COMPRESSION METRICS EVALUATION
# =============================================================================

echo -e "\n📊 PHASE 1: COMPRESSION METRICS EVALUATION"
echo "-------------------------------------------"

# Chạy evaluation compression cơ bản trước
echo "Running compression metrics evaluation..."
python3 server_jpeg_evaluation.py \
    --data_dir "$DATASET_DIR" \
    --max_images 100 \
    --quality_levels 10 20 30 40 50 60 70 80 90 95 \
    --output_dir results/comprehensive_evaluation \
    --output_file compression_metrics.csv

if [ $? -ne 0 ]; then
    echo "❌ Compression metrics evaluation failed"
    exit 1
fi

echo "✅ Compression metrics completed"

# =============================================================================
# PHASE 2: AI ACCURACY EVALUATION
# =============================================================================

echo -e "\n🤖 PHASE 2: AI ACCURACY EVALUATION"
echo "----------------------------------"

# Chạy AI accuracy evaluation
echo "Running AI accuracy evaluation (this may take longer)..."
python3 evaluate_ai_accuracy.py \
    --data_dir "$DATASET_DIR" \
    --max_images 50 \
    --codecs JPEG JPEG2000 \
    --quality_levels 10 30 50 70 90 \
    --output_dir results/comprehensive_evaluation \
    --temp_dir temp_compressed

if [ $? -ne 0 ]; then
    echo "⚠️ AI accuracy evaluation had issues, but continuing..."
fi

echo "✅ AI accuracy evaluation completed"

# =============================================================================
# PHASE 3: WAVENET-MV EVALUATION (if available)
# =============================================================================

echo -e "\n🚀 PHASE 3: WAVENET-MV EVALUATION"
echo "---------------------------------"

# Kiểm tra xem có WAVENET-MV model không
if [ -f "models/wavenet_mv_trained.pth" ] || [ -f "checkpoints/best_model.pth" ]; then
    echo "Found WAVENET-MV model, running evaluation..."
    
    # Chạy WAVENET-MV evaluation
    python3 evaluate_wavenet_mv.py \
        --data_dir "$DATASET_DIR" \
        --max_images 50 \
        --lambda_values 64 128 256 512 1024 2048 \
        --output_dir results/comprehensive_evaluation \
        --model_path checkpoints/best_model.pth
    
    if [ $? -eq 0 ]; then
        echo "✅ WAVENET-MV evaluation completed"
    else
        echo "⚠️ WAVENET-MV evaluation failed"
    fi
else
    echo "⚠️ WAVENET-MV model not found, skipping evaluation"
    echo "To include WAVENET-MV comparison:"
    echo "  1. Train WAVENET-MV model first"
    echo "  2. Save model to checkpoints/best_model.pth"
    echo "  3. Re-run this script"
fi

# =============================================================================
# PHASE 4: COMBINED ANALYSIS AND REPORTING
# =============================================================================

echo -e "\n📈 PHASE 4: COMBINED ANALYSIS"
echo "-----------------------------"

python3 -c "
import pandas as pd
import numpy as np
import os
from pathlib import Path

results_dir = Path('results/comprehensive_evaluation')

print('📊 COMPREHENSIVE EVALUATION SUMMARY')
print('=' * 60)

# Load compression metrics
compression_file = results_dir / 'compression_metrics.csv'
if compression_file.exists():
    comp_df = pd.read_csv(compression_file)
    print(f'✅ Compression metrics: {len(comp_df)} data points')
else:
    print('❌ Compression metrics not found')
    comp_df = None

# Load AI accuracy metrics
ai_file = results_dir / 'ai_accuracy_evaluation.csv'
if ai_file.exists():
    ai_df = pd.read_csv(ai_file)
    print(f'✅ AI accuracy metrics: {len(ai_df)} data points')
else:
    print('❌ AI accuracy metrics not found')
    ai_df = None

# Load WAVENET-MV metrics (if available)
wavenet_file = results_dir / 'wavenet_mv_evaluation.csv'
if wavenet_file.exists():
    wavenet_df = pd.read_csv(wavenet_file)
    print(f'✅ WAVENET-MV metrics: {len(wavenet_df)} data points')
else:
    print('⚠️ WAVENET-MV metrics not available')
    wavenet_df = None

print('\n📋 DETAILED RESULTS:')
print('-' * 40)

# Analyze JPEG results
if ai_df is not None:
    jpeg_data = ai_df[ai_df['codec'] == 'JPEG']
    if not jpeg_data.empty:
        print('\nJPEG PERFORMANCE:')
        for quality in sorted(jpeg_data['quality'].unique()):
            q_data = jpeg_data[jpeg_data['quality'] == quality]
            avg_psnr = q_data['psnr'].mean()
            avg_ssim = q_data['ssim'].mean() 
            avg_bpp = q_data['bpp'].mean()
            avg_map = q_data['mAP'].mean()
            avg_miou = q_data['mIoU'].mean()
            
            # Handle infinite PSNR
            psnr_str = f'{avg_psnr:.2f}' if np.isfinite(avg_psnr) else 'inf'
            
            print(f'  Q={quality:2d}: PSNR={psnr_str:>6}dB, SSIM={avg_ssim:.3f}, '
                  f'BPP={avg_bpp:.3f}, mAP={avg_map:.3f}, mIoU={avg_miou:.3f}')

# Analyze JPEG2000 results
if ai_df is not None:
    jp2_data = ai_df[ai_df['codec'] == 'JPEG2000']
    if not jp2_data.empty:
        print('\nJPEG2000 PERFORMANCE:')
        for quality in sorted(jp2_data['quality'].unique()):
            q_data = jp2_data[jp2_data['quality'] == quality]
            avg_psnr = q_data['psnr'].mean()
            avg_ssim = q_data['ssim'].mean()
            avg_bpp = q_data['bpp'].mean() 
            avg_map = q_data['mAP'].mean()
            avg_miou = q_data['mIoU'].mean()
            
            psnr_str = f'{avg_psnr:.2f}' if np.isfinite(avg_psnr) else 'inf'
            
            print(f'  Q={quality:2d}: PSNR={psnr_str:>6}dB, SSIM={avg_ssim:.3f}, '
                  f'BPP={avg_bpp:.3f}, mAP={avg_map:.3f}, mIoU={avg_miou:.3f}')

# Analyze WAVENET-MV results (if available)
if wavenet_df is not None:
    print('\nWAVENET-MV PERFORMANCE:')
    for lambda_val in sorted(wavenet_df['lambda'].unique()):
        l_data = wavenet_df[wavenet_df['lambda'] == lambda_val]
        avg_psnr = l_data['psnr'].mean()
        avg_ssim = l_data['ssim'].mean()
        avg_bpp = l_data['bpp'].mean()
        avg_map = l_data['mAP'].mean()
        avg_miou = l_data['mIoU'].mean()
        
        print(f'  λ={lambda_val:4d}: PSNR={avg_psnr:6.2f}dB, SSIM={avg_ssim:.3f}, '
              f'BPP={avg_bpp:.3f}, mAP={avg_map:.3f}, mIoU={avg_miou:.3f}')

print('\n💡 ANALYSIS NOTES:')
print('- PSNR=inf indicates lossless compression (perfect reconstruction)')
print('- SSIM=1.0 indicates perfect structural similarity')
print('- Higher mAP = better object detection performance')
print('- Higher mIoU = better segmentation performance')
print('- Lower BPP = better compression efficiency')

# Generate combined CSV for paper
if ai_df is not None:
    print('\n📝 Generating combined results for paper...')
    
    # Prepare data for paper table
    paper_results = []
    
    # JPEG results
    jpeg_data = ai_df[ai_df['codec'] == 'JPEG']
    for quality in sorted(jpeg_data['quality'].unique()):
        q_data = jpeg_data[jpeg_data['quality'] == quality]
        if not q_data.empty:
            paper_results.append({
                'Method': 'JPEG',
                'Setting': f'Q={quality}',
                'PSNR_dB': f'{q_data[\"psnr\"].mean():.1f} ± {q_data[\"psnr\"].std():.1f}',
                'MS_SSIM': f'{q_data[\"ssim\"].mean():.3f} ± {q_data[\"ssim\"].std():.3f}',
                'BPP': f'{q_data[\"bpp\"].mean():.3f} ± {q_data[\"bpp\"].std():.3f}',
                'AI_Accuracy_mAP': f'{q_data[\"mAP\"].mean():.3f} ± {q_data[\"mAP\"].std():.3f}'
            })
    
    # WAVENET-MV results (if available)
    if wavenet_df is not None:
        for lambda_val in sorted(wavenet_df['lambda'].unique()):
            l_data = wavenet_df[wavenet_df['lambda'] == lambda_val]
            if not l_data.empty:
                paper_results.append({
                    'Method': 'WAVENET-MV',
                    'Setting': f'λ={lambda_val}',
                    'PSNR_dB': f'{l_data[\"psnr\"].mean():.1f} ± {l_data[\"psnr\"].std():.1f}',
                    'MS_SSIM': f'{l_data[\"ssim\"].mean():.3f} ± {l_data[\"ssim\"].std():.3f}',
                    'BPP': f'{l_data[\"bpp\"].mean():.3f} ± {l_data[\"bpp\"].std():.3f}',
                    'AI_Accuracy_mAP': f'{l_data[\"mAP\"].mean():.3f} ± {l_data[\"mAP\"].std():.3f}'
                })
    
    # Save paper results
    if paper_results:
        paper_df = pd.DataFrame(paper_results)
        paper_file = results_dir / 'paper_results_table.csv'
        paper_df.to_csv(paper_file, index=False)
        print(f'📊 Paper results table saved: {paper_file}')
        
        # Show table preview
        print('\n📋 PAPER TABLE PREVIEW:')
        print(paper_df.to_string(index=False))

print('\n✅ Combined analysis completed!')
"

# =============================================================================
# FINAL SUMMARY
# =============================================================================

echo -e "\n🎉 COMPREHENSIVE EVALUATION COMPLETED!"
echo "====================================="
echo "📂 Results location: results/comprehensive_evaluation/"
echo ""
echo "📊 Generated files:"
ls -la results/comprehensive_evaluation/
echo ""
echo "✅ Ready for paper results!"
echo ""
echo "📈 Next steps:"
echo "  1. Use 'paper_results_table.csv' for LaTeX table"
echo "  2. Analyze AI accuracy vs compression trade-offs"
echo "  3. Compare WAVENET-MV vs traditional codecs"
echo "  4. Generate rate-distortion curves" 