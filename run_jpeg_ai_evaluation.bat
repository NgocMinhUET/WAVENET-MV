@echo off
echo ====================================
echo 📸 JPEG AI ACCURACY EVALUATION
echo ====================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

REM Check dataset
if not exist "datasets\COCO\val2017" (
    echo ❌ COCO dataset not found!
    echo Please setup dataset first:
    echo   python datasets/setup_coco_official.py
    pause
    exit /b 1
)

echo ✅ Python and dataset found
echo.

REM Install required packages
echo 📦 Installing required packages...
pip install -q opencv-contrib-python pillow tqdm pandas scikit-image
pip install -q ultralytics
echo.

REM Create results directory
mkdir results\jpeg_ai_accuracy 2>nul

echo 🚀 Starting JPEG AI accuracy evaluation...
echo This will evaluate JPEG compression with AI task performance
echo Estimated time: 5-15 minutes for 50 images
echo.

REM Run JPEG AI evaluation
python evaluate_jpeg_ai_accuracy.py ^
    --data_dir datasets/COCO ^
    --max_images 50 ^
    --quality_levels 10 20 30 40 50 60 70 80 90 95 ^
    --output_dir results/jpeg_ai_accuracy ^
    --temp_dir temp_jpeg

if %errorlevel% neq 0 (
    echo ❌ JPEG AI evaluation failed
    pause
    exit /b 1
)

echo.
echo ✅ JPEG AI accuracy evaluation completed!
echo.
echo 📊 Results saved to: results/jpeg_ai_accuracy/
echo 📋 Generated files:
dir results\jpeg_ai_accuracy\
echo.
echo 📈 Files created:
echo   - jpeg_ai_accuracy.csv        (Raw data)
echo   - jpeg_latex_table.txt        (LaTeX table for paper)
echo.
echo 🎯 Key results preview:
echo Reading results...
python -c "
import pandas as pd
import numpy as np
from pathlib import Path

try:
    df = pd.read_csv('results/jpeg_ai_accuracy/jpeg_ai_accuracy.csv')
    print('📊 JPEG PERFORMANCE SUMMARY:')
    print('=' * 40)
    
    for quality in [10, 30, 50, 70, 90]:
        q_data = df[df['quality'] == quality]
        if not q_data.empty:
            avg_psnr = q_data['psnr'].mean()
            avg_ssim = q_data['ssim'].mean()
            avg_bpp = q_data['bpp'].mean()
            avg_map = q_data['mAP'].mean()
            
            print(f'Q={quality:2d}: PSNR={avg_psnr:5.1f}dB, SSIM={avg_ssim:.3f}, BPP={avg_bpp:.3f}, mAP={avg_map:.3f}')
    
    print('\n✅ Use jpeg_latex_table.txt for your paper!')
    
except Exception as e:
    print(f'Could not read results: {e}')
"

echo.
echo 🎉 JPEG AI Accuracy Evaluation Completed!
echo.
echo 📝 Next steps:
echo   1. Copy jpeg_latex_table.txt content to your paper
echo   2. Use jpeg_ai_accuracy.csv for further analysis
echo   3. Compare with WAVENET-MV results (when available)
echo.
pause 