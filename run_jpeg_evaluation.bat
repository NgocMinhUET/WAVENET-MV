@echo off
REM =============================================================================
REM JPEG/JPEG2000 BASELINE EVALUATION SCRIPT FOR WINDOWS
REM =============================================================================

echo.
echo 🔧 JPEG/JPEG2000 BASELINE EVALUATION
echo =====================================

REM Kiểm tra Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python first
    pause
    exit /b 1
)

REM Kiểm tra dataset
set "DATASET_DIR=datasets\COCO_Official"
if not exist "%DATASET_DIR%" (
    echo ❌ Dataset not found at %DATASET_DIR%
    echo Please run: python datasets\setup_coco_official.py
    pause
    exit /b 1
)

if not exist "%DATASET_DIR%\val2017" (
    echo ❌ val2017 directory not found
    echo Please setup COCO dataset first
    pause
    exit /b 1
)

echo ✅ Dataset found: %DATASET_DIR%

REM Cài đặt và kiểm tra codec JPEG/JPEG2000
echo.
echo Installing and testing JPEG/JPEG2000 codecs...
python install_codecs.py

if %errorlevel% neq 0 (
    echo ❌ Codec installation failed
    pause
    exit /b 1
)

REM Cài đặt dependencies
echo.
echo Installing additional dependencies...
pip install opencv-contrib-python pillow tqdm pandas scikit-image imageio

REM Tạo results directory
mkdir results 2>nul
mkdir results\jpeg_baseline 2>nul

REM =============================================================================
REM RUN JPEG/JPEG2000 EVALUATION
REM =============================================================================

echo.
echo 🔄 RUNNING JPEG/JPEG2000 EVALUATION
echo -----------------------------------

REM Chọn script evaluation tốt nhất
set "EVAL_SCRIPT=server_jpeg_evaluation.py"
if exist "improved_jpeg_evaluation.py" (
    echo 📈 Using improved evaluation script with better codecs
    set "EVAL_SCRIPT=improved_jpeg_evaluation.py"
)

REM Chạy evaluation quick
echo.
echo Running quick evaluation (50 images)...
python %EVAL_SCRIPT% ^
    --data_dir "%DATASET_DIR%" ^
    --max_images 50 ^
    --quality_levels 10 20 30 40 50 60 70 80 90 95 ^
    --output_dir results/jpeg_baseline ^
    --output_file jpeg_baseline_quick.csv

if %errorlevel% neq 0 (
    echo ❌ Quick evaluation failed
    pause
    exit /b 1
)

echo ✅ Quick evaluation completed successfully

REM Chạy evaluation full
echo.
echo Running full evaluation (200 images)...
python %EVAL_SCRIPT% ^
    --data_dir "%DATASET_DIR%" ^
    --max_images 200 ^
    --quality_levels 10 20 30 40 50 60 70 80 90 95 ^
    --output_dir results/jpeg_baseline ^
    --output_file jpeg_baseline_full.csv

if %errorlevel% neq 0 (
    echo ❌ Full evaluation failed
    pause
    exit /b 1
)

echo ✅ Full evaluation completed successfully

REM =============================================================================
REM GENERATE SUMMARY REPORT
REM =============================================================================

echo.
echo 📊 GENERATING SUMMARY REPORT
echo ----------------------------

python -c "
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

REM =============================================================================
REM FINAL SUMMARY
REM =============================================================================

echo.
echo 🎉 JPEG/JPEG2000 BASELINE EVALUATION COMPLETED!
echo ===============================================
echo 📂 Results saved in: results/jpeg_baseline/
echo 📊 CSV files:
dir /b results\jpeg_baseline\*.csv
echo.
echo ✅ Baseline evaluation ready for WAVENET-MV comparison
echo 📈 Use these results to benchmark WAVENET-MV performance
echo.
echo 🚀 Next steps:
echo   1. Fix WAVENET-MV training pipeline issues
echo   2. Run actual WAVENET-MV training
echo   3. Compare results with JPEG/JPEG2000 baselines

pause 