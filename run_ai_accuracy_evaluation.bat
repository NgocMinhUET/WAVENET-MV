@echo off
echo ========================================
echo 🤖 AI ACCURACY EVALUATION FOR CODECS
echo ========================================
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
pip install -q ultralytics segmentation-models-pytorch
pip install -q torch torchvision
echo.

REM Create results directory
mkdir results\ai_accuracy 2>nul

echo 🚀 Starting AI accuracy evaluation...
echo This will take 10-30 minutes depending on your hardware
echo.

REM Run AI accuracy evaluation
python evaluate_ai_accuracy.py ^
    --data_dir datasets/COCO ^
    --max_images 50 ^
    --codecs JPEG JPEG2000 ^
    --quality_levels 10 30 50 70 90 ^
    --output_dir results/ai_accuracy ^
    --temp_dir temp_compressed

if %errorlevel% neq 0 (
    echo ❌ AI accuracy evaluation failed
    pause
    exit /b 1
)

echo.
echo ✅ AI accuracy evaluation completed!
echo.
echo 📊 Results saved to: results/ai_accuracy/
echo 📋 Files generated:
dir results\ai_accuracy\
echo.

REM Optional: Run comprehensive evaluation
echo 🔄 Do you want to run full comprehensive evaluation? (y/n)
set /p choice=
if /i "%choice%"=="y" (
    echo.
    echo 🚀 Running comprehensive evaluation...
    bash run_comprehensive_evaluation.sh
    if %errorlevel% neq 0 (
        echo ⚠️ Comprehensive evaluation had issues
    )
)

echo.
echo 🎉 Evaluation completed!
echo.
echo 📈 Next steps:
echo   1. Check results/ai_accuracy/ai_accuracy_evaluation.csv
echo   2. Use results for paper table
echo   3. Compare AI performance vs compression trade-offs
echo.
pause 