@echo off
REM Git Workflow Script for WAVENET-MV Project
REM Usage: git_workflow.bat "commit message"

echo ========================================
echo 🚀 WAVENET-MV Git Workflow
echo ========================================

REM Activate environment
call wavenet-dev\Scripts\activate

REM Run quick test
echo 🔍 Running quick tests...
python quick_test.py
if %ERRORLEVEL% neq 0 (
    echo ❌ Tests failed! Please fix issues before pushing.
    pause
    exit /b 1
)

REM Format code
echo 🎨 Formatting code...
black . --line-length 88 --skip-string-normalization
isort . --profile black

REM Git operations
echo 📦 Preparing git commit...
git add .
git status

if "%1"=="" (
    set /p commit_msg="Enter commit message: "
) else (
    set commit_msg=%1
)

git commit -m "%commit_msg%"

echo 📤 Pushing to remote...
git push origin master

echo ✅ Successfully pushed to git!
echo 🖥️ Now you can pull on Ubuntu server and run training.

pause 