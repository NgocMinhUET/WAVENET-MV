@echo off
REM Quick Commit Script for WAVENET-MV
echo ========================================
echo 🚀 WAVENET-MV Quick Commit & Push
echo ========================================

REM Check git status
echo 📊 Current Git Status:
git status

echo.
echo ⚡ Choose commit type:
echo [1] Fix: Bug fix
echo [2] Feature: New feature  
echo [3] Update: General update
echo [4] Custom message
echo.

set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    set /p message="Describe the bug fixed: "
    set "commit_msg=Fix: %message%"
) else if "%choice%"=="2" (
    set /p message="Describe the new feature: "
    set "commit_msg=Feature: %message%"
) else if "%choice%"=="3" (
    set /p message="Describe the update: "
    set "commit_msg=Update: %message%"  
) else if "%choice%"=="4" (
    set /p commit_msg="Enter custom commit message: "
) else (
    echo ❌ Invalid choice. Using default message.
    set "commit_msg=Update: General changes"
)

echo.
echo 📝 Commit message: "%commit_msg%"
echo.

REM Add all changes
echo 📦 Adding all changes...
git add .

REM Commit
echo 💾 Committing...
git commit -m "%commit_msg%"

if %ERRORLEVEL% neq 0 (
    echo ❌ Commit failed. Please check for issues.
    pause
    exit /b 1
)

REM Push to remote
echo 🚀 Pushing to remote...
git push origin master

if %ERRORLEVEL% neq 0 (
    echo ❌ Push failed. Please check network and remote settings.
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ Successfully committed and pushed!
echo ========================================
echo 📋 Next steps:
echo 1. Go to Server (Máy B)
echo 2. Run: git pull origin master  
echo 3. Continue with training
echo ========================================

pause 