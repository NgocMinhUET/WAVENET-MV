@echo off
REM Initialize Git Repository for WAVENET-MV Project
echo ========================================
echo 🚀 Initializing Git Repository
echo ========================================

REM Check if git is installed
git --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Git is not installed. Please install Git first.
    echo Download from: https://git-scm.com/download/win
    pause
    exit /b 1
)

REM Initialize git repository
echo 📦 Initializing git repository...
git init

REM Add remote repository (you'll need to update this)
echo 🔗 Setting up remote repository...
echo Please create a repository on GitHub/GitLab first, then update this script with your URL
echo Example: git remote add origin https://github.com/yourusername/wavenet-mv.git
set /p remote_url="Enter your remote repository URL (or press Enter to skip): "

if not "%remote_url%"=="" (
    git remote add origin %remote_url%
    echo ✅ Remote repository added: %remote_url%
) else (
    echo ⚠️ Skipped remote repository setup. You can add it later with:
    echo    git remote add origin YOUR_REPO_URL
)

REM Configure Git user (if not already configured)
echo 👤 Configuring Git user...
git config user.name >nul 2>&1
if %ERRORLEVEL% neq 0 (
    set /p git_name="Enter your Git username: "
    git config --global user.name "%git_name%"
)

git config user.email >nul 2>&1
if %ERRORLEVEL% neq 0 (
    set /p git_email="Enter your Git email: "
    git config --global user.email "%git_email%"
)

REM Create .gitignore if not exists
if not exist ".gitignore" (
    echo 📝 Creating .gitignore file...
    copy /y .gitignore.template .gitignore >nul 2>&1
)

REM Initial commit
echo 📝 Creating initial commit...
git add .
git commit -m "Initial commit: WAVENET-MV project setup"

REM Show status
echo 📊 Git status:
git status

echo ========================================
echo ✅ Git repository initialized successfully!
echo ========================================
echo 📋 Next steps:
echo 1. Create a repository on GitHub/GitLab if you haven't
echo 2. Update remote URL if needed: git remote set-url origin YOUR_URL
echo 3. Push to remote: git push -u origin main
echo 4. Use git_workflow.bat for future commits
echo ========================================

pause 