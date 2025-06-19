# 🪟 Windows Development Setup - WAVENET-MV

## 1. Environment Setup
```bash
# Tạo conda environment (recommended)
conda create -n wavenet-dev python=3.9
conda activate wavenet-dev

# Hoặc dùng venv
python -m venv wavenet-dev
wavenet-dev\Scripts\activate
```

## 2. Install Dependencies (CPU Only for Development)
```bash
# Install PyTorch CPU version cho development
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install các dependencies khác
pip install -r requirements.txt

# CompressAI (có thể cần build từ source)
pip install compressai
```

## 3. Lightweight Testing Setup
```bash
# Tạo sample data cho test nhanh (không cần full COCO)
mkdir test_data
# Copy vài images từ val2017 vào test_data/
```

## 4. Code Quality Tools
```bash
pip install black flake8 isort pytest
```

## 5. Git Configuration
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@domain.com"

# Setup gitignore cho ML project
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "runs/" >> .gitignore
echo "checkpoints/" >> .gitignore
echo "datasets/COCO*/" >> .gitignore
echo "*.pth" >> .gitignore
echo ".vscode/" >> .gitignore
``` 