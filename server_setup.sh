#!/bin/bash
# Ubuntu Server Setup Script for WAVENET-MV
# Run this once on your Ubuntu server

echo "🐧 Setting up WAVENET-MV on Ubuntu Server..."
echo "=============================================="

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y git python3 python3-pip python3-venv curl wget

# Install CUDA (if GPU available)
echo "🔍 Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    # Install CUDA toolkit
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda
else
    echo "⚠️ No GPU detected, will use CPU"
fi

# Create Python environment
echo "🐍 Setting up Python environment..."
python3 -m venv wavenet-prod
source wavenet-prod/bin/activate

# Install PyTorch
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    # GPU version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    # CPU version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Clone repository
echo "📥 Cloning repository..."
read -p "Enter your git repository URL: " REPO_URL
git clone $REPO_URL wavenet-mv
cd wavenet-mv

# Install dependencies
echo "📦 Installing project dependencies..."
pip install -r requirements.txt

# Setup COCO dataset
echo "💾 Setting up COCO dataset..."
python datasets/setup_coco_official.py

# Create necessary directories
mkdir -p runs checkpoints fig results

# Test installation
echo "🧪 Testing installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "✅ Server setup complete!"
echo "🚀 You can now run training scripts:"
echo "   python training/stage1_train_wavelet.py" 