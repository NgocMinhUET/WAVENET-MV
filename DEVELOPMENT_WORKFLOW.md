# 🔄 WAVENET-MV Development Workflow
**Windows Development → Ubuntu Server Training**

## 📋 Quy Trình Làm Việc Hiệu Quả

### 🪟 **Phase 1: Development trên Windows**

#### 1.1 Setup Ban Đầu (Chỉ làm 1 lần)
```bash
# Chạy setup
conda create -n wavenet-dev python=3.9
conda activate wavenet-dev
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install black flake8 isort pytest
```

#### 1.2 Quy Trình Hàng Ngày
```bash
# 1. Kích hoạt environment
conda activate wavenet-dev

# 2. Sử dụng Cursor AI để sửa code
# - Mở Cursor, chỉnh sửa code
# - Sử dụng AI để optimize, fix bugs

# 3. Test nhanh trước khi push
python quick_test.py

# 4. Push code lên git
git_workflow.bat "Describe your changes"
```

### 🐧 **Phase 2: Training trên Ubuntu Server**

#### 2.1 Setup Server (Chỉ làm 1 lần)
```bash
# Trên Ubuntu server
chmod +x server_setup.sh
./server_setup.sh
```

#### 2.2 Quy Trình Training
```bash
# 1. Pull code mới nhất
cd wavenet-mv
git pull origin master

# 2. Activate environment
source wavenet-prod/bin/activate

# 3. Chạy training
python training/stage1_train_wavelet.py
python training/stage2_train_compressor.py
python training/stage3_train_ai.py

# 4. Monitor kết quả
tensorboard --logdir=runs
```

## 🎯 Workflow Tối Ưu

### 🔄 **Chu Trình Phát Triển**
```
Windows (Development) → Git → Ubuntu (Training) → Results → Windows (Analysis)
```

### 📊 **Phân Chia Nhiệm Vụ**

| Công Việc | Nơi Thực Hiện | Lý Do |
|-----------|---------------|-------|
| Code development | Windows + Cursor AI | UI tốt, AI assistant |
| Quick testing | Windows (CPU) | Kiểm tra syntax nhanh |
| Model training | Ubuntu Server (GPU) | Tài nguyên tính toán |
| Result analysis | Windows | Visualization, charts |

### 🚀 **Commands Nhanh**

#### Trên Windows:
```bash
# Test + Push trong 1 lệnh
git_workflow.bat "Add new feature"

# Chỉ test
python quick_test.py

# Format code
black . && isort .
```

#### Trên Ubuntu:
```bash
# Quick pull + train
git pull origin master && python training/stage1_train_wavelet.py

# Monitor training
htop  # System resources
nvidia-smi  # GPU usage
tensorboard --logdir=runs --port=6006
```

## 💡 **Best Practices**

### ✅ **DOs**
- ✅ **Luôn test trước khi push**: `python quick_test.py`
- ✅ **Commit thường xuyên**: Những thay đổi nhỏ, dễ track
- ✅ **Sử dụng meaningful commit messages**
- ✅ **Backup checkpoints**: Sao lưu model weights quan trọng
- ✅ **Monitor training**: Dùng TensorBoard
- ✅ **Version control datasets**: Ghi lại dataset version

### ❌ **DON'Ts**
- ❌ **Không push code chưa test**
- ❌ **Không commit file lớn** (datasets, checkpoints)
- ❌ **Không hard-code paths**: Dùng relative paths
- ❌ **Không train trên Windows**: Waste resources
- ❌ **Không ignore model weights**: Thêm vào .gitignore

## 🔧 **Troubleshooting**

### ⚠️ **Lỗi Thường Gặp**

#### Windows:
```bash
# Lỗi import
pip install -r requirements.txt

# Lỗi torch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Lỗi git
git config --global user.name "Your Name"
```

#### Ubuntu:
```bash
# Lỗi CUDA
nvidia-smi  # Check GPU
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Lỗi memory
export CUDA_VISIBLE_DEVICES=0  # Limit GPU
```

## 📈 **Monitoring Training**

### 📊 **TensorBoard**
```bash
# Trên server
tensorboard --logdir=runs --host=0.0.0.0 --port=6006

# Trên Windows (SSH tunnel)
ssh -L 6006:localhost:6006 user@server-ip
# Mở http://localhost:6006
```

### 📋 **Log Files**
```bash
# Training logs
tail -f runs/stage1_*/events.out.tfevents.*

# System logs
tail -f /var/log/syslog
```

## 🎯 **Kết Quả Mong Đợi**

Sau khi hoàn thành workflow này, bạn sẽ có:
- ✅ **Môi trường development mượt mà** trên Windows
- ✅ **Quy trình test tự động** trước khi push
- ✅ **Server training ổn định** trên Ubuntu
- ✅ **Git workflow hiệu quả**
- ✅ **Monitoring system** đầy đủ

---
*💡 Tip: Bookmark file này và follow từng bước cho lần đầu setup!* 