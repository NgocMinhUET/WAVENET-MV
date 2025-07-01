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

## 🔄 **GIT WORKFLOW: DEVELOPMENT (Máy A) → SERVER (Máy B)**

### **Quy trình tổng quan**
```
Máy A (Windows - Development) → Git Repository → Máy B (Server - Training)
```

### **Bước 1: Phát triển trên Máy A (Windows)**

#### 1.1 Kiểm tra và commit thay đổi
```bash
# Kiểm tra status
git status

# Thêm file đã sửa
git add .

# Commit với message rõ ràng
git commit -m "Fix: [mô tả lỗi được sửa]"
# hoặc
git commit -m "Feature: [tính năng mới]"
# hoặc
git commit -m "Update: [cập nhật nào]"
```

#### 1.2 Push lên remote repository
```bash
git push origin master
```

### **Bước 2: Chạy trên Máy B (Server)**

#### 2.1 Clone repository (lần đầu)
```bash
git clone [YOUR_REPO_URL]
cd wavenet-mv
```

#### 2.2 Pull latest changes (các lần sau)
```bash
git pull origin master
```

#### 2.3 Setup environment (nếu cần)
```bash
# Install dependencies
pip install -r requirements.txt

# Setup datasets
./datasets/setup_coco.sh
./datasets/setup_davis.sh
```

#### 2.4 Chạy training
```bash
# Stage 1: Wavelet Training
python training/stage1_train_wavelet.py

# Stage 2: Compressor Training  
python training/stage2_train_compressor.py

# Stage 3: AI Heads Training
python training/stage3_train_ai.py
```

### **Bước 3: Xử lý lỗi (Error Handling)**

#### 3.1 Khi gặp lỗi trên Máy B
1. **Copy full error message** (bao gồm traceback)
2. **Copy command đã chạy**
3. **Note môi trường** (Python version, CUDA, etc)

#### 3.2 Debug trên Máy A
1. Paste error vào Cursor AI
2. Phân tích và sửa lỗi
3. Test local nếu có thể
4. Commit fix:
```bash
git add .
git commit -m "Fix: [mô tả lỗi cụ thể]"
git push origin master
```

#### 3.3 Quay lại Máy B
```bash
git pull origin master
# Chạy lại command bị lỗi
```

### **📋 Checklist trước khi Push**

**Trên Máy A:**
- [ ] Code không có syntax error
- [ ] Commit message rõ ràng
- [ ] Đã test cơ bản (nếu có thể)
- [ ] Cập nhật requirements.txt nếu thêm dependency

**Trên Máy B:**
- [ ] Pull latest changes
- [ ] Check Python environment
- [ ] Verify dataset paths
- [ ] Check disk space cho checkpoints

### **🔧 Useful Git Commands**

```bash
# Kiểm tra commit history
git log --oneline -10

# So sánh với remote
git fetch
git diff HEAD origin/master

# Rollback nếu cần
git reset --hard HEAD~1

# Tạo branch cho feature lớn
git checkout -b feature/new-architecture
git push -u origin feature/new-architecture
```

### **📊 Monitoring Commands cho Máy B**

```bash
# Check GPU usage
nvidia-smi

# Monitor training progress
tensorboard --logdir=./runs

# Check disk space
df -h

# Check running processes
ps aux | grep python
```

### **🚨 Common Issues & Solutions**

| Issue | Solution |
|-------|----------|
| `git pull` conflicts | `git stash` → `git pull` → `git stash pop` |
| CUDA out of memory | Reduce batch size in config |
| Dataset not found | Check paths in `dataset_loaders.py` |
| Permission denied | `chmod +x scripts/*.sh` |
| Python version mismatch | Use conda/venv with exact version |

### **📝 Error Reporting Template**

```
**Environment:**
- OS: [Ubuntu 20.04 / CentOS 7 / etc]
- Python: [3.8.x]
- PyTorch: [1.13.x]
- CUDA: [11.6]

**Command:**
```bash
[exact command that failed]
```

**Error:**
```
[full traceback]
```

**Additional Info:**
- Commit hash: [git rev-parse HEAD]
- Disk space: [df -h]
- GPU info: [nvidia-smi]
```

## 🚀 **COMPLETE 3-STAGE TRAINING PIPELINE**

### **🎯 Toàn Cảnh Workflow**
```
Stage 1: WaveletCNN (30 epochs) → Stage 2: Compressor (40 epochs) → Stage 3: AI Heads (50 epochs)
```

### **📋 Stage-by-Stage Commands**

#### **Stage 1: Wavelet Training** ✅
```bash
python training/stage1_train_wavelet.py \
    --dataset coco \
    --data_dir datasets/COCO_Official \
    --epochs 30 \
    --batch_size 8
```

#### **Stage 2: Compressor Training** ✅ (FIXED BUGS)
```bash
python training/stage2_train_compressor.py \
    --dataset coco \
    --data_dir datasets/COCO_Official \
    --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
    --lambda_rd 128 \
    --epochs 40 \
    --batch_size 8
```

#### **🆕 Stage 3: AI Heads Training** 
```bash
# Detection Only
python training/stage3_train_ai.py \
    --dataset coco \
    --data_dir datasets/COCO_Official \
    --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
    --stage2_checkpoint checkpoints/stage2_compressor_coco_lambda128_best.pth \
    --lambda_rd 128 \
    --enable_detection \
    --epochs 50 \
    --batch_size 4

# Segmentation Only  
python training/stage3_train_ai.py \
    --dataset coco \
    --data_dir datasets/COCO_Official \
    --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
    --stage2_checkpoint checkpoints/stage2_compressor_coco_lambda128_best.pth \
    --lambda_rd 128 \
    --enable_segmentation \
    --epochs 50 \
    --batch_size 4

# Both Tasks
python training/stage3_train_ai.py \
    --dataset coco \
    --data_dir datasets/COCO_Official \
    --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
    --stage2_checkpoint checkpoints/stage2_compressor_coco_lambda128_best.pth \
    --lambda_rd 128 \
    --enable_detection \
    --enable_segmentation \
    --epochs 50 \
    --batch_size 4
```

---

## 🎯 **ROADMAP SAU KHI HOÀN THÀNH STAGE 2**

### **Bước 1: Kiểm Tra Stage 2 Results** 
```bash
# Kiểm tra checkpoints
ls -la checkpoints/stage2_*

# Kiểm tra TensorBoard logs
tensorboard --logdir runs/stage2_*

# Expected files:
# - checkpoints/stage2_compressor_coco_lambda128_best.pth
# - checkpoints/stage2_compressor_coco_lambda128_latest.pth
```

### **Bước 2: Push Stage 2 Results (Windows)**
```bash
git add .
git commit -m "Complete Stage 2: Compressor training with MSE fixes
- MSE stable at 0.001-0.1 range (not collapsing)
- BPP in 1-10 range (proper calculation)  
- Ready for Stage 3 AI heads training"
git push origin master
```

### **Bước 3: Start Stage 3 Training (Server)**
```bash
# Pull latest changes
git pull origin master

# Start with detection task (easier to debug)
python training/stage3_train_ai.py \
    --dataset coco \
    --data_dir datasets/COCO_Official \
    --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
    --stage2_checkpoint checkpoints/stage2_compressor_coco_lambda128_best.pth \
    --lambda_rd 128 \
    --enable_detection \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-4
```

### **Bước 4: Monitor Stage 3 Training**
```bash
# TensorBoard monitoring
tensorboard --logdir runs/stage3_ai_heads_*

# Expected logs:
# - Train/TotalLoss
# - Train/DetectionLoss  
# - Train/SegmentationLoss
# - Val/TotalLoss
```

### **Bước 5: Evaluation & Comparison**
```bash
# Sau khi Stage 3 hoàn thành
python evaluation/codec_metrics.py \
    --model_path checkpoints/stage3_ai_heads_coco_best.pth \
    --dataset coco \
    --output_dir results/

# Compare với baselines
python evaluation/compare_baselines.py \
    --wavenet_path checkpoints/stage3_ai_heads_coco_best.pth \
    --output_csv results/comparison.csv
```

---

## 📊 **FULL PIPELINE ARCHITECTURE**

```
Input Image (3×256×256)
    ↓
Stage 1: WaveletCNN → Wavelet Coeffs (4×64×H×W) 
    ↓  
Stage 2a: AdaMixNet → Mixed Features (128×H/4×W/4)
    ↓
Stage 2b: CompressorVNVC → Compressed Features (128×H/4×W/4) + BPP
    ↓
Stage 3a: YOLO-tiny → Detection Boxes [x,y,w,h,conf,class]
Stage 3b: SegFormer → Segmentation Masks (21×H×W)
```

---

## 🎉 **SUCCESS CRITERIA**

### **Stage 2 Completion Indicators:**
- ✅ MSE: 0.001-0.1 (stable, không collapse)
- ✅ BPP: 1-10 (reasonable compression rate)
- ✅ Health check: "✅ MSE HEALTHY" + "✅ BALANCED"
- ✅ Debug: "✅ CompressorVNVC applying compression"

### **Stage 3 Success Indicators:**
- ✅ Detection Loss: Decreasing smoothly
- ✅ Segmentation Loss: Converging  
- ✅ Frozen pipeline: Compression features remain consistent
- ✅ GPU memory: Efficient usage với batch_size=4

### **Final Pipeline Success:**
- ✅ Full WAVENET-MV working end-to-end
- ✅ Competitive với JPEG/H.264 baselines
- ✅ Real-time inference capable
- ✅ Multiple task performance

--- 