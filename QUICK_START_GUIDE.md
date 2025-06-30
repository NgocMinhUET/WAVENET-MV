# 🚀 QUICK START GUIDE - WAVENET-MV
**Hướng dẫn setup từ đầu cho Windows + Ubuntu Server**

## 🎯 **Bước 1: Khởi Tạo Git Repository (Windows)**

### ⚠️ **Nếu bạn gặp lỗi: "fatal: not a git repository"**

Điều này có nghĩa là project chưa được khởi tạo Git. Hãy làm theo các bước sau:

#### 1.1 Chạy Script Khởi Tạo
```bash
# Trong thư mục dự án
init_git_repo.bat
```

#### 1.2 Hoặc Setup Thủ Công
```bash
# 1. Khởi tạo git repository
git init

# 2. Cấu hình user (nếu chưa có) -S
git config --global user.name "Your Name"
git config --global user.email "your.email@domain.com"

# 3. Tạo repository trên GitHub/GitLab
# Đi tới https://github.com → New Repository → Tạo "wavenet-mv"

# 4. Thêm remote repository
git remote add origin https://github.com/yourusername/wavenet-mv.git

# 5. Add và commit files
git add .
git commit -m "Initial commit: WAVENET-MV project setup"

# 6. Push lên remote
git push -u origin main
```

## 📋 **Bước 2: Setup Môi Trường Development (Windows)**

```bash
# 1. Tạo conda environment
conda create -n wavenet-dev python=3.9
conda activate wavenet-dev

# 2. Install PyTorch CPU (cho development)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install development tools
pip install black flake8 isort pytest
```

## 🐧 **Bước 3: Setup Ubuntu Server**

### 3.1 Tạo Repository trên GitHub/GitLab
- Đi tới GitHub/GitLab
- Tạo repository mới tên "wavenet-mv"
- Copy URL repository

### 3.2 Chạy Server Setup
```bash
# Trên Ubuntu server
chmod +x server_setup.sh
./server_setup.sh
# Nhập URL repository khi được hỏi
```

## 🔄 **Bước 4: Test Workflow**

### 4.1 Trên Windows
```bash
# 1. Activate environment
conda activate wavenet-dev

# 2. Test project
python quick_test.py

# 3. Make some changes và push
git_workflow.bat "Test initial setup"
```

### 4.2 Trên Ubuntu Server
```bash
# 1. Pull changes
cd wavenet-mv
git pull origin main

# 2. Activate environment
source wavenet-prod/bin/activate

# 3. Test training (với dummy data)
python training/stage1_train_wavelet.py --epochs 1 --test-mode
```

## 🆘 **Troubleshooting**

### ❌ **"fatal: not a git repository"**
**Giải pháp**: Chạy `init_git_repo.bat` hoặc `git init`

### ❌ **"git: command not found"**
**Giải pháp**: Install Git từ https://git-scm.com/download/win

### ❌ **"Permission denied (publickey)"**
**Giải pháp**: Setup SSH key hoặc dùng HTTPS URL

### ❌ **Import errors**
**Giải pháp**: 
```bash
conda activate wavenet-dev
pip install -r requirements.txt
```

## 📊 **Kiểm Tra Setup Thành Công**

### ✅ **Windows Development Ready**
```bash
conda activate wavenet-dev
python quick_test.py
# Nên thấy: "🎉 ALL TESTS PASSED!"
```

### ✅ **Git Workflow Ready**
```bash
git status
# Nên thấy: "On branch main" và "nothing to commit"
```

### ✅ **Server Training Ready**
```bash
# Trên Ubuntu
source wavenet-prod/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Nên thấy: "CUDA: True" (nếu có GPU)
```

## 🎯 **Workflow Hàng Ngày**

### 📝 **Development (Windows)**
1. `conda activate wavenet-dev`
2. Code với Cursor AI
3. `python quick_test.py`
4. `git_workflow.bat "Your changes"`

### 🚀 **Training (Ubuntu)**
1. `git pull origin main`
2. `source wavenet-prod/bin/activate`
3. `python training/stage1_train_wavelet.py`

---

## 🎊 **Hoàn Thành!**

Sau khi hoàn thành các bước trên, bạn sẽ có:
- ✅ Git repository hoạt động
- ✅ Windows development environment
- ✅ Ubuntu training server
- ✅ Automated workflow scripts

**🚀 Giờ bạn có thể bắt đầu phát triển WAVENET-MV!** 