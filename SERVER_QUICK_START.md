# WAVENET-MV SERVER QUICK START GUIDE

## 🚨 VẤN ĐỀ ĐƯỢC PHÁT HIỆN

Sau khi phân tích toàn diện dự án, các vấn đề chính đã được xác định:

### ❌ **Vấn đề nghiêm trọng:**
1. **Checkpoints folder rỗng** - Training chưa bao giờ hoàn thành thành công
2. **Python environment issues** - Python không được tìm thấy đúng cách
3. **Dataset loading problems** - Annotation files có thể không tồn tại
4. **Training pipeline errors** - Có nhiều lỗi tiềm ẩn trong code

### ⚠️ **Kết quả hiện tại là FAKE DATA** [[memory:645487]]
- Tất cả evaluation results đều là mock data
- Chưa có model nào được train thật sự
- Training pipeline framework hoàn thiện nhưng chưa execute

---

## 🔧 GIẢI PHÁP KHẮC PHỤC

### **Bước 1: Kiểm tra và sửa chữa hệ thống**

```bash
# Chạy diagnostic script để kiểm tra tất cả vấn đề
python3 fix_training_pipeline.py

# Kiểm tra training pipeline
python3 verify_training_pipeline.py
```

### **Bước 2: Kiểm tra và cài đặt codec JPEG/JPEG2000**

```bash
# Test codec availability (nhanh)
python3 test_codecs.py

# Cài đặt codec đầy đủ (nếu cần)
python3 install_codecs.py
```

### **Bước 3: Đánh giá JPEG/JPEG2000 baseline (như yêu cầu)**

```bash
# Linux/Mac: Chạy đánh giá baseline JPEG/JPEG2000
bash run_jpeg_evaluation.sh

# Windows: Chạy batch file
run_jpeg_evaluation.bat

# Hoặc chạy trực tiếp Python script
python3 server_jpeg_evaluation.py \
    --data_dir datasets/COCO_Official \
    --max_images 100 \
    --output_dir results/jpeg_baseline
```

### **Bước 3: Sửa chữa environment issues**

```bash
# Kiểm tra Python environment
python3 --version
pip3 --version

# Cài đặt missing dependencies
pip3 install -r requirements.txt

# Nếu không có pip3, dùng:
python -m pip install -r requirements.txt
```

### **Bước 4: Setup dataset đúng cách**

```bash
# Kiểm tra dataset
ls -la datasets/COCO_Official/val2017/
ls -la datasets/COCO_Official/annotations/

# Nếu missing, setup lại:
python3 datasets/setup_coco_official.py
```

### **Bước 5: Chạy training thật sự (sau khi fix)**

```bash
# Sau khi fix tất cả issues
bash server_training.sh

# Monitor training progress
tail -f runs/*/events.out.tfevents.*
```

---

## 📊 EXPECTED RESULTS vs CURRENT STATUS

### **Current Status (FAKE DATA):**
- PSNR: 6.87 dB (fake)
- BPP: 48.0 (fake)
- AI accuracy: 50% (fake)

### **Expected Results (after fixing):**
- PSNR: 28-38 dB
- BPP: 0.1-8.0
- AI accuracy: 85-95%
- Quantization: 20-80% non-zero

### **JPEG/JPEG2000 Baseline (for comparison):**
- PSNR: 20-40 dB (depending on quality)
- BPP: 0.5-4.0
- SSIM: 0.7-0.95

---

## 🎯 IMMEDIATE ACTIONS REQUIRED

### **1. Diagnostics First**
```bash
python3 fix_training_pipeline.py
```

### **2. Run JPEG Baseline**
```bash
bash run_jpeg_evaluation.sh
```

### **3. Fix Environment**
- Install proper Python environment
- Fix missing dependencies
- Ensure CUDA is available

### **4. Real Training**
```bash
# Only after all fixes
bash server_training.sh
```

---

## 🔍 DEBUGGING COMMANDS

### **Check Environment:**
```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python3 -c "import compressai; print('CompressAI OK')"

# Test codec availability
python3 test_codecs.py
```

### **Check Dataset:**
```bash
ls -la datasets/COCO_Official/val2017/ | head -10
python3 -c "from datasets.dataset_loaders import COCODatasetLoader; print('Dataset loader OK')"
```

### **Check Models:**
```bash
python3 -c "from models.wavelet_transform_cnn import WaveletTransformCNN; print('Wavelet OK')"
python3 -c "from models.compressor_vnvc import CompressorVNVC; print('Compressor OK')"
```

---

## 📈 MONITORING TRAINING

### **Check Progress:**
```bash
# Monitor checkpoints
watch -n 5 'ls -la checkpoints/'

# Monitor TensorBoard logs
tensorboard --logdir runs/

# Monitor GPU usage
nvidia-smi -l 1
```

### **Validate Results:**
```bash
# After training, run real evaluation
python3 evaluation/codec_metrics_final.py \
    --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
    --stage2_checkpoint checkpoints/stage2_compressor_coco_lambda256_best.pth \
    --stage3_checkpoint checkpoints/stage3_ai_heads_coco_best.pth
```

---

## 🚀 SUCCESS CRITERIA

### **✅ Training Pipeline Fixed When:**
- All scripts run without errors
- Checkpoints are created in checkpoints/ folder
- TensorBoard logs show actual training progress
- Real evaluation results (not fake data)

### **✅ Good Results When:**
- PSNR > 25 dB
- BPP < 2.0
- AI accuracy > 80%
- Non-zero quantization > 50%

---

## 💡 TROUBLESHOOTING

### **Common Issues:**
1. **"Python not found"** → Use `python3` or set PATH
2. **"Module not found"** → Install missing packages
3. **"CUDA error"** → Check GPU availability
4. **"Dataset not found"** → Run setup_coco_official.py
5. **"No checkpoints"** → Training never completed

### **Quick Fixes:**
```bash
# Fix Python path
export PATH=/usr/bin:$PATH

# Fix CUDA
export CUDA_VISIBLE_DEVICES=0

# Fix dataset
python3 datasets/setup_coco_official.py

# Clean restart
rm -rf checkpoints/* runs/* results/*
```

---

## 📞 SUPPORT

Nếu vẫn gặp vấn đề, hãy:

1. **Chạy diagnostic script** và gửi kết quả
2. **Kiểm tra log files** trong runs/
3. **Verify dataset** có đúng không
4. **Check environment** với các lệnh trên

**Mục tiêu:** Có được training pipeline hoạt động đúng và results thật sự, không phải fake data. 