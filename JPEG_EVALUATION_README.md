# JPEG/JPEG2000 BASELINE EVALUATION GUIDE

## 📋 TÓM TẮT SCRIPTS

### **1. Scripts chính:**
- `server_jpeg_evaluation.py` - Script Python đánh giá JPEG/JPEG2000 baseline
- `run_jpeg_evaluation.sh` - Shell script cho Linux/Mac
- `run_jpeg_evaluation.bat` - Batch file cho Windows
- `improved_jpeg_evaluation.py` - Script cải tiến (được tạo tự động)

### **2. Scripts hỗ trợ:**
- `test_codecs.py` - Test nhanh codec availability
- `install_codecs.py` - Cài đặt codec JPEG/JPEG2000 đầy đủ
- `fix_training_pipeline.py` - Diagnostic toàn diện dự án

---

## 🚀 CÁCH SỬ DỤNG

### **Bước 1: Test codec (bắt buộc)**
```bash
# Test nhanh
python3 test_codecs.py

# Nếu có lỗi, cài đặt codec đầy đủ
python3 install_codecs.py
```

### **Bước 2: Chạy evaluation**

#### **Linux/Mac:**
```bash
bash run_jpeg_evaluation.sh
```

#### **Windows:**
```bash
run_jpeg_evaluation.bat
```

#### **Python trực tiếp:**
```bash
python3 server_jpeg_evaluation.py \
    --data_dir datasets/COCO_Official \
    --max_images 100 \
    --quality_levels 10 20 30 40 50 60 70 80 90 95 \
    --output_dir results/jpeg_baseline
```

---

## 📊 KẾT QUẢ MONG ĐỢI

### **Output files:**
- `results/jpeg_baseline/jpeg_baseline_quick.csv` - Kết quả 50 images
- `results/jpeg_baseline/jpeg_baseline_full.csv` - Kết quả 200 images
- `results/jpeg_baseline/jpeg_baseline_stats.csv` - Thống kê tổng hợp

### **Metrics được tính:**
- **PSNR** (Peak Signal-to-Noise Ratio) - dB
- **SSIM** (Structural Similarity Index) - 0-1
- **BPP** (Bits Per Pixel) - compression ratio
- **File size** - bytes

### **Expected ranges:**
- **JPEG:** PSNR 20-40dB, SSIM 0.7-0.95, BPP 0.5-4.0
- **JPEG2000:** PSNR 22-42dB, SSIM 0.75-0.97, BPP 0.3-3.5

---

## 🔧 CODEC SUPPORT

### **JPEG Compression:**
- **Method:** OpenCV cv2.imwrite with JPEG_QUALITY
- **Quality levels:** 10, 20, 30, 40, 50, 60, 70, 80, 90, 95
- **Status:** ✅ Fully supported on all platforms

### **JPEG2000 Compression:**
- **Method 1:** OpenCV cv2.imwrite with JPEG2000_COMPRESSION_X1000 (preferred)
- **Method 2:** Pillow Image.save with JPEG2000 format (fallback)
- **Method 3:** ImageIO imwrite with JPEG2000 format (fallback)
- **Status:** ⚠️ Depends on system codec availability

### **Codec Installation:**
```bash
# Full OpenCV with contrib modules
pip install opencv-contrib-python

# Pillow with JPEG2000 support
pip install "Pillow[jpeg2000]"

# ImageIO with codecs
pip install imageio imageio-ffmpeg

# Linux: OpenJPEG library
sudo apt-get install libopenjp2-7-dev
```

---

## 🔍 TROUBLESHOOTING

### **Common Issues:**

#### **1. "OpenCV JPEG2000 not supported"**
```bash
# Solution: Install OpenCV with contrib
pip uninstall opencv-python
pip install opencv-contrib-python
```

#### **2. "Pillow JPEG2000 not supported"**
```bash
# Solution: Install with JPEG2000 support
pip install "Pillow[jpeg2000]"

# Linux: Install OpenJPEG library
sudo apt-get install libopenjp2-7-dev
```

#### **3. "Dataset not found"**
```bash
# Solution: Setup COCO dataset
python3 datasets/setup_coco_official.py

# Or check path
ls -la datasets/COCO_Official/val2017/
```

#### **4. "Python/pip not found"**
```bash
# Windows: Use full path
C:\Python39\python.exe test_codecs.py

# Linux: Install Python3
sudo apt-get install python3 python3-pip
```

### **Debugging Commands:**
```bash
# Test codec availability
python3 test_codecs.py

# Check OpenCV build info
python3 -c "import cv2; print(cv2.getBuildInformation())"

# Check Pillow features
python3 -c "from PIL import features; print(features.check('jpg_2000'))"

# Check dataset
ls -la datasets/COCO_Official/val2017/ | head -5
```

---

## 📈 PERFORMANCE COMPARISON

### **WAVENET-MV Target Performance:**
- **PSNR:** 28-38 dB
- **BPP:** 0.1-8.0
- **AI accuracy:** 85-95%

### **JPEG Baseline Performance:**
- **PSNR:** 20-40 dB
- **BPP:** 0.5-4.0
- **SSIM:** 0.7-0.95

### **JPEG2000 Baseline Performance:**
- **PSNR:** 22-42 dB
- **BPP:** 0.3-3.5
- **SSIM:** 0.75-0.97

**Goal:** WAVENET-MV should achieve competitive PSNR with lower BPP than traditional codecs.

---

## 🎯 SCRIPT ARGUMENTS

### **server_jpeg_evaluation.py:**
```bash
python3 server_jpeg_evaluation.py \
    --data_dir datasets/COCO_Official \           # Dataset directory
    --max_images 100 \                           # Number of images to process
    --quality_levels 10 20 30 40 50 60 70 80 90 95 \  # Quality levels
    --output_dir results/jpeg_baseline \         # Output directory
    --output_file jpeg_results.csv \             # Output CSV file
    --num_workers 4                              # Parallel workers
```

### **install_codecs.py:**
```bash
python3 install_codecs.py
# No arguments needed - fully automatic
```

### **test_codecs.py:**
```bash
python3 test_codecs.py
# No arguments needed - quick test
```

---

## 📄 OUTPUT FORMAT

### **CSV Columns:**
- `codec`: 'JPEG' or 'JPEG2000'
- `quality`: Quality level (10-95)
- `psnr`: PSNR value in dB
- `ssim`: SSIM value (0-1)
- `bpp`: Bits per pixel
- `file_size`: Compressed file size in bytes
- `image_path`: Path to source image

### **Example Results:**
```csv
codec,quality,psnr,ssim,bpp,file_size,image_path
JPEG,90,35.42,0.9234,2.1456,123456,datasets/COCO_Official/val2017/000001.jpg
JPEG2000,90,36.78,0.9345,1.8923,109876,datasets/COCO_Official/val2017/000001.jpg
```

---

## ✅ SUCCESS CRITERIA

### **Codec Test Success:**
- ✅ All imports work
- ✅ JPEG compression/decompression works
- ✅ At least one JPEG2000 method works
- ✅ Test images can be processed

### **Evaluation Success:**
- ✅ CSV files are generated
- ✅ Results contain expected metrics
- ✅ PSNR values are reasonable (>15 dB)
- ✅ BPP values are reasonable (<10)
- ✅ No errors during processing

### **Ready for WAVENET-MV Comparison:**
- ✅ Baseline results available
- ✅ Multiple quality levels tested
- ✅ Comprehensive metrics calculated
- ✅ Results formatted for easy comparison

---

## 📞 SUPPORT

Nếu gặp vấn đề:

1. **Chạy test codec trước:** `python3 test_codecs.py`
2. **Kiểm tra dataset:** `ls -la datasets/COCO_Official/val2017/`
3. **Cài đặt codec:** `python3 install_codecs.py`
4. **Chạy evaluation:** `bash run_jpeg_evaluation.sh` hoặc `run_jpeg_evaluation.bat`

**Mục tiêu:** Có được baseline JPEG/JPEG2000 results để so sánh với WAVENET-MV performance. 