# COCO DATASET SETUP GUIDE - WAVENET-MV

## 🎯 **OFFICIAL COCO DATASET INTEGRATION**

Dựa trên [COCO official website](https://cocodataset.org/#download) và [Ultralytics documentation](https://docs.ultralytics.com/datasets/detect/coco/), guide này sẽ giúp bạn download và setup COCO dataset chính thức cho WAVENET-MV framework.

---

## 📊 **COCO DATASET OVERVIEW**

### **Dataset Statistics**
- **Total Images**: 330K images
- **Annotated Images**: 200K images với detailed annotations
- **Object Categories**: 80 classes (person, car, bicycle, etc.)
- **Tasks**: Detection, Segmentation, Keypoints, Captioning

### **Dataset Splits**
| Split | Images | Size | Usage |
|-------|--------|------|-------|
| **Train2017** | 118K | ~19GB | Training models |
| **Val2017** | 5K | ~1GB | Validation during training |
| **Test2017** | 20K | ~7GB | Final evaluation |
| **Annotations** | - | ~241MB | Object detection annotations |

---

## 🚀 **QUICK SETUP (RECOMMENDED)**

### **Option 1: Minimal Setup (Testing)**
```bash
# Download validation set + annotations (~1.2GB)
python datasets/setup_coco_official.py --minimal

# Result: datasets/COCO/ với val2017 images + annotations
```

### **Option 2: Full Setup (Production)**  
```bash
# Download complete dataset (~46GB)
python datasets/setup_coco_official.py --full

# Result: Complete COCO 2017 dataset
```

### **Option 3: Custom Setup**
```bash
# Download specific subsets
python datasets/setup_coco_official.py --datasets val2017 train2017 annotations_trainval2017

# Custom directory
python datasets/setup_coco_official.py --minimal --dir /path/to/my/coco
```

---

## 📁 **EXPECTED DIRECTORY STRUCTURE**

Sau khi setup thành công, bạn sẽ có cấu trúc:

```
datasets/COCO/
├── images/
│   ├── train2017/        # 118,287 training images
│   │   ├── 000000000009.jpg
│   │   ├── 000000000025.jpg
│   │   └── ...
│   ├── val2017/          # 5,000 validation images  
│   │   ├── 000000000139.jpg
│   │   ├── 000000000285.jpg
│   │   └── ...
│   └── test2017/         # 40,670 test images (optional)
│       └── ...
├── annotations/
│   ├── instances_train2017.json    # Training annotations
│   ├── instances_val2017.json      # Validation annotations
│   ├── person_keypoints_train2017.json
│   ├── person_keypoints_val2017.json
│   ├── captions_train2017.json
│   └── captions_val2017.json
├── labels/               # YOLO format (optional)
│   ├── train2017/
│   └── val2017/
└── dataset_info.json     # Setup information
```

---

## 🔧 **WAVENET-MV INTEGRATION**

### **COCODatasetLoader Compatibility**
COCO dataset structure hoàn toàn compatible với COCODatasetLoader:

```python
from datasets.dataset_loaders import COCODatasetLoader

# Load validation set
dataset = COCODatasetLoader(
    data_dir='datasets/COCO',  # Path to downloaded COCO
    subset='val',              # Use val2017
    image_size=256,            # Resize for WAVENET-MV
    task='detection',          # Object detection
    augmentation=True          # Data augmentation
)

print(f"Dataset length: {len(dataset)}")  # Should show 5,000 for val2017
sample = dataset[0]
print(f"Sample keys: {list(sample.keys())}")  # ['image', 'boxes', 'labels']
```

### **Training Integration**
```python
# Stage 1: Wavelet Training
# Uses COCO images for reconstruction loss
python training/stage1_train_wavelet.py

# Stage 2: Compression Training  
# Uses COCO images for rate-distortion optimization
python training/stage2_train_compressor.py

# Stage 3: AI Heads Training
# Uses COCO images + annotations for detection/segmentation
python training/stage3_train_ai.py
```

---

## 🌐 **DOWNLOAD METHODS**

### **Method 1: Python Script (Recommended)**
```bash
# Use official setup script
python datasets/setup_coco_official.py --minimal
```

### **Method 2: Manual Download**
Theo [COCO website](https://cocodataset.org/#download):

```bash
# Create directories
mkdir -p datasets/COCO/images
cd datasets/COCO

# Download images
wget http://images.cocodataset.org/zips/val2017.zip     # 1GB
wget http://images.cocodataset.org/zips/train2017.zip   # 19GB (optional)

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip  # 241MB

# Extract
unzip val2017.zip -d images/
unzip annotations_trainval2017.zip
```

### **Method 3: Using Ultralytics**
```python
from ultralytics.utils.downloads import download

# Download using Ultralytics helper
urls = [
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
]
download(urls, dir="datasets/COCO", threads=2)
```

---

## ✅ **VERIFICATION CHECKLIST**

### **After Download, Verify:**
```bash
# Verify dataset structure
python datasets/setup_coco_official.py --verify-only --dir datasets/COCO
```

**Expected Output:**
```
🔍 Verifying COCO dataset structure...
✅ images/val2017 (5,000 images)
✅ annotations/instances_val2017.json (25MB)
✅ annotations/instances_train2017.json (123MB) 
🎉 COCO dataset structure verified successfully!
```

### **Test DataLoader:**
```bash
# Test COCO loading
python -c "
from datasets.dataset_loaders import COCODatasetLoader
dataset = COCODatasetLoader(data_dir='datasets/COCO', subset='val')
print(f'✅ Loaded {len(dataset)} samples')
sample = dataset[0]
print(f'✅ Sample shape: {sample[\"image\"].shape}')
print(f'✅ Boxes shape: {sample[\"boxes\"].shape}')
print(f'✅ Labels shape: {sample[\"labels\"].shape}')
"
```

---

## 🎨 **COCO CATEGORIES (80 Classes)**

COCO dataset bao gồm 80 object categories:

```python
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
```

---

## ⚡ **PERFORMANCE TIPS**

### **Fast Download:**
```bash
# Parallel downloads (nếu có bandwidth)
# Method from community: https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9
wget -c http://images.cocodataset.org/zips/val2017.zip &
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip &
wait
```

### **Space Optimization:**
```bash
# Download + extract + cleanup in one go
python datasets/setup_coco_official.py --minimal  # Auto cleanup ZIP files
```

### **Resume Downloads:**
```bash
# If download interrupted
python datasets/setup_coco_official.py --minimal --force  # Re-download
```

---

## 🔧 **TROUBLESHOOTING**

### **Common Issues:**

#### **Issue 1: Slow Download**
```bash
# Solution: Use parallel downloads hoặc different mirror
# Try: aria2c for faster downloads
aria2c -x 4 http://images.cocodataset.org/zips/val2017.zip
```

#### **Issue 2: Space Issues**
```bash
# Check available space
df -h .
# Minimal setup chỉ cần ~1.5GB free space
python datasets/setup_coco_official.py --minimal
```

#### **Issue 3: Network Issues**
```bash
# Resume interrupted downloads
python datasets/setup_coco_official.py --minimal --force
```

#### **Issue 4: Permission Issues**
```bash
# Fix permissions
chmod -R 755 datasets/COCO/
```

---

## 📊 **TRAINING RECOMMENDATIONS**

### **For WAVENET-MV Training:**

1. **Development/Testing**: Use `--minimal` (val2017 only)
   - Fast download (~1.2GB)
   - Good for debugging và initial testing
   - 5,000 validation images

2. **Full Training**: Use `--full` (train+val+test)
   - Complete dataset (~46GB)  
   - Production-ready training
   - 118,000+ training images

3. **Custom Training**: Mix và match
   - Training: Use train2017 (118K images)
   - Validation: Use val2017 (5K images)
   - Testing: Use test2017 (40K images)

### **WAVENET-MV Stages:**
- **Stage 1** (Wavelet): Images only (no annotations needed)
- **Stage 2** (Compression): Images only (optional annotations)  
- **Stage 3** (AI Heads): Images + annotations required

---

## 📚 **REFERENCES**

- **COCO Official**: https://cocodataset.org/#download
- **COCO Paper**: [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)
- **Ultralytics COCO**: https://docs.ultralytics.com/datasets/detect/coco/
- **Community Scripts**: https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9

---

## 🎯 **NEXT STEPS**

1. **Download Dataset**: `python datasets/setup_coco_official.py --minimal`
2. **Verify Setup**: `python datasets/setup_coco_official.py --verify-only`
3. **Test Loading**: Test COCODatasetLoader compatibility
4. **Start Training**: Begin Stage 1 WAVENET-MV training
5. **Scale Up**: Download full dataset when ready for production

**Ready to revolutionize video compression với WAVENET-MV + COCO! 🚀** 