# Hướng Dẫn Chạy Lại Evaluation Trên Server

## 🚀 Các Fix Đã Thực Hiện

### 1. **VCM Evaluation Collate Function**
- ✅ Thêm `vcm_collate_fn` để xử lý tensor có kích thước khác nhau
- ✅ Tự động pad boxes, labels, areas về cùng kích thước
- ✅ Tránh lỗi `RuntimeError: stack expects each tensor to be equal size`

### 2. **Column Name Fixes**
- ✅ Sửa column name từ `'psnr'` thành `'psnr_db'` trong các script
- ✅ Tự động detect column names: `psnr_db`, `bpp`, `ms_ssim`
- ✅ Tránh lỗi `KeyError: 'psnr'` trong generate_paper_results.py

### 3. **Import Fix**
- ✅ Sửa circular import trong `evaluate_vcm.py`
- ✅ Import trực tiếp `VCMEvaluator` thay vì `main` function

### 4. **AI Heads Input Channels Fix** ⭐ NEW
- ✅ Sửa input_channels từ 192 → 128 cho YOLO và SegFormer heads
- ✅ Đảm bảo nhất quán giữa training và evaluation
- ✅ Dùng synthesis_transform output (x_hat) thay vì analysis_transform output
- ✅ Tránh lỗi `RuntimeError: expected input[8, 128, 256, 256] to have 192 channels`

### 5. **YOLO Anchor Mismatch Fix** ⭐ NEW
- ✅ Sửa số lượng anchors từ 9 → 3 cho YOLO head
- ✅ Thêm `anchor_indices=[0,1,2]` để chỉ dùng 3 anchor đầu tiên
- ✅ Tránh lỗi `The size of tensor a (3) must match the size of tensor b (9)`

### 6. **JSON Serialization Fix** ⭐ NEW
- ✅ Convert tất cả PyTorch tensors thành Python native types (float, int)
- ✅ Sửa `evaluate_detection()` và `evaluate_segmentation()` để return Python types
- ✅ Tránh lỗi `Object of type Tensor is not JSON serializable`

## 📋 Quy Trình Chạy Lại Trên Server

### Bước 1: Pull Code Mới
```bash
cd /work/u9564043/Minh/Thesis/week_propose/p1/WAVENET-MV
git pull origin master
```

### Bước 2: Kiểm Tra Fixes
```bash
# Test collate function và column names
python test_vcm_fixes.py
```

### Bước 3: Chạy Complete Evaluation
```bash
python run_complete_evaluation.py \
    --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
    --stage2_checkpoint checkpoints/stage2_compressor_coco_lambda128_best.pth \
    --stage3_checkpoint checkpoints/stage3_ai_heads_best.pth \
    --dataset coco \
    --data_dir datasets/COCO \
    --max_samples 1000
```

### Bước 4: Chạy Từng Bước Riêng Lẻ (Nếu Cần)

#### VCM Evaluation
```bash
python evaluate_vcm.py \
    --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
    --stage2_checkpoint checkpoints/stage2_compressor_coco_lambda128_best.pth \
    --stage3_checkpoint checkpoints/stage3_ai_heads_best.pth \
    --dataset coco \
    --data_dir datasets/COCO \
    --enable_detection \
    --enable_segmentation \
    --batch_size 8 \
    --max_samples 1000 \
    --output_json results/vcm_results.json
```

#### Codec Metrics
```bash
python evaluation/codec_metrics_final.py \
    --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
    --stage2_checkpoint checkpoints/stage2_compressor_coco_lambda128_best.pth \
    --dataset coco \
    --data_dir datasets/COCO \
    --split val \
    --batch_size 8 \
    --max_samples 1000 \
    --lambdas 64 128 256 512 1024 \
    --output_csv results/codec_metrics_final.csv
```

#### Baseline Comparison
```bash
python evaluation/compare_baselines.py \
    --dataset coco \
    --data_dir datasets/COCO \
    --split val \
    --max_samples 1000 \
    --methods JPEG WebP PNG \
    --qualities 10 30 50 70 90 \
    --output_csv results/baseline_comparison.csv
```

## 🔧 Troubleshooting

### Lỗi OpenMP
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Lỗi CUDA Memory
```bash
# Giảm batch size
--batch_size 4

# Hoặc clear cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Lỗi Checkpoint Không Tồn Tại
- Script sẽ tự động dùng random weights và thông báo
- Kiểm tra đường dẫn checkpoint trong `checkpoints/`

### Lỗi Dataset
- Đảm bảo COCO dataset đã được setup: `datasets/COCO/`
- Kiểm tra annotations: `datasets/COCO/annotations/instances_val2017.json`

### Lỗi Channels Mismatch
- ✅ Đã fix: AI heads dùng 128 channels (synthesis output)
- ✅ Training và evaluation nhất quán
- ✅ Không còn lỗi `expected input to have 192 channels`

### Lỗi JSON Serialization
- ✅ Đã fix: Convert tensors thành Python native types
- ✅ Không còn lỗi `Object of type Tensor is not JSON serializable`

## 📊 Expected Results

### VCM Evaluation
- ✅ Không còn lỗi collate function
- ✅ Không còn lỗi channels mismatch
- ✅ Không còn lỗi anchor mismatch
- ✅ Không còn lỗi JSON serialization
- ✅ Detection và segmentation chạy được
- ✅ Output JSON: `results/vcm_results.json`

### Codec Metrics
- ✅ PSNR, MS-SSIM, BPP được tính đúng
- ✅ Output CSV: `results/codec_metrics_final.csv`
- ✅ Columns: `lambda`, `psnr_db`, `ms_ssim`, `bpp`

### Paper Generation
- ✅ Rate-distortion curves: `fig/rate_distortion_curves.pdf`
- ✅ Tables: `tables/codec_table.tex`
- ✅ Summary report: `results/evaluation_summary.md`

## 🎯 Success Criteria

1. ✅ VCM evaluation chạy không lỗi collate
2. ✅ VCM evaluation chạy không lỗi channels mismatch
3. ✅ VCM evaluation chạy không lỗi anchor mismatch
4. ✅ VCM evaluation chạy không lỗi JSON serialization
5. ✅ Codec metrics tính đúng PSNR/BPP/MS-SSIM
6. ✅ Paper figures và tables được generate
7. ✅ Complete evaluation pipeline hoàn thành

## 📞 Support

Nếu gặp lỗi:
1. Copy full error message
2. Paste vào Cursor AI trên máy Windows
3. Commit fix và push
4. Pull lại trên server

**Good luck! 🚀**