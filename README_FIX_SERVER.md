# 🔧 Hướng dẫn sửa lỗi device mismatch trên server

## Lỗi hiện tại
```
Error processing batch X: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
```

## Nguyên nhân
- Lỗi xảy ra do mismatch giữa tensors trên CUDA và trên CPU
- Các module con trong compressor cải tiến không được chuyển đến cùng device

## Các bước sửa lỗi trên server

### Bước 1: Pull code mới nhất
```bash
cd wavenet-mv  # hoặc thư mục project trên server
git pull origin master
```

### Bước 2: Chạy script sửa lỗi
```bash
python fix_device_mismatch.py
```

Script này sẽ:
- Thêm phương thức `to(device)` cho `ImprovedCompressorVNVC`
- Thêm phương thức `to(device)` cho `ImprovedMultiLambdaCompressorVNVC`
- Sửa `codec_metrics.py` để đảm bảo `.to(device)` được gọi sau khi khởi tạo compressor

### Bước 3: Tích hợp compressor cải tiến (nếu cần)
```bash
python integrate_improved_compressor.py
```

### Bước 4: Chạy đánh giá với số lượng mẫu nhỏ để test
```bash
python evaluation/codec_metrics.py \
    --checkpoint checkpoints/stage3_ai_heads_coco_best.pth \
    --dataset coco \
    --data_dir datasets/COCO \
    --split val \
    --lambdas 128 \
    --batch_size 4 \
    --max_samples 20 \
    --output_csv results/test_fixed.csv
```

### Bước 5: Nếu không còn lỗi, chạy đánh giá đầy đủ
```bash
python generate_paper_results.py \
    --checkpoint checkpoints/stage3_ai_heads_coco_best.pth \
    --dataset coco \
    --data_dir datasets/COCO \
    --split val \
    --max_samples 500 \
    --batch_size 4
```

## Xác nhận kết quả
Sau khi sửa lỗi, kết quả sẽ hiển thị:
- PSNR > 0 dB
- MS-SSIM > 0
- BPP > 0

## Báo cáo vấn đề
Nếu vẫn gặp lỗi, vui lòng chụp ảnh màn hình đầy đủ và gửi lại để được hỗ trợ. 