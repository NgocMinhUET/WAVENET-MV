# 🔧 Hướng dẫn sửa triệt để lỗi device mismatch

## Vấn đề
```
Error processing batch X: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
```

Lỗi này xảy ra khi có module trong pipeline không được chuyển đến cùng device (CUDA) đúng cách.

## Hướng dẫn sửa trên server

### Bước 1: Pull code mới nhất từ git
```bash
git pull origin master
```

### Bước 2: Chạy script sửa triệt để lỗi device mismatch
```bash
python fix_device_deep.py
```

Script sẽ thực hiện các thao tác sau:
- Thêm phương thức `to(device)` cho tất cả các module chính
- Sửa lại file evaluation/codec_metrics.py để kiểm tra device của từng module
- Đảm bảo mọi thành phần con của các module đều được chuyển đến cùng device

### Bước 3: Chạy đánh giá với batch_size=1 để test
```bash
python evaluation/codec_metrics.py \
    --checkpoint checkpoints/stage3_ai_heads_coco_best.pth \
    --dataset coco \
    --data_dir datasets/COCO \
    --split val \
    --lambdas 128 \
    --batch_size 1 \
    --max_samples 10 \
    --output_csv results/test_fixed.csv
```

Nếu thành công (không có lỗi device mismatch), kết quả sẽ có:
- PSNR > 0 dB
- MS-SSIM > 0
- BPP > 0

### Bước 4: Chạy đánh giá đầy đủ
```bash
python generate_paper_results.py \
    --checkpoint checkpoints/stage3_ai_heads_coco_best.pth \
    --dataset coco \
    --data_dir datasets/COCO \
    --split val \
    --max_samples 500 \
    --batch_size 4
```

## Nếu vẫn gặp lỗi?

Nếu vẫn gặp lỗi device mismatch, hãy thực hiện các bước sau:

1. Chạy với `--batch_size 1` để đơn giản hóa quá trình debug
2. Kiểm tra output của `fix_device_deep.py` để xem có module nào không được sửa
3. Kiểm tra terminal output để xem device của từng module được in ra
4. Tìm module cụ thể gặp lỗi và kiểm tra liệu nó có phải module được tạo động

## Khắc phục thủ công

Nếu script tự động không giải quyết được, có thể cần phải can thiệp thủ công:

```python
# Kiểm tra device của các thành phần
print(f"Wavelet CNN device: {next(model.wavelet_cnn.parameters()).device}")
print(f"AdaMixNet device: {next(model.adamixnet.parameters()).device}")
print(f"Compressor device: {next(model.compressor.parameters()).device}")
```

Sau đó chuyển thủ công các module chưa được chuyển đúng sang CUDA.

---

💡 **Lưu ý**: Batch size nhỏ hơn (1-2) và max_samples nhỏ hơn (10-20) giúp dễ debug hơn và tránh lỗi CUDA out of memory. 