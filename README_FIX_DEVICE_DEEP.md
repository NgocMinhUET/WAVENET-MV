# 🔧 Hướng dẫn sửa triệt để lỗi device mismatch

## Vấn đề

Khi chạy đánh giá (evaluation) trên server, bạn có thể gặp lỗi sau:

```
Error processing batch XXXX: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
```

Đây là lỗi **device mismatch** - một số tham số của mô hình đang ở CPU trong khi đầu vào đang ở CUDA.

## Nguyên nhân

Lỗi này xảy ra khi:
1. Checkpoint được lưu trên một thiết bị (ví dụ: CUDA) nhưng được tải trên thiết bị khác (ví dụ: CPU)
2. Một số module con trong mô hình không được chuyển đúng cách sang device mới
3. Phương thức `.to(device)` không được áp dụng đệ quy cho tất cả các tham số và buffers

## Giải pháp

Script `fix_device_deep.py` sẽ giúp bạn sửa lỗi này bằng cách:
1. Tải checkpoint
2. Khởi tạo lại các mô hình
3. Tải state_dict từ checkpoint
4. Đảm bảo tất cả các tham số và buffers đều ở cùng một device
5. Lưu checkpoint đã sửa

## Cách sử dụng

### Bước 1: Pull về các thay đổi mới nhất

```bash
git pull origin master
```

### Bước 2: Chạy script với checkpoint cần sửa

```bash
python fix_device_deep.py --checkpoint checkpoints/stage3_ai_heads_coco_best.pth
```

Các tham số:
- `--checkpoint`: Đường dẫn đến file checkpoint cần sửa (bắt buộc)
- `--device`: Device đích (cuda hoặc cpu, mặc định là cuda nếu có)
- `--mode`: Chế độ sửa (simple: chỉ sửa tensors, deep: sửa toàn bộ models, mặc định là deep)

### Bước 3: Sử dụng checkpoint đã sửa

Script sẽ tạo một checkpoint mới với hậu tố `_fixed_deep` (ví dụ: `stage3_ai_heads_coco_best_fixed_deep.pth`).

Sử dụng checkpoint đã sửa này để chạy đánh giá:

```bash
python evaluation/codec_metrics.py --checkpoint checkpoints/stage3_ai_heads_coco_best_fixed_deep.pth --dataset coco --data_dir datasets/COCO --split val --lambdas 128 --batch_size 1 --max_samples 10 --output_csv results/test_fixed.csv
```

## Chi tiết kỹ thuật

Script `fix_device_deep.py` thực hiện các bước sau:

1. **Tải checkpoint**: Đọc file checkpoint vào bộ nhớ
2. **Khởi tạo mô hình**: Tạo instances mới của WaveletTransformCNN, AdaMixNet và MultiLambdaCompressorVNVC
3. **Tải state_dict**: Áp dụng state_dict từ checkpoint vào các mô hình
4. **Sửa device mismatch**: Đảm bảo tất cả tham số và buffers đều ở cùng một device
   - Chuyển toàn bộ mô hình sang device đích bằng `.to(device)`
   - Kiểm tra từng module con và đảm bảo tất cả tham số đều ở đúng device
   - Kiểm tra và sửa tất cả buffers
5. **Lưu checkpoint mới**: Tạo checkpoint mới với state_dict đã được sửa

## Các lỗi thường gặp

### 1. ModuleNotFoundError

Nếu bạn gặp lỗi `ModuleNotFoundError`, hãy đảm bảo bạn đang chạy script từ thư mục gốc của dự án.

### 2. ImportError

Nếu bạn gặp lỗi `ImportError`, có thể do script không tìm thấy các module cần thiết. Hãy đảm bảo bạn đã cài đặt tất cả các dependencies:

```bash
pip install -r requirements.txt
```

### 3. RuntimeError khi tải checkpoint

Nếu bạn gặp lỗi `RuntimeError` khi tải checkpoint, có thể do phiên bản PyTorch không tương thích. Hãy đảm bảo bạn đang sử dụng PyTorch ≥1.13.

## Phương pháp thủ công

Nếu script không hoạt động, bạn có thể sửa lỗi thủ công bằng cách:

1. Thêm phương thức `.to(device)` vào các class mô hình:

```python
def to(self, device):
    super().to(device)
    if hasattr(self, 'module1'):
        self.module1.to(device)
    if hasattr(self, 'module2'):
        self.module2.to(device)
    # ... và các module khác
    return self
```

2. Khi tải mô hình, hãy đảm bảo gọi `.to(device)` sau khi tải state_dict:

```python
model = YourModel()
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)  # Chuyển toàn bộ mô hình sang device
```

## Liên hệ hỗ trợ

Nếu bạn vẫn gặp vấn đề, hãy tạo issue trên GitHub hoặc liên hệ với nhóm phát triển.

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