# WAVENET-MV: Wavelet-based Video Compression for Machine Vision

Framework nén video thông minh kết hợp Wavelet Transform và Adaptive Mixing Network (AdaMixNet) để optimize cho các tác vụ AI trên datasets COCO 2017 và DAVIS 2017.

## 🚀 Tính năng chính

- **Wavelet Transform CNN**: Lifting-based wavelet decomposition với PredictCNN và UpdateCNN
- **AdaMixNet**: Adaptive mixing với 4 parallel filters và attention mechanism
- **Compressor VNVC**: Quantization và entropy coding với CompressAI GaussianConditional
- **AI Heads**: YOLO-tiny detection và SegFormer-lite segmentation trên compressed features
- **3-Stage Training**: Pre-training → Rate-Distortion → Task-specific training
- **Multi-Lambda Support**: λ ∈ {256, 512, 1024} cho different compression rates

## 📋 Requirements

```bash
# Cài đặt dependencies
pip install -r requirements.txt
```

### Core Dependencies
- PyTorch ≥ 1.13.0
- TorchVision ≥ 0.14.0
- CompressAI ≥ 1.2.0
- OpenCV, NumPy, SciPy
- TensorBoard, Matplotlib, Seaborn
- COCO API, Albumentations

## 🏗️ Cấu trúc dự án

```
WAVENET-MV/
├── models/                          # Core models
│   ├── wavelet_transform_cnn.py     # WaveletTransformCNN với PredictCNN/UpdateCNN
│   ├── adamixnet.py                 # AdaMixNet với 4 parallel filters
│   ├── compressor_vnvc.py           # Compressor với quantization + entropy
│   └── ai_heads.py                  # YOLO-tiny + SegFormer-lite heads
├── training/                        # Training scripts
│   ├── stage1_train_wavelet.py      # Stage 1: Wavelet pre-training
│   ├── stage2_train_compressor.py   # Stage 2: Rate-distortion training
│   └── stage3_train_ai.py           # Stage 3: AI heads training
├── evaluation/                      # Evaluation utilities
│   ├── codec_metrics.py             # PSNR, MS-SSIM, BPP calculation
│   ├── task_metrics.py              # mAP, IoU, Top-1 metrics
│   ├── plot_rd_curves.py            # Rate-distortion curves
│   └── compare_baselines.py         # Baseline comparison
├── datasets/                        # Dataset setup và loaders
│   ├── setup_coco.sh                # COCO dataset setup script
│   ├── setup_davis.sh               # DAVIS dataset setup script
│   └── dataset_loaders.py           # PyTorch dataset loaders
├── checkpoints/                     # Model checkpoints
├── runs/                            # TensorBoard logs
├── fig/                             # Generated plots
└── PROJECT_CONTEXT.md               # Context tracking cho development
```

## 🔧 Dataset Setup

### COCO 2017
```bash
# Cài đặt COCO dataset
chmod +x datasets/setup_coco.sh
bash datasets/setup_coco.sh
```

### DAVIS 2017
```bash
# Cài đặt DAVIS dataset
chmod +x datasets/setup_davis.sh
bash datasets/setup_davis.sh
```

## 🎯 Training Pipeline

### Stage 1: Wavelet Pre-training (30 epochs)
```bash
# Pre-train WaveletTransformCNN với L2 reconstruction loss
python training/stage1_train_wavelet.py \
    --dataset coco \
    --epochs 30 \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --image_size 256
```

### Stage 2: Rate-Distortion Training (40 epochs)
```bash
# Train compressor với RD loss, freeze wavelet weights
python training/stage2_train_compressor.py \
    --dataset coco \
    --epochs 40 \
    --batch_size 8 \
    --lambda_rd 256 \
    --wavelet_checkpoint checkpoints/stage1_wavelet_coco_best.pth
```

### Stage 3: AI Heads Training (50 epochs)
```bash
# Train AI heads trên compressed features, freeze compressor
python training/stage3_train_ai.py \
    --task detection \
    --dataset coco \
    --epochs 50 \
    --batch_size 8 \
    --model yolo-tiny \
    --compressor_checkpoint checkpoints/stage2_compressor_coco_best.pth

# Segmentation training
python training/stage3_train_ai.py \
    --task segmentation \
    --dataset davis \
    --epochs 50 \
    --batch_size 8 \
    --model segformer-lite \
    --compressor_checkpoint checkpoints/stage2_compressor_davis_best.pth
```

## 📊 Evaluation

### Codec Performance
```bash
# Evaluate compression metrics
python evaluation/codec_metrics.py \
    --dataset coco \
    --split val \
    --checkpoint checkpoints/stage2_compressor_coco_best.pth \
    --output_csv results/codec_metrics.csv
```

### Task Performance
```bash
# Object detection evaluation
python evaluation/task_metrics.py \
    --task detection \
    --dataset coco \
    --checkpoint checkpoints/stage3_detection_coco_best.pth \
    --output_csv results/detection_metrics.csv

# Video segmentation evaluation
python evaluation/task_metrics.py \
    --task segmentation \
    --dataset davis \
    --checkpoint checkpoints/stage3_segmentation_davis_best.pth \
    --output_csv results/segmentation_metrics.csv
```

### Rate-Distortion Curves
```bash
# Generate RD curves cho paper
python evaluation/plot_rd_curves.py \
    --checkpoints checkpoints/stage2_compressor_*_best.pth \
    --output fig/rd_curves.png \
    --lambdas 256 512 1024
```

### Baseline Comparison
```bash
# So sánh với HEVC, VVC, và DVC
python evaluation/compare_baselines.py \
    --dataset coco \
    --wavenet_checkpoint checkpoints/stage3_detection_coco_best.pth \
    --output_csv results/baseline_comparison.csv
```

## 🧪 Testing

### Unit Tests
```bash
# Test individual components
python models/wavelet_transform_cnn.py
python models/adamixnet.py
python models/compressor_vnvc.py
python models/ai_heads.py
```

### Integration Tests
```bash
# Test full pipeline
python -m pytest tests/ -v
```

## 📈 Monitoring

### TensorBoard
```bash
# Monitor training progress
tensorboard --logdir runs/
```

### Logging
- Training logs: `runs/stage{1,2,3}_{model}_{dataset}/`
- Model checkpoints: `checkpoints/`
- Evaluation results: `results/`
- Generated plots: `fig/`

## 🎛️ Configuration

### Hyperparameters
- **Learning Rate**: 2e-4 với cosine decay
- **Batch Size**: 8 (default)
- **Seed**: 42 (for reproducibility)
- **Mixed Precision**: Enabled với AMP
- **Lambda Values**: [256, 512, 1024]

### Model Architecture
- **Wavelet Channels**: 64 (C')
- **AdaMixNet Output**: 128 (C_mix)
- **Latent Channels**: 192
- **Parallel Filters**: 4 (N)

## 📋 Validation Checklist

Framework tự động generate `checklist_report.md` để verify:

- ✅ WaveletTransformCNN có đúng PredictCNN & UpdateCNN layers
- ✅ AdaMixNet implement 4 parallel conv + softmax attention
- ✅ Compressor sử dụng CompressAI GaussianConditional
- ✅ AI heads consume compressed features (không pixel reconstruction)
- ✅ Training scripts save checkpoints + TensorBoard logs
- ✅ Evaluation outputs CSV + RD-plots
- ✅ README có dataset download + run commands

## 📊 Expected Results

### COCO 2017 Performance
| Lambda | PSNR (dB) | MS-SSIM | BPP | mAP@0.5 |
|--------|-----------|---------|-----|---------|
| 256    | ~28-30    | ~0.85   | 0.1 | ~0.65   |
| 512    | ~32-34    | ~0.90   | 0.2 | ~0.70   |
| 1024   | ~36-38    | ~0.95   | 0.4 | ~0.75   |

### DAVIS 2017 Performance
| Lambda | J&F Mean | Compression Ratio |
|--------|----------|-------------------|
| 256    | ~0.70    | 20:1              |
| 512    | ~0.75    | 10:1              |
| 1024   | ~0.80    | 5:1               |

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --batch_size 4
   ```

2. **Dataset Not Found**
   ```bash
   # Re-run setup scripts
   bash datasets/setup_coco.sh
   bash datasets/setup_davis.sh
   ```

3. **Checkpoint Loading Error**
   ```bash
   # Check checkpoint path và compatibility
   python -c "import torch; print(torch.load('checkpoint.pth').keys())"
   ```

## 📚 Citation

```bibtex
@article{wavenet_mv_2024,
  title={WAVENET-MV: Wavelet-based Video Compression for Machine Vision},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Run tests
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 📞 Support

- Issues: GitHub Issues
- Documentation: PROJECT_CONTEXT.md
- Results: RESULTS.md (auto-generated)

---

**Note**: Framework được design để chạy trên GPU. CPU training không được recommend do performance issues.

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/NgocMinhUET/WAVENET-MV.git
cd WAVENET-MV
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

3. Cài đặt dataset:
```bash
cd datasets
bash setup_coco.sh
```

## Training Pipeline

### Cách 1: Sử dụng script tự động

Chạy toàn bộ quá trình training tự động:

```bash
# Trên Linux/Mac
bash server_training.sh

# Trên Windows
# Sử dụng Git Bash hoặc WSL để chạy script bash
```

### Cách 2: Training từng giai đoạn

#### Stage 1: Train Wavelet Transform

```bash
python training/stage1_train_wavelet.py \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --dataset coco \
    --data_dir datasets/COCO
```

#### Stage 2: Train Compressor

```bash
python training/stage2_train_compressor.py \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --dataset coco \
    --data_dir datasets/COCO \
    --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
    --lambda_rd 256
```

#### Stage 3: Train AI Heads

```bash
python training/stage3_train_ai.py \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --dataset coco \
    --data_dir datasets/COCO \
    --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
    --stage2_checkpoint checkpoints/stage2_compressor_coco_lambda256_best.pth \
    --lambda_rd 256 \
    --enable_detection \
    --enable_segmentation
```

## Đánh giá

### Đánh giá tự động

Chạy đánh giá đầy đủ:

```bash
# Trên Linux/Mac
bash server_evaluation.sh

# Trên Windows
# Sử dụng Git Bash hoặc WSL để chạy script bash
```

### Đánh giá thủ công

```bash
python evaluation/codec_metrics_final.py \
    --stage1_checkpoint checkpoints/stage1_wavelet_coco_best.pth \
    --stage2_checkpoint checkpoints/stage2_compressor_coco_lambda256_best.pth \
    --dataset coco \
    --data_dir datasets/COCO \
    --split val \
    --max_samples 100 \
    --batch_size 4 \
    --output_file results/wavenet_mv_lambda256_evaluation.csv
```

## Tạo báo cáo

Tạo báo cáo tổng hợp từ kết quả đánh giá:

```bash
python evaluation/generate_summary_report.py \
    --input_dir results \
    --output_file results/wavenet_mv_comprehensive_results.csv
```

Phân tích thống kê:

```bash
python evaluation/statistical_analysis.py \
    --input_file results/wavenet_mv_comprehensive_results.csv \
    --output_file results/wavelet_contribution_analysis.csv \
    --analysis_type wavelet \
    --visualize
```

## Lưu ý quan trọng

1. **Fake Data**: Tất cả dữ liệu trong các file `.json` và `.csv` ban đầu là fake data, được tạo ra để minh họa định dạng kết quả. Để có kết quả thật, cần chạy training và evaluation.

2. **Checkpoints**: Cần chạy đầy đủ training pipeline để tạo ra các checkpoint thật sự trước khi đánh giá.

3. **Lambda Values**: Mô hình được train với nhiều giá trị lambda khác nhau (64, 128, 256, 512, 1024, 2048) để tạo ra các điểm khác nhau trên đường cong rate-distortion.

4. **Quantization**: Đã sửa các bug liên quan đến quantization collapse trong stage 2 training, đảm bảo mô hình hoạt động đúng.

## Liên hệ

Nếu có bất kỳ câu hỏi hoặc góp ý nào, vui lòng liên hệ qua email hoặc tạo issue trên GitHub. 