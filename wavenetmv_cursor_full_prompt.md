# Cursor AI – **Full Build Prompt** for *WAVENET‑MV*

> **Goal:** Generate all source code, training scripts, and evaluation utilities for the single‑version WAVENET‑MV framework (wavelet + AdaMixNet), ready to run on COCO 2017 (val) and DAVIS 2017.\
> *Framework:* PyTorch ≥ 1.13 + CompressAI.

---

## 0  Project Skeleton

```
./
 ├─ models/
 │   ├─ wavelet_transform_cnn.py     # detailed lifting CNN
 │   ├─ adamixnet.py                 # adaptive mixing network
 │   ├─ compressor_vnvc.py           # quant + entropy bottleneck
 │   └─ ai_heads.py                  # YOLO‑tiny, SegFormer‑lite
 ├─ training/
 │   ├─ stage1_train_wavelet.py      # pre‑train lifting CNN (L2)
 │   ├─ stage2_train_compressor.py   # rate‑distortion training
 │   └─ stage3_train_ai.py           # compressed‑domain AI heads
 ├─ evaluation/
 │   ├─ codec_metrics.py             # PSNR · MS‑SSIM · BPP
 │   ├─ task_metrics.py              # mAP · IoU · Top‑1
 │   └─ plot_rd_curves.py            # RD curves for paper
 ├─ datasets/  (scripts ↔ COCO · DAVIS)
 ├─ fig/          (auto‑saved plots)
 ├─ requirements.txt
 ├─ README.md    (run instructions)
 └─ RESULTS.md   (auto‑filled summary)
```

---

##

```
# Stage‑0  (RGB → feature)
Conv3x3  in=3   out=64  stride=1  pad=1   + ReLU

#=== Predict branch =======================================
# Predict high‑freq residual  H = P(X)
PredictCNN ≡
   Conv3x3  64→64  s1 p1  + ReLU
   Conv3x3  64→64  s1 p1  + ReLU
   Conv1x1  64→C′  s1 p0              ⟶  LH, HL, HH  (3×C′)

#=== Update branch ========================================
# Update low‑freq base  L = U(X, H)
UpdateCNN ≡
   [X ‖ H]  (# concat along channel)
   Conv3x3  (64+3C′)→64  + ReLU
   Conv3x3  64→64  + ReLU
   Conv1x1  64→C′            ⟶  LL  (1×C′)

#=== Output ===============================================
Return  cat(LL, LH, HL, HH)  # 4×C′ channels
```

**Loss (Stage‑1):**\
\(\mathcal L_{wavelet}=\|x-\hat{x}\|_2^2\)  using inverse lifting (shared transpose Convs).

### 1.2  `AdaMixNet`  (𝐵, 4C′, H, W) → (𝐵, C\_{mix}, H, W)

*`N = 4`*\* parallel filters, \**`C_{mix}=128`*

```
# Split input into N groups (or group‑conv)
for i = 1..N:
   Fi = Conv3x3  (4C′/N)→(C′/2)  + ReLU    # Multi‑Conv filters

# Attention weights
AttCNN ≡
   Conv3x3  4C′→64  + ReLU
   Conv1x1  64→N                  # logits → Softmax across N
   wi(x) = Softmax(logits)_i

# Weighted mixing
Y(x) = Σ_i  wi(x) · Fi(x)

# Channel reduction (optional)
Conv1x1  (C′/2)→C_{mix}
Return Y
```

### 1.3  `CompressorVNVC`  (𝐵, C\_{mix}, H, W) → bitstream

- **Quantizer:** round‑with‑noise.
- **Entropy model:** CompressAI `GaussianConditional` (learned σ).
- **Loss (Stage‑2):**\
  \(\mathcal L = \lambda\,\mathcal L_{rec} + \text{BPP}\),  λ ∈ {256, 512, 1024}.

### 1.4  `AIHeads`  (YOLO‑tiny & SegFormer‑lite)

- **Input:** decoded feature maps (float) without pixel recon.
- **Output:** detection boxes / seg masks / action logits.
- **Loss (Stage‑3):** task‑specific + optional KD from pixel‑domain teacher.

---

## 2  Training Schedule

| Stage  | Script                        | Epochs | Trainable                    | Loss        |
| ------ | ----------------------------- | ------ | ---------------------------- | ----------- |
|  1     |  `stage1_train_wavelet.py`    |  30    | WaveletCNN                   |  L₂ recon   |
|  2     |  `stage2_train_compressor.py` |  40    | Compressor (+freeze wavelet) |  λ·L₂ + BPP |
|  3     |  `stage3_train_ai.py`         |  50    | AIHeads (+freeze prev.)      |  Task loss  |

Use Adam, LR = 2e‑4 → cosine decay; batch = 8; seed = 42.

---

## 3  Evaluation & Reporting

1. **Codec metrics**: `codec_metrics.py` → CSV (BPP, PSNR, MS‑SSIM)
2. **Task metrics**: `task_metrics.py` → CSV (mAP, IoU, Top‑1)
3. **Plots**: `plot_rd_curves.py` → PDF & PNG in ./fig/
4. Auto‑write `RESULTS.md` summarising tables & discussion.

---

## 4  Coding & Logging Rules

- PyTorch ≥1.13, TorchVision ≥0.14, CompressAI ≥1.2.
- Use `torch.cuda.amp` mixed‑precision.
- Each script logs to TensorBoard (`./runs/`).
- Raise `ValueError` for size mismatch; unit‑test forward pass in `pytest`.

---

## 5  Begin 🡒

1. Generate folder tree & empty files.
2. Fill in `wavelet_transform_cnn.py` with the layer‑exact architecture above.
3. Implement AdaMixNet as specified.
4. Write Compressor, AI heads, training scripts, evaluation utilities.
5. Populate `README.md` with installation & run commands.
6. Halt only when all TODO markers are resolved and `RESULTS.md` is created.

---

## 5  Validation Checklist (auto‑run)

Before finishing, the pipeline must generate `` answering **YES / NO** for each item:

1. WaveletTransformCNN includes PredictCNN & UpdateCNN layers exactly (3×3‑ReLU‑3×3‑ReLU‑1×1).
2. AdaMixNet implements 4 parallel conv branches **and** softmax attention mixing.
3. Compressor uses CompressAI `GaussianConditional`; λ ∈ {256, 512, 1024}.
4. AI heads consume *compressed features* without pixel reconstruction.
5. All training scripts save checkpoints and TensorBoard logs.
6. Evaluation scripts output CSV + RD‑plots under `./fig/`.
7. README contains dataset download and run commands.

`checklist_report.md` must be auto‑generated by the final evaluation script and copied to the project root.

---

## 6  Dataset Setup Scripts

Create bash scripts inside `datasets/`:

```bash
# setup_coco.sh
mkdir -p datasets/COCO && cd datasets/COCO
wget http://images.cocodataset.org/zips/val2017.zip -O val2017.zip
unzip val2017.zip && rm val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O ann.zip
unzip ann.zip && rm ann.zip
```

```bash
# setup_davis.sh
mkdir -p datasets/DAVIS && cd datasets/DAVIS
wget https://davischallenge.org/challenge2017/DAVIS-data/DAVIS-2017-trainval-480p.zip -O davis.zip
unzip davis.zip && rm davis.zip
```

Add to README:

```bash
bash datasets/setup_coco.sh
bash datasets/setup_davis.sh
```

---

## 7  Baseline Comparison Utilities

1. Implement `evaluation/compare_baselines.py`: • Decode WAVENET‑MV bitstreams. • Decode HEVC and VVC (use `ffmpeg` or `x265`). • Load public DVC checkpoint. • Run same AI heads on each decoded input. • Produce a table:\
   `| Codec | BPP | PSNR | MS‑SSIM | mAP | IoU | Latency |`\
   • Save as `baseline_comparison.csv`.
2. Provide helper shell script `hevc_encode.sh` to batch‑encode sample MP4s via x265 at CRF=28.

---

## 8  Execution Order

1. **Generate** folder tree & stubs.
2. **Fill** `wavelet_transform_cnn.py`, `adamixnet.py`, `compressor_vnvc.py`, `ai_heads.py` per spec.
3. **Create** dataset setup scripts and add instructions to README.
4. **Write** training & evaluation scripts; ensure they call `checklist_report.md` generator.
5. **Run** minimal smoke‑test (single batch) to validate forward pass.
6. **Stop** when `RESULTS.md`, `checklist_report.md`, and plots are present.

