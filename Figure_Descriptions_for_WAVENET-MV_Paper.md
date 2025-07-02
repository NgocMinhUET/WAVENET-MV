# Figure Descriptions for WAVENET-MV IEEE Paper

## Figure 1: Overall Architecture (`fig_architecture.png`)
**Type:** Full-width figure (spans both columns)
**Dimensions:** 1200×400 pixels, 300 DPI

### Layout Description:
```
Input Image → Wavelet Transform CNN → AdaMixNet → Compressor → Multi-task Heads
  (256×256×3)     (4 subbands)      (128 ch)    (compressed)  (Detection + Segmentation)
```

### Detailed Components:
1. **Input Image Box:**
   - 256×256×3 RGB image sample (use a COCO image with cars and people)
   - Label: "Input Image (H×W×3)"

2. **Wavelet Transform CNN:**
   - Show decomposition into 4 subbands: LL, LH, HL, HH
   - Each subband: 64×64×64 dimensions
   - Use different colors: LL (blue), LH (green), HL (red), HH (yellow)
   - Label each subband clearly

3. **AdaMixNet Module:**
   - Show 4 parallel filters processing each subband
   - Attention mechanism with softmax weights
   - Output: 64×64×128 mixed features
   - Use arrows to show attention flow

4. **Variational Neural Codec:**
   - Analysis Transform → Quantization → Synthesis Transform
   - Show compressed representation as smaller feature maps
   - Include rate-distortion loss equation

5. **Multi-task Heads:**
   - Detection Head: YOLO-tiny architecture
   - Segmentation Head: SegFormer-lite architecture
   - Show sample outputs: bounding boxes and segmentation masks

### Color Coding:
- **Blue:** Frozen components during training
- **Green:** Trainable components
- **Orange:** Data flow arrows
- **Red:** Gradient flow arrows (dashed)

### Annotations:
- Dimension labels for each stage
- Component names in clear fonts
- Stage indicators (Stage 1, 2, 3)

---

## Figure 2: AdaMixNet Detail (`fig_adamixnet.png`)
**Type:** Single column figure
**Dimensions:** 600×500 pixels, 300 DPI

### Layout Description:
```
Wavelet Coefficients (4×C')
    ↓ (split into 4 groups)
[Filter1] [Filter2] [Filter3] [Filter4]
    ↓        ↓        ↓        ↓
[Conv3×3] [Conv3×3] [Conv3×3] [Conv3×3]
    ↓        ↓        ↓        ↓
[F1]     [F2]     [F3]     [F4]
    ↓        ↓        ↓        ↓
    Attention Mechanism
         ↓
   Mixed Features (128)
```

### Components:
1. **Input:** 256 channels (4×64) shown as colored blocks
2. **Parallel Filters:** 4 identical Conv3×3 blocks
3. **Attention Branch:** 
   - Conv3×3(256→64) + ReLU
   - Conv1×1(64→4) + Softmax
   - Show attention weights as heatmap
4. **Output:** 128-channel mixed features

### Color Scheme:
- LL subband: Blue (#2E86AB)
- LH subband: Green (#A23B72)
- HL subband: Orange (#F18F01)
- HH subband: Purple (#C73E1D)

---

## Figure 3: Training Pipeline (`fig_training_pipeline.png`)
**Type:** Single column figure
**Dimensions:** 600×600 pixels, 300 DPI

### Layout Description:
Three horizontal sections representing stages:

#### Stage 1 (Top):
```
Input → [Wavelet CNN] → Inverse → L2 Loss
         (trainable)
```
- Duration: 30 epochs
- Loss curve showing convergence

#### Stage 2 (Middle):
```
Input → [Wavelet CNN] → [AdaMixNet] → [Compressor] → RD Loss
         (frozen)       (trainable)   (trainable)
```
- Duration: 40 epochs
- Show λ parameter sweep

#### Stage 3 (Bottom):
```
Input → [Full Pipeline] → [Multi-task Heads] → Task Losses
         (frozen)          (trainable)
```
- Duration: 50 epochs
- Detection + Segmentation losses

### Visual Elements:
- Timeline at bottom showing epoch progression
- Color coding: Frozen (blue), Trainable (green)
- Loss curves for each stage
- Gradient flow arrows

---

## Figure 4: Rate-Distortion Curves (`fig_rd_curves.png`)
**Type:** Single column figure
**Dimensions:** 600×400 pixels, 300 DPI

### Layout Description:
Two subplots side by side:

#### Subplot (a): PSNR vs BPP
- X-axis: BPP (0.1 to 10.0, log scale)
- Y-axis: PSNR (dB) (0 to 35)
- Grid: Light gray

#### Subplot (b): MS-SSIM vs BPP
- X-axis: BPP (0.1 to 10.0, log scale)
- Y-axis: MS-SSIM (0 to 1.0)
- Grid: Light gray

### Data Points:
- **JPEG:** Blue circles, quality points 10,30,50,70,90
- **WebP:** Green squares, quality points 10,30,50,70,90
- **PNG:** Orange triangle, single point
- **WAVENET-MV:** Red diamonds, λ values 64,128,256,512,1024

### Styling:
- Line width: 2px
- Marker size: 8px
- Legend in upper right
- Axis labels: 14pt font
- Title: 16pt font

---

## Figure 5: Qualitative Results (`fig_qualitative_results.png`)
**Type:** Full-width figure (spans both columns)
**Dimensions:** 1200×600 pixels, 300 DPI

### Layout Description:
3 rows × 4 columns grid:

#### Row 1: Original Images + Ground Truth
- 4 diverse COCO images
- Green bounding boxes for objects
- Colored segmentation masks (semi-transparent)

#### Row 2: JPEG Results (Q=70)
- Same images after JPEG compression
- Detection/segmentation results overlaid
- Slight quality degradation visible

#### Row 3: WAVENET-MV Results (λ=512)
- Images processed through WAVENET-MV
- Detection/segmentation results overlaid
- Maintained accuracy despite compression

### Image Selection:
1. Street scene with cars and pedestrians
2. Indoor scene with furniture
3. Sports scene with multiple people
4. Nature scene with animals

### Annotation Colors:
- Correct detections: Green (#00FF00)
- False positives: Yellow (#FFFF00)
- Missed detections: Red (#FF0000)
- Segmentation masks: Class-specific colors

---

## Figure 6: Ablation Study (`fig_ablation_study.png`)
**Type:** Single column figure
**Dimensions:** 600×400 pixels, 300 DPI

### Layout Description:
Two subplots:

#### Subplot (a): Component Contribution (Bar Chart)
- X-axis: Configuration names
  - "Full Model"
  - "w/o Wavelet CNN"
  - "w/o AdaMixNet"
  - "w/o Staged Training"
- Y-axis: Performance (mAP/mIoU)
- Two bars per configuration: mAP (blue), mIoU (orange)

#### Subplot (b): Training Convergence
- X-axis: Training epochs (0-120)
- Y-axis: Loss value (log scale)
- Two lines:
  - Staged training: Smooth blue line
  - End-to-end: Jagged red line
- Show clear instability in end-to-end approach

### Styling:
- Bar width: 0.35
- Error bars if available
- Grid: Light gray
- Legend for both subplots

---

## Technical Specifications for All Figures:

### Fonts:
- Title: Arial Bold, 16pt
- Axis labels: Arial, 14pt
- Tick labels: Arial, 12pt
- Captions: Times New Roman, 10pt

### Colors (Colorblind-friendly palette):
- Primary: #1f77b4 (blue)
- Secondary: #ff7f0e (orange)
- Tertiary: #2ca02c (green)
- Quaternary: #d62728 (red)
- Background: White (#FFFFFF)
- Grid: Light gray (#E0E0E0)

### File Format:
- Save as PNG with 300 DPI
- Also save as PDF for LaTeX compilation
- Use vector graphics where possible for scalability

### LaTeX Integration:
- All figures should fit within column/page margins
- Use consistent sizing with IEEE template
- Include proper figure references in text
- Ensure high contrast for black/white printing

---

## Software Recommendations:
- **Architecture diagrams:** Draw.io, Visio, or Inkscape
- **Plots:** Matplotlib (Python) or ggplot2 (R)
- **Image editing:** GIMP or Photoshop
- **Vector graphics:** Inkscape or Adobe Illustrator 