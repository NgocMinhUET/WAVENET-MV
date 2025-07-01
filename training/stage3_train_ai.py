"""
Stage 3 Training: AI Heads Training 
- Epochs: 50
- Trainable: YOLO-tiny + SegFormer-lite AI Heads
- Frozen: WaveletCNN + AdaMixNet + CompressorVNVC (chỉ dùng compressed features)
- Loss: Task-specific (detection + segmentation) + optional KD
"""

import os
import sys
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path

# Add parent directory to path để import models
sys.path.append(str(Path(__file__).parent.parent))

from models.wavelet_transform_cnn import WaveletTransformCNN
from models.adamixnet import AdaMixNet
from models.compressor_vnvc import CompressorVNVC
from models.ai_heads import YOLOTinyHead, SegFormerLiteHead, KnowledgeDistillationLoss
from datasets.dataset_loaders import COCODatasetLoader, DAVISDatasetLoader


def stage3_collate_fn(batch):
    """
    Custom collate function cho Stage 3 - images + task labels
    """
    images = []
    detection_labels = []
    segmentation_labels = []
    
    for item in batch:
        if isinstance(item, dict):
            images.append(item['image'])
            if 'detection' in item:
                detection_labels.append(item['detection'])
            if 'segmentation' in item:
                segmentation_labels.append(item['segmentation'])
        else:
            images.append(item[0])
    
    result = {'image': torch.stack(images, 0)}
    
    if detection_labels:
        result['detection'] = detection_labels
    if segmentation_labels:
        result['segmentation'] = torch.stack(segmentation_labels, 0)
    
    return result


def set_seed(seed=42):
    """Set seed cho reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Stage3Trainer:
    """Trainer cho Stage 3 - AI Heads training"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set seed
        set_seed(args.seed)
        
        # Initialize models
        self.setup_models()
        
        # Loss functions
        self.setup_loss_functions()
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
        # Setup datasets
        self.setup_datasets()
        
        # Setup optimizer và scheduler
        self.setup_optimizer()
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=f'runs/stage3_ai_heads_{args.dataset}')
        
        # Best model tracking
        self.best_loss = float('inf')
        
    def setup_models(self):
        """Setup models: frozen compression pipeline + trainable AI heads"""
        
        # === FROZEN COMPRESSION PIPELINE ===
        
        # 1. Load WaveletCNN (frozen)
        self.wavelet_model = WaveletTransformCNN(
            input_channels=3,
            feature_channels=64,
            wavelet_channels=64
        ).to(self.device)
        
        if self.args.stage1_checkpoint:
            print(f"Loading Stage 1 checkpoint: {self.args.stage1_checkpoint}")
            checkpoint = torch.load(self.args.stage1_checkpoint, map_location=self.device)
            self.wavelet_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 2. Load AdaMixNet (frozen)
        self.adamix_model = AdaMixNet(
            input_channels=4 * 64,  # 4×C' = 4×64 = 256
            C_prime=64,
            C_mix=128,
            N=4
        ).to(self.device)
        
        # 3. Load CompressorVNVC (frozen)
        self.compressor_model = CompressorVNVC(
            input_channels=128,
            lambda_rd=self.args.lambda_rd
        ).to(self.device)
        
        if self.args.stage2_checkpoint:
            print(f"Loading Stage 2 checkpoint: {self.args.stage2_checkpoint}")
            checkpoint = torch.load(self.args.stage2_checkpoint, map_location=self.device)
            self.adamix_model.load_state_dict(checkpoint['adamix_state_dict'])
            self.compressor_model.load_state_dict(checkpoint['compressor_state_dict'])
        
        # Freeze compression pipeline
        for model in [self.wavelet_model, self.adamix_model, self.compressor_model]:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        
        # === TRAINABLE AI HEADS ===
        
        # 4. YOLO-tiny Head
        if self.args.enable_detection:
            self.yolo_head = YOLOTinyHead(
                input_channels=128,  # Compressed features
                num_classes=80,      # COCO classes
                num_anchors=3
            ).to(self.device)
        
        # 5. SegFormer-lite Head  
        if self.args.enable_segmentation:
            self.segformer_head = SegFormerLiteHead(
                input_channels=128,  # Compressed features
                num_classes=21       # PASCAL VOC classes
            ).to(self.device)
        
        # Print parameter counts
        frozen_params = sum(p.numel() for model in [self.wavelet_model, self.adamix_model, self.compressor_model] 
                           for p in model.parameters())
        trainable_params = 0
        
        if self.args.enable_detection:
            trainable_params += sum(p.numel() for p in self.yolo_head.parameters() if p.requires_grad)
            print(f"✓ YOLO-tiny Head: {sum(p.numel() for p in self.yolo_head.parameters() if p.requires_grad):,} params")
        
        if self.args.enable_segmentation:
            trainable_params += sum(p.numel() for p in self.segformer_head.parameters() if p.requires_grad)
            print(f"✓ SegFormer-lite Head: {sum(p.numel() for p in self.segformer_head.parameters() if p.requires_grad):,} params")
        
        print(f"✓ Frozen compression pipeline: {frozen_params:,} params")
        print(f"✓ Trainable AI heads: {trainable_params:,} params")
        
    def setup_loss_functions(self):
        """Setup loss functions cho tasks"""
        
        # Detection loss (simplified YOLO loss)
        self.detection_criterion = nn.CrossEntropyLoss()
        self.bbox_criterion = nn.MSELoss()
        self.obj_criterion = nn.BCEWithLogitsLoss()
        
        # Segmentation loss
        self.segmentation_criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Knowledge distillation (optional)
        if self.args.use_kd:
            self.kd_loss = KnowledgeDistillationLoss(
                temperature=self.args.kd_temperature,
                alpha=self.args.kd_alpha
            )
        
    def setup_datasets(self):
        """Setup datasets with task annotations"""
        if self.args.dataset == 'coco':
            dataset_loader = COCODatasetLoader(
                data_dir=self.args.data_dir,
                image_size=self.args.image_size,
                subset='train',
                load_detection=self.args.enable_detection,
                load_segmentation=self.args.enable_segmentation
            )
            val_dataset_loader = COCODatasetLoader(
                data_dir=self.args.data_dir,
                image_size=self.args.image_size,
                subset='val',
                load_detection=self.args.enable_detection,
                load_segmentation=self.args.enable_segmentation
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")
        
        self.train_loader = DataLoader(
            dataset_loader,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=stage3_collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset_loader,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=stage3_collate_fn
        )
        
    def setup_optimizer(self):
        """Setup optimizer và scheduler cho AI heads"""
        trainable_params = []
        
        if self.args.enable_detection:
            trainable_params.extend(list(self.yolo_head.parameters()))
        if self.args.enable_segmentation:
            trainable_params.extend(list(self.segformer_head.parameters()))
        
        self.optimizer = optim.Adam(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Cosine scheduler
        num_training_steps = len(self.train_loader) * self.args.epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=num_training_steps,
            eta_min=1e-6
        )
        
    def get_compressed_features(self, images):
        """Get compressed features từ frozen pipeline"""
        with torch.no_grad():
            # Stage 1: Wavelet transform
            wavelet_coeffs = self.wavelet_model(images)
            
            # Stage 2a: AdaMixNet
            mixed_features = self.adamix_model(wavelet_coeffs)
            
            # Stage 2b: CompressorVNVC (chỉ lấy compressed features)
            compressed_features, _, _ = self.compressor_model(mixed_features)
            
        return compressed_features.detach()  # Detach để ensure frozen
        
    def train_epoch(self, epoch):
        """Train một epoch"""
        if self.args.enable_detection:
            self.yolo_head.train()
        if self.args.enable_segmentation:
            self.segformer_head.train()
        
        running_loss = 0.0
        running_det_loss = 0.0
        running_seg_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            with autocast():
                # Get compressed features từ frozen pipeline
                compressed_features = self.get_compressed_features(images)
                
                total_loss = 0.0
                det_loss = 0.0
                seg_loss = 0.0
                
                # === DETECTION TASK ===
                if self.args.enable_detection and 'detection' in batch:
                    detection_pred = self.yolo_head(compressed_features)
                    # Simplified detection loss (implement proper YOLO loss later)
                    det_loss = torch.tensor(0.1, device=self.device, requires_grad=True)  # Placeholder
                    total_loss += det_loss
                
                # === SEGMENTATION TASK ===
                if self.args.enable_segmentation and 'segmentation' in batch:
                    seg_targets = batch['segmentation'].to(self.device)
                    seg_pred = self.segformer_head(compressed_features)
                    
                    # Resize predictions to match targets if needed
                    if seg_pred.shape[2:] != seg_targets.shape[1:]:
                        seg_pred = F.interpolate(
                            seg_pred,
                            size=seg_targets.shape[1:],
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    seg_loss = self.segmentation_criterion(seg_pred, seg_targets)
                    total_loss += seg_loss
                
                # Ensure we have some loss
                if total_loss == 0:
                    total_loss = torch.tensor(0.01, device=self.device, requires_grad=True)
            
            # Backward pass
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update scheduler
            self.scheduler.step()
            
            # Update running losses
            running_loss += total_loss.item()
            running_det_loss += det_loss.item() if isinstance(det_loss, torch.Tensor) else det_loss
            running_seg_loss += seg_loss.item() if isinstance(seg_loss, torch.Tensor) else seg_loss
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.6f}',
                'Det': f'{det_loss.item() if isinstance(det_loss, torch.Tensor) else det_loss:.6f}',
                'Seg': f'{seg_loss.item() if isinstance(seg_loss, torch.Tensor) else seg_loss:.6f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to TensorBoard
            global_step = epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/TotalLoss', total_loss.item(), global_step)
            self.writer.add_scalar('Train/DetectionLoss', running_det_loss, global_step)
            self.writer.add_scalar('Train/SegmentationLoss', running_seg_loss, global_step)
            self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], global_step)
        
        avg_loss = running_loss / num_batches
        avg_det_loss = running_det_loss / num_batches
        avg_seg_loss = running_seg_loss / num_batches
        
        return avg_loss, avg_det_loss, avg_seg_loss
        
    def validate(self, epoch):
        """Validation"""
        if self.args.enable_detection:
            self.yolo_head.eval()
        if self.args.enable_segmentation:
            self.segformer_head.eval()
        
        val_loss = 0.0
        val_det_loss = 0.0
        val_seg_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                
                with autocast():
                    # Get compressed features
                    compressed_features = self.get_compressed_features(images)
                    
                    total_loss = 0.0
                    det_loss = 0.0
                    seg_loss = 0.0
                    
                    # Detection validation
                    if self.args.enable_detection and 'detection' in batch:
                        detection_pred = self.yolo_head(compressed_features)
                        det_loss = 0.1  # Placeholder
                        total_loss += det_loss
                    
                    # Segmentation validation
                    if self.args.enable_segmentation and 'segmentation' in batch:
                        seg_targets = batch['segmentation'].to(self.device)
                        seg_pred = self.segformer_head(compressed_features)
                        
                        if seg_pred.shape[2:] != seg_targets.shape[1:]:
                            seg_pred = F.interpolate(
                                seg_pred,
                                size=seg_targets.shape[1:],
                                mode='bilinear',
                                align_corners=False
                            )
                        
                        seg_loss = self.segmentation_criterion(seg_pred, seg_targets).item()
                        total_loss += seg_loss
                    
                    if total_loss == 0:
                        total_loss = 0.01
                
                val_loss += total_loss
                val_det_loss += det_loss
                val_seg_loss += seg_loss
        
        avg_val_loss = val_loss / num_batches
        avg_val_det = val_det_loss / num_batches
        avg_val_seg = val_seg_loss / num_batches
        
        # Log to TensorBoard
        self.writer.add_scalar('Val/TotalLoss', avg_val_loss, epoch)
        self.writer.add_scalar('Val/DetectionLoss', avg_val_det, epoch)
        self.writer.add_scalar('Val/SegmentationLoss', avg_val_seg, epoch)
        
        return avg_val_loss, avg_val_det, avg_val_seg
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'scaler_state_dict': self.scaler.state_dict(),
            'args': self.args
        }
        
        # Save AI heads
        if self.args.enable_detection:
            checkpoint['yolo_state_dict'] = self.yolo_head.state_dict()
        if self.args.enable_segmentation:
            checkpoint['segformer_state_dict'] = self.segformer_head.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = f'checkpoints/stage3_ai_heads_{self.args.dataset}_latest.pth'
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = f'checkpoints/stage3_ai_heads_{self.args.dataset}_best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved với val loss: {loss:.6f}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting Stage 3 Training - AI Heads")
        print(f"Dataset: {self.args.dataset}")
        print(f"Device: {self.device}")
        print(f"Detection: {self.args.enable_detection}")
        print(f"Segmentation: {self.args.enable_segmentation}")
        
        for epoch in range(self.args.epochs):
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            
            # Train
            train_loss, train_det, train_seg = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_det, val_seg = self.validate(epoch)
            
            print(f"Train - Loss: {train_loss:.6f}, Det: {train_det:.6f}, Seg: {train_seg:.6f}")
            print(f"Val   - Loss: {val_loss:.6f}, Det: {val_det:.6f}, Seg: {val_seg:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping check
            if epoch > 20 and val_loss > self.best_loss * 1.05:
                print("Early stopping triggered!")
                break
        
        print(f"\nStage 3 Training completed!")
        print(f"Best validation loss: {self.best_loss:.6f}")
        
        # Close TensorBoard writer
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Stage 3: AI Heads Training')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, choices=['coco'], default='coco',
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='datasets/COCO_Official',
                       help='Dataset directory')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    
    # Checkpoint arguments
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                       help='Path to Stage 1 checkpoint')
    parser.add_argument('--stage2_checkpoint', type=str, required=True,
                       help='Path to Stage 2 checkpoint')
    parser.add_argument('--lambda_rd', type=int, default=128,
                       help='Lambda RD value used in Stage 2')
    
    # Task arguments
    parser.add_argument('--enable_detection', action='store_true',
                       help='Enable object detection task')
    parser.add_argument('--enable_segmentation', action='store_true',
                       help='Enable segmentation task')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Knowledge distillation arguments
    parser.add_argument('--use_kd', action='store_true',
                       help='Use knowledge distillation')
    parser.add_argument('--kd_temperature', type=float, default=4.0,
                       help='KD temperature')
    parser.add_argument('--kd_alpha', type=float, default=0.7,
                       help='KD alpha')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loader workers')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.enable_detection and not args.enable_segmentation:
        raise ValueError("Must enable at least one task: --enable_detection or --enable_segmentation")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create trainer và run
    trainer = Stage3Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main() 