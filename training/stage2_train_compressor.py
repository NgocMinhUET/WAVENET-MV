"""
Stage 2 Training: CompressorVNVC + AdaMixNet Training
- Epochs: 40
- Trainable: CompressorVNVC (freeze WaveletCNN)
- Loss: λ·L₂ + BPP (Rate-Distortion)
- λ values: {256, 512, 1024}
- Optimizer: Adam, LR=2e-4 → cosine decay, batch=8, seed=42
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
from datasets.dataset_loaders import COCODatasetLoader, DAVISDatasetLoader


def stage2_collate_fn(batch):
    """
    Custom collate function cho Stage 2 - chỉ lấy images
    """
    images = []
    for item in batch:
        if isinstance(item, dict):
            images.append(item['image'])
        else:
            images.append(item[0])
    
    return {
        'image': torch.stack(images, 0)
    }


def set_seed(seed=42):
    """Set seed cho reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, eta_min=1e-6):
    """Cosine decay scheduler với warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(eta_min, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Stage2Trainer:
    """Trainer cho Stage 2 - Compressor training"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set seed
        set_seed(args.seed)
        
        # Initialize models
        self.setup_models()
        
        # Loss functions
        self.mse_criterion = nn.MSELoss()
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
        # Setup datasets
        self.setup_datasets()
        
        # Setup optimizer và scheduler
        self.setup_optimizer()
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=f'runs/stage2_compressor_{args.dataset}_lambda{args.lambda_rd}')
        
        # Best model tracking
        self.best_loss = float('inf')
        
    def setup_models(self):
        """Setup models: frozen WaveletCNN + trainable Compressor"""
        # Load pretrained WaveletCNN
        self.wavelet_model = WaveletTransformCNN(
            input_channels=3,
            feature_channels=64,
            wavelet_channels=64
        ).to(self.device)
        
        # Load Stage 1 checkpoint
        if self.args.stage1_checkpoint:
            print(f"Loading Stage 1 checkpoint: {self.args.stage1_checkpoint}")
            checkpoint = torch.load(self.args.stage1_checkpoint, map_location=self.device)
            self.wavelet_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze WaveletCNN
        for param in self.wavelet_model.parameters():
            param.requires_grad = False
        self.wavelet_model.eval()
        
        # AdaMixNet: 4C' → C_mix=128
        wavelet_channels = 64  # From Stage 1
        self.adamix_model = AdaMixNet(
            input_channels=4 * wavelet_channels,  # 4×C' = 4×64 = 256
            C_prime=wavelet_channels,  # C' = 64
            C_mix=128,  # Output channels
            N=4  # Number of parallel filters
        ).to(self.device)
        
        # CompressorVNVC
        self.compressor_model = CompressorVNVC(
            input_channels=128,  # C_mix from AdaMixNet
            lambda_rd=self.args.lambda_rd
        ).to(self.device)
        
        print(f"✓ WaveletCNN (frozen): {sum(p.numel() for p in self.wavelet_model.parameters()):,} params")
        print(f"✓ AdaMixNet: {sum(p.numel() for p in self.adamix_model.parameters() if p.requires_grad):,} params")
        print(f"✓ CompressorVNVC: {sum(p.numel() for p in self.compressor_model.parameters() if p.requires_grad):,} params")
        
    def setup_datasets(self):
        """Setup datasets"""
        if self.args.dataset == 'coco':
            dataset_loader = COCODatasetLoader(
                data_dir=self.args.data_dir,
                image_size=self.args.image_size,
                subset='train'
            )
            val_dataset_loader = COCODatasetLoader(
                data_dir=self.args.data_dir,
                image_size=self.args.image_size,
                subset='val'
            )
        elif self.args.dataset == 'davis':
            dataset_loader = DAVISDatasetLoader(
                data_dir=self.args.data_dir,
                image_size=self.args.image_size,
                subset='train'
            )
            val_dataset_loader = DAVISDatasetLoader(
                data_dir=self.args.data_dir,
                image_size=self.args.image_size,
                subset='val'
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")
        
        self.train_loader = DataLoader(
            dataset_loader,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=stage2_collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset_loader,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=stage2_collate_fn
        )
        
    def setup_optimizer(self):
        """Setup optimizer và scheduler"""
        # Chỉ train AdaMixNet + CompressorVNVC
        trainable_params = list(self.adamix_model.parameters()) + list(self.compressor_model.parameters())
        
        self.optimizer = optim.Adam(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Setup scheduler
        num_training_steps = len(self.train_loader) * self.args.epochs
        num_warmup_steps = len(self.train_loader) * 2  # 2 epochs warmup
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
    def train_epoch(self, epoch):
        """Train một epoch"""
        self.adamix_model.train()
        self.compressor_model.train()
        
        running_loss = 0.0
        running_mse_loss = 0.0
        running_bpp_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass với mixed precision
            with autocast():
                # Stage 1: Wavelet transform (frozen)
                with torch.no_grad():
                    wavelet_coeffs = self.wavelet_model(images)  # [B, 4*C', H, W]
                
                # Stage 2a: AdaMixNet
                mixed_features = self.adamix_model(wavelet_coeffs)  # [B, 128, H, W]
                
                # Stage 2b: CompressorVNVC
                x_hat, likelihoods, y_quantized = self.compressor_model(mixed_features)
                compressed_features = x_hat
                
                # Calculate BPP from likelihoods - FIXED: Use original image dimensions
                # Original image dimensions for BPP calculation (NOT mixed_features dimensions)
                batch_size = images.size(0)  # FIXED: Use images, not mixed_features
                num_pixels = images.size(2) * images.size(3)  # H * W of ORIGINAL images
                
                # Rate calculation: -log2(likelihoods) summed over all dimensions
                log_likelihoods = torch.log(likelihoods.clamp(min=1e-10))
                total_bits = -log_likelihoods.sum() / math.log(2)
                bpp = total_bits / (batch_size * num_pixels)
                
                # DEBUG: Check shapes và tensor values
                if epoch == 0 and batch_idx == 0:
                    print(f"🔍 DEBUG - Mixed features: {mixed_features.shape}, range: [{mixed_features.min():.4f}, {mixed_features.max():.4f}]")
                    print(f"🔍 DEBUG - Compressed features: {compressed_features.shape}, range: [{compressed_features.min():.4f}, {compressed_features.max():.4f}]")
                    diff = torch.abs(compressed_features - mixed_features)
                    print(f"🔍 DEBUG - Difference: mean={diff.mean():.8f}, max={diff.max():.8f}")
                    print(f"🔍 DEBUG - Original images: {images.shape}, Mixed features: {mixed_features.shape}")
                    print(f"🔍 DEBUG - BPP calculation: {total_bits:.2f} bits / ({batch_size} * {num_pixels}) = {bpp:.4f}")
                    if diff.max() < 1e-6:
                        print("🚨 BUG: CompressorVNVC acting as identity function!")
                    else:
                        print("✅ CompressorVNVC applying compression (good!)")
                
                # Shape check và resize nếu cần
                if compressed_features.shape != mixed_features.shape:
                    print(f"⚠️ Shape mismatch: {compressed_features.shape} vs {mixed_features.shape}")
                    compressed_features = F.interpolate(
                        compressed_features,
                        size=mixed_features.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    print(f"✅ Resized to: {compressed_features.shape}")
                
                # Reconstruction loss (MSE) - REMOVED FLOOR to expose real MSE
                mse_loss = self.mse_criterion(compressed_features, mixed_features)
                # NO MSE FLOOR! Let's see the real MSE values
                
                # Total loss: λ·MSE + BPP
                total_loss = self.args.lambda_rd * mse_loss + bpp
                
                # MSE Health Check (first epoch only)
                if epoch == 0 and batch_idx == 0:
                    mse_component = self.args.lambda_rd * mse_loss
                    bpp_component = bpp
                    total_loss_val = total_loss.item()
                    
                    mse_ratio = (mse_component / total_loss_val * 100).item()
                    bpp_ratio = (bpp_component / total_loss_val * 100).item()
                    
                    print(f"🏥 MSE HEALTH CHECK:")
                    print(f"   MSE Loss: {mse_loss.item():.6f}")
                    print(f"   λ*MSE: {mse_component.item():.4f} ({mse_ratio:.1f}%)")
                    print(f"   BPP: {bpp_component.item():.4f} ({bpp_ratio:.1f}%)")
                    print(f"   Total: {total_loss_val:.4f}")
                    
                    # Health indicators
                    if mse_loss < 1e-6:
                        print("   ❌ MSE TOO SMALL: Potential identity function!")
                    elif mse_loss < 1e-3:
                        print("   ⚠️ MSE VERY SMALL: Monitor for collapse")
                    elif mse_loss < 0.1:
                        print("   ✅ MSE HEALTHY: Good compression range")
                    else:
                        print("   ⚠️ MSE LARGE: High distortion, consider increasing λ")
                    
                    if mse_ratio < 1.0:
                        print("   ❌ MSE IGNORED: MSE component < 1% of total loss!")
                    elif mse_ratio < 10.0:
                        print("   ⚠️ MSE WEAK: MSE component < 10% of total loss")
                    elif mse_ratio > 90.0:
                        print("   ⚠️ BPP IGNORED: BPP component < 10% of total loss")  
                    else:
                        print("   ✅ BALANCED: Good MSE/BPP balance")
            
            # Backward pass
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update scheduler
            self.scheduler.step()
            
            # Update running losses
            running_loss += total_loss.item()
            running_mse_loss += mse_loss.item()
            running_bpp_loss += bpp.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.6f}',
                'MSE': f'{mse_loss.item():.6f}',
                'BPP': f'{bpp.item():.4f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to TensorBoard
            global_step = epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/TotalLoss', total_loss.item(), global_step)
            self.writer.add_scalar('Train/MSELoss', mse_loss.item(), global_step)
            self.writer.add_scalar('Train/BPPLoss', bpp.item(), global_step)
            self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], global_step)
        
        avg_loss = running_loss / num_batches
        avg_mse = running_mse_loss / num_batches
        avg_bpp = running_bpp_loss / num_batches
        
        return avg_loss, avg_mse, avg_bpp
    
    def validate(self, epoch):
        """Validation"""
        self.adamix_model.eval()
        self.compressor_model.eval()
        
        val_loss = 0.0
        val_mse_loss = 0.0
        val_bpp_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                
                with autocast():
                    # Stage 1: Wavelet (frozen)
                    wavelet_coeffs = self.wavelet_model(images)
                    
                    # Stage 2: AdaMixNet + Compressor
                    mixed_features = self.adamix_model(wavelet_coeffs)
                    x_hat, likelihoods, y_quantized = self.compressor_model(mixed_features)
                    compressed_features = x_hat
                    
                    # Calculate BPP from likelihoods - FIXED: Use original image dimensions
                    # Original image dimensions for BPP calculation (NOT mixed_features dimensions)
                    batch_size = images.size(0)  # FIXED: Use images, not mixed_features
                    num_pixels = images.size(2) * images.size(3)  # H * W of ORIGINAL images
                    
                    # Rate calculation: -log2(likelihoods) summed over all dimensions
                    log_likelihoods = torch.log(likelihoods.clamp(min=1e-10))
                    total_bits = -log_likelihoods.sum() / math.log(2)
                    bpp = total_bits / (batch_size * num_pixels)
                    
                    # Shape check cho validation
                    if compressed_features.shape != mixed_features.shape:
                        compressed_features = F.interpolate(
                            compressed_features,
                            size=mixed_features.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    # Losses - REMOVED MSE FLOOR to expose real MSE values
                    mse_loss = self.mse_criterion(compressed_features, mixed_features)
                    # NO MSE FLOOR! Let's see the real MSE values
                    total_loss = self.args.lambda_rd * mse_loss + bpp
                
                val_loss += total_loss.item()
                val_mse_loss += mse_loss.item()
                val_bpp_loss += bpp.item()
        
        avg_val_loss = val_loss / num_batches
        avg_val_mse = val_mse_loss / num_batches
        avg_val_bpp = val_bpp_loss / num_batches
        
        # Log to TensorBoard
        self.writer.add_scalar('Val/TotalLoss', avg_val_loss, epoch)
        self.writer.add_scalar('Val/MSELoss', avg_val_mse, epoch)
        self.writer.add_scalar('Val/BPPLoss', avg_val_bpp, epoch)
        
        return avg_val_loss, avg_val_mse, avg_val_bpp
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'adamix_state_dict': self.adamix_model.state_dict(),
            'compressor_state_dict': self.compressor_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'scaler_state_dict': self.scaler.state_dict(),
            'args': self.args
        }
        
        # Save latest checkpoint
        checkpoint_path = f'checkpoints/stage2_compressor_{self.args.dataset}_lambda{self.args.lambda_rd}_latest.pth'
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = f'checkpoints/stage2_compressor_{self.args.dataset}_lambda{self.args.lambda_rd}_best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved với val loss: {loss:.6f}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting Stage 2 Training - Compressor")
        print(f"Dataset: {self.args.dataset}")
        print(f"Device: {self.device}")
        print(f"Lambda RD: {self.args.lambda_rd}")
        
        for epoch in range(self.args.epochs):
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            
            # Train
            train_loss, train_mse, train_bpp = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_mse, val_bpp = self.validate(epoch)
            
            print(f"Train - Loss: {train_loss:.6f}, MSE: {train_mse:.6f}, BPP: {train_bpp:.4f}")
            print(f"Val   - Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, BPP: {val_bpp:.4f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping check
            if epoch > 15 and val_loss > self.best_loss * 1.05:
                print("Early stopping triggered!")
                break
        
        print(f"\nStage 2 Training completed!")
        print(f"Best validation loss: {self.best_loss:.6f}")
        
        # Close TensorBoard writer
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Compressor Training')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, choices=['coco', 'davis'], default='coco',
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='datasets/COCO',
                       help='Dataset directory')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    
    # Stage 1 checkpoint
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                       help='Path to Stage 1 checkpoint')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=40,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--lambda_rd', type=int, choices=[64, 128, 256, 512, 1024, 2048, 4096], 
                       default=128, help='Lambda for rate-distortion tradeoff')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loader workers')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create trainer và run
    trainer = Stage2Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main() 
