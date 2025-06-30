"""
Stage 1 Training: Pre-train WaveletTransformCNN
- Epochs: 30
- Trainable: WaveletCNN only
- Loss: L2 reconstruction loss
- Optimizer: Adam, LR=2e-4 → cosine decay, batch=8, seed=42
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
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
from datasets.dataset_loaders import COCODatasetLoader, DAVISDatasetLoader


def stage1_collate_fn(batch):
    """
    Custom collate function cho Stage 1 - chỉ lấy images
    Bỏ qua annotations để tránh tensor size mismatch
    """
    images = []
    for item in batch:
        if isinstance(item, dict):
            images.append(item['image'])
        else:
            images.append(item[0])  # Tuple format
    
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


class WaveletTrainer:
    """Trainer cho Stage 1 - Wavelet pre-training"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set seed
        set_seed(args.seed)
        
        # Initialize model
        self.model = WaveletTransformCNN(
            input_channels=3,
            feature_channels=64,
            wavelet_channels=64
        ).to(self.device)
        
        # Loss function - L2 reconstruction
        self.criterion = nn.MSELoss()
        
        # Optimizer - Adam với LR=2e-4
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
        # Setup datasets
        self.setup_datasets()
        
        # Setup scheduler
        num_training_steps = len(self.train_loader) * args.epochs
        num_warmup_steps = len(self.train_loader) * 2  # 2 epochs warmup
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=f'runs/stage1_wavelet_{args.dataset}')
        
        # Best model tracking
        self.best_loss = float('inf')
        
    def setup_datasets(self):
        """Setup train và validation datasets"""
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
            collate_fn=stage1_collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset_loader,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=stage1_collate_fn
        )
        
    def train_epoch(self, epoch):
        """Train một epoch"""
        self.model.train()
        
        running_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get images (chỉ cần images cho reconstruction)
            if isinstance(batch, dict):
                images = batch['image'].to(self.device)
            else:
                images = batch[0].to(self.device)  # Tuple format
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass với mixed precision
            with autocast():
                # Wavelet forward
                wavelet_coeffs = self.model(images)
                
                # Reconstruction
                reconstructed = self.model.inverse_transform(wavelet_coeffs)
                
                # L2 reconstruction loss
                loss = self.criterion(reconstructed, images)
            
            # Backward pass với mixed precision
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update scheduler
            self.scheduler.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{running_loss / (batch_idx + 1):.6f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to TensorBoard
            global_step = epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], global_step)
        
        avg_loss = running_loss / num_batches
        return avg_loss
    
    def validate(self, epoch):
        """Validation"""
        self.model.eval()
        
        val_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Get images
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                else:
                    images = batch[0].to(self.device)
                
                # Forward pass
                with autocast():
                    wavelet_coeffs = self.model(images)
                    reconstructed = self.model.inverse_transform(wavelet_coeffs)
                    loss = self.criterion(reconstructed, images)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / num_batches
        
        # Log to TensorBoard
        self.writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'scaler_state_dict': self.scaler.state_dict(),
            'args': self.args
        }
        
        # Save latest checkpoint
        checkpoint_path = f'checkpoints/stage1_wavelet_{self.args.dataset}_latest.pth'
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = f'checkpoints/stage1_wavelet_{self.args.dataset}_best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved với val loss: {loss:.6f}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting Stage 1 Training - Wavelet Pre-training")
        print(f"Dataset: {self.args.dataset}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.args.epochs):
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping check
            if epoch > 10 and val_loss > self.best_loss * 1.1:
                print("Early stopping triggered!")
                break
        
        print(f"\nStage 1 Training completed!")
        print(f"Best validation loss: {self.best_loss:.6f}")
        
        # Close TensorBoard writer
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Wavelet Pre-training')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, choices=['coco', 'davis'], default='coco',
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='datasets/COCO',
                       help='Dataset directory')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loader workers')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = WaveletTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main() 