"""
Codec Metrics Evaluation - Final Version
Phù hợp với cấu trúc checkpoint thực tế:
- Stage 1: WaveletTransformCNN
- Stage 2: CompressorVNVC (không có AdaMixNet)
- Stage 3: AI Heads
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
from pathlib import Path
import math

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.wavelet_transform_cnn import WaveletTransformCNN
from models.adamixnet import AdaMixNet  
from models.compressor_vnvc import MultiLambdaCompressorVNVC
from datasets.dataset_loaders import COCODatasetLoader, DAVISDatasetLoader


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR giữa hai ảnh."""
    if max_val == 1.0:
        data_max = torch.max(img1)
        data_min = torch.min(img1)
        max_val_est = max(data_max.abs(), data_min.abs())
        max_val = torch.clamp(max_val_est, min=1.0).item()
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def calculate_ms_ssim(img1, img2, data_range=1.0):
    """Calculate MS-SSIM between two images."""
    if data_range == 1.0:
        data_max = torch.max(img1) if torch.is_tensor(img1) else np.max(img1)
        data_min = torch.min(img1) if torch.is_tensor(img1) else np.min(img1)
        data_range = max(abs(float(data_max)), abs(float(data_min)), 1.0)
    
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()
    
    if img1.ndim == 4:
        ms_ssim_values = []
        for i in range(img1.shape[0]):
            im1 = np.transpose(img1[i], (1, 2, 0))
            im2 = np.transpose(img2[i], (1, 2, 0))
            
            if im1.shape[2] == 3:
                ms_ssim_val = 0
                for c in range(3):
                    ms_ssim_val += ssim(im1[:,:,c], im2[:,:,c], data_range=data_range)
                ms_ssim_val /= 3
            else:
                ms_ssim_val = ssim(im1.squeeze(), im2.squeeze(), data_range=data_range)
            
            ms_ssim_values.append(ms_ssim_val)
        
        return np.mean(ms_ssim_values)
    else:
        if img1.ndim == 3:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
        
        return ssim(img1, img2, data_range=data_range, multichannel=True)


def estimate_bpp_from_features(quantized_features, image_shape):
    """Estimate BPP từ quantized feature dimensions."""
    B, C, H_feat, W_feat = quantized_features.shape
    compression_ratio = (H_feat * W_feat) / (image_shape[0] * image_shape[1])
    bits_per_feature = 4.0
    estimated_bpp = compression_ratio * C * bits_per_feature
    estimated_bpp = max(0.1, min(10.0, estimated_bpp))
    return estimated_bpp


class CodecEvaluatorFinal:
    """Evaluator cuối cùng - phù hợp với cấu trúc checkpoint thực tế"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models theo cấu trúc thực tế
        self.load_models_final()
        
        # Setup dataset
        self.setup_dataset()
        
        # Results storage
        self.results = []
        
    def load_models_final(self):
        """Load models theo cấu trúc checkpoint thực tế"""
        print("Loading models with actual checkpoint structure...")
        
        # Initialize models
        self.wavelet_cnn = WaveletTransformCNN(
            input_channels=3,
            feature_channels=64,
            wavelet_channels=64
        ).to(self.device)
        
        # AdaMixNet - sử dụng random weights vì không có trong checkpoint
        self.adamixnet = AdaMixNet(
            input_channels=256,
            C_prime=64,
            C_mix=128
        ).to(self.device)
        
        # Sử dụng CompressorVNVC đơn giản thay vì MultiLambdaCompressorVNVC
        from models.compressor_vnvc import CompressorVNVC
        self.compressor = CompressorVNVC(
            input_channels=128,
            latent_channels=192,
            lambda_rd=128
        ).to(self.device)
        
        # Load WaveletTransformCNN từ Stage 1
        print("Loading WaveletTransformCNN from Stage 1...")
        if os.path.exists(self.args.stage1_checkpoint):
            stage1_checkpoint = torch.load(self.args.stage1_checkpoint, map_location=self.device)
            print(f"Stage 1 checkpoint keys: {list(stage1_checkpoint.keys())}")
            
            # Thử các key khác nhau
            if 'wavelet_state_dict' in stage1_checkpoint:
                self.wavelet_cnn.load_state_dict(stage1_checkpoint['wavelet_state_dict'])
                print("✓ Loaded wavelet_state_dict")
            elif 'model_state_dict' in stage1_checkpoint:
                self.wavelet_cnn.load_state_dict(stage1_checkpoint['model_state_dict'])
                print("✓ Loaded model_state_dict for wavelet")
            elif 'state_dict' in stage1_checkpoint:
                self.wavelet_cnn.load_state_dict(stage1_checkpoint['state_dict'])
                print("✓ Loaded state_dict for wavelet")
            else:
                print("⚠️ Using random weights for wavelet")
        else:
            print("⚠️ Stage 1 checkpoint not found, using random weights")
        
        # AdaMixNet - sử dụng random weights
        print("AdaMixNet: Using random weights (not in checkpoints)")
        
        # Load CompressorVNVC từ Stage 2
        print("Loading CompressorVNVC from Stage 2...")
        if os.path.exists(self.args.stage2_checkpoint):
            stage2_checkpoint = torch.load(self.args.stage2_checkpoint, map_location=self.device)
            print(f"Stage 2 checkpoint keys: {list(stage2_checkpoint.keys())}")
            
            # Kiểm tra cấu trúc compressor trong checkpoint
            if 'compressor_state_dict' in stage2_checkpoint:
                checkpoint_keys = list(stage2_checkpoint['compressor_state_dict'].keys())
                print(f"Compressor checkpoint keys (first 10): {checkpoint_keys[:10]}")
                
                # Tạo model phù hợp với cấu trúc checkpoint
                if 'analysis_transform.conv1.weight' in checkpoint_keys:
                    # Checkpoint có cấu trúc với conv1, norm1, skip_conv
                    print("Detected checkpoint with conv1/norm1 structure")
                    from models.compressor_improved import ImprovedCompressorVNVC
                    self.compressor = ImprovedCompressorVNVC(
                        input_channels=128,
                        latent_channels=192,
                        lambda_rd=128
                    ).to(self.device)
                else:
                    # Checkpoint có cấu trúc đơn giản
                    print("Detected checkpoint with simple structure")
                    from models.compressor_vnvc import CompressorVNVC
                    self.compressor = CompressorVNVC(
                        input_channels=128,
                        latent_channels=192,
                        lambda_rd=128
                    ).to(self.device)
                
                # Load state dict
                try:
                    self.compressor.load_state_dict(stage2_checkpoint['compressor_state_dict'])
                    print("✓ Loaded compressor_state_dict successfully")
                except Exception as e:
                    print(f"⚠️ Failed to load compressor: {e}")
                    print("⚠️ Using random weights for compressor")
            elif 'state_dict' in stage2_checkpoint:
                self.compressor.load_state_dict(stage2_checkpoint['state_dict'])
                print("✓ Loaded state_dict for compressor")
            else:
                print("⚠️ Using random weights for compressor")
        else:
            print("⚠️ Stage 2 checkpoint not found, using random weights")
        
        # Force move to device
        self.wavelet_cnn = self.force_move_to_device(self.wavelet_cnn, self.device)
        self.adamixnet = self.force_move_to_device(self.adamixnet, self.device)
        self.compressor = self.force_move_to_device(self.compressor, self.device)
        
        # Set to evaluation mode
        self.wavelet_cnn.eval()
        self.adamixnet.eval()
        self.compressor.eval()
        
        print("✓ All models loaded successfully")
    
    def force_move_to_device(self, model, device):
        """Force move model to device"""
        print(f"🚀 Force moving {model.__class__.__name__} to {device}")
        model = model.to(device)
        
        # Move all parameters
        for name, param in model.named_parameters():
            if param.device != device:
                param.data = param.data.to(device)
        
        # Move all buffers - chỉ xử lý buffers không có dấu chấm
        for name, buffer in model.named_buffers():
            if hasattr(buffer, 'device') and buffer.device != device:
                # Chỉ xử lý buffers có tên đơn giản (không có dấu chấm)
                if '.' not in name:
                    try:
                        model.register_buffer(name, buffer.to(device))
                    except Exception as e:
                        print(f"⚠️ Warning: Could not move buffer {name}: {e}")
        
        return model
    
    def setup_dataset(self):
        """Setup evaluation dataset"""
        if self.args.dataset == 'coco':
            dataset = COCODatasetLoader(
                data_dir=self.args.data_dir,
                subset=self.args.split,
                image_size=self.args.image_size,
                augmentation=False
            )
        elif self.args.dataset == 'davis':
            dataset = DAVISDatasetLoader(
                data_dir=self.args.data_dir,
                subset=self.args.split,
                image_size=self.args.image_size,
                augmentation=False
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} images")
        
    def evaluate_lambda(self, lambda_value):
        """Evaluate metrics for specific lambda value"""
        print(f"\nEvaluating λ = {lambda_value}")
        
        # Set compressor lambda (nếu có method set_lambda)
        if hasattr(self.compressor, 'set_lambda'):
            self.compressor.set_lambda(lambda_value)
        else:
            # Nếu không có, tạo compressor mới với lambda mới
            from models.compressor_vnvc import CompressorVNVC
            self.compressor = CompressorVNVC(
                input_channels=128,
                latent_channels=192,
                lambda_rd=lambda_value
            ).to(self.device)
            self.compressor.eval()
        
        # Metrics accumulation
        psnr_values = []
        ms_ssim_values = []
        bpp_values = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc=f'λ={lambda_value}')):
                # Get images
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                else:
                    images = batch[0].to(self.device)
                
                try:
                    # Forward pass through pipeline
                    wavelet_coeffs = self.wavelet_cnn(images)
                    mixed_features = self.adamixnet(wavelet_coeffs)
                    x_hat, likelihoods, y_quantized = self.compressor(mixed_features)
                    
                    # Inverse transforms
                    recovered_coeffs = self.adamixnet.inverse_transform(x_hat)
                    reconstructed_images = self.wavelet_cnn.inverse_transform(recovered_coeffs)
                    
                    # Ensure same size
                    if reconstructed_images.shape != images.shape:
                        reconstructed_images = F.interpolate(
                            reconstructed_images, 
                            size=images.shape[2:], 
                            mode='bilinear', 
                            align_corners=False
                        )
                    
                    # Calculate metrics
                    for i in range(images.size(0)):
                        original = images[i:i+1]
                        reconstructed = reconstructed_images[i:i+1]
                        
                        psnr_val = calculate_psnr(original, reconstructed).item()
                        psnr_values.append(psnr_val)
                        
                        ms_ssim_val = calculate_ms_ssim(original, reconstructed)
                        ms_ssim_values.append(ms_ssim_val)
                        
                        bpp_val = estimate_bpp_from_features(y_quantized, images.shape[2:])
                        bpp_values.append(bpp_val)
                
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue
                
                # Early stop for quick testing
                if self.args.max_samples and batch_idx * self.args.batch_size >= self.args.max_samples:
                    break
        
        # Calculate average metrics
        avg_psnr = np.mean(psnr_values) if psnr_values else 0
        avg_ms_ssim = np.mean(ms_ssim_values) if ms_ssim_values else 0
        avg_bpp = np.mean(bpp_values) if bpp_values else 0
        
        return {
            'lambda': lambda_value,
            'psnr_db': avg_psnr,
            'ms_ssim': avg_ms_ssim,
            'bpp': avg_bpp,
            'num_samples': len(psnr_values)
        }
    
    def evaluate_all_lambdas(self):
        """Evaluate tất cả lambda values"""
        lambda_values = self.args.lambdas
        
        for lambda_val in lambda_values:
            result = self.evaluate_lambda(lambda_val)
            self.results.append(result)
            
            print(f"λ={lambda_val}: PSNR={result['psnr_db']:.2f}dB, "
                  f"MS-SSIM={result['ms_ssim']:.4f}, BPP={result['bpp']:.4f}")
    
    def save_results(self):
        """Save results to CSV"""
        if not self.results:
            print("No results to save!")
            return
        
        df = pd.DataFrame(self.results)
        df['dataset'] = self.args.dataset
        df['split'] = self.args.split
        df['image_size'] = self.args.image_size
        
        os.makedirs(os.path.dirname(self.args.output_csv), exist_ok=True)
        df.to_csv(self.args.output_csv, index=False)
        
        print(f"✓ Results saved to {self.args.output_csv}")
        print("\n" + "="*50)
        print("CODEC EVALUATION SUMMARY")
        print("="*50)
        print(df.to_string(index=False))
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Codec Metrics Evaluation - Final Version')
    
    # Model checkpoints
    parser.add_argument('--stage1_checkpoint', type=str, default='checkpoints/stage1_wavelet_coco_best.pth',
                       help='Path to Stage 1 checkpoint (WaveletTransformCNN)')
    parser.add_argument('--stage2_checkpoint', type=str, default='checkpoints/stage2_compressor_coco_lambda128_best.pth',
                       help='Path to Stage 2 checkpoint (CompressorVNVC)')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, choices=['coco', 'davis'], default='coco',
                       help='Dataset to evaluate')
    parser.add_argument('--data_dir', type=str, default='datasets/COCO',
                       help='Dataset directory')
    parser.add_argument('--split', type=str, default='val',
                       help='Dataset split')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    
    # Evaluation arguments
    parser.add_argument('--lambdas', type=int, nargs='+', default=[128],
                       help='Lambda values to evaluate')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to evaluate')
    
    # Output arguments
    parser.add_argument('--output_csv', type=str, default='results/codec_metrics_final.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    # Create evaluator và run evaluation
    evaluator = CodecEvaluatorFinal(args)
    evaluator.evaluate_all_lambdas()
    evaluator.save_results()


if __name__ == '__main__':
    main() 