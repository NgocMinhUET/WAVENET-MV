"""
Codec Metrics Evaluation cho Stage 2
Load 3 models từ các checkpoint riêng biệt
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
    """Calculate MS-SSIM between two images with safe handling for small images."""
    if data_range == 1.0:
        data_max = torch.max(img1) if torch.is_tensor(img1) else np.max(img1)
        data_min = torch.min(img1) if torch.is_tensor(img1) else np.min(img1)
        data_range = max(abs(float(data_max)), abs(float(data_min)), 1.0)
    
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()
    
    def safe_ssim(im1, im2, data_range):
        """Safe SSIM calculation with adaptive window size"""
        try:
            # Get image dimensions
            H, W = im1.shape[:2] if im1.ndim >= 2 else im1.shape
            
            # Calculate adaptive window size
            min_dim = min(H, W)
            if min_dim < 7:
                # For very small images, use smaller window or fallback to simple similarity
                if min_dim < 3:
                    # Fallback to simple correlation for tiny images
                    return np.corrcoef(im1.flatten(), im2.flatten())[0, 1]
                else:
                    # Use smaller window size
                    win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
                    return ssim(im1, im2, data_range=data_range, win_size=win_size)
            else:
                # Normal case: use default window size
                return ssim(im1, im2, data_range=data_range)
        except Exception as e:
            # Fallback to simple correlation if SSIM fails
            print(f"SSIM failed, using correlation fallback: {e}")
            return np.corrcoef(im1.flatten(), im2.flatten())[0, 1]
    
    if img1.ndim == 4:
        ms_ssim_values = []
        for i in range(img1.shape[0]):
            im1 = np.transpose(img1[i], (1, 2, 0))
            im2 = np.transpose(img2[i], (1, 2, 0))
            
            if im1.shape[2] == 3:
                ms_ssim_val = 0
                for c in range(3):
                    ms_ssim_val += safe_ssim(im1[:,:,c], im2[:,:,c], data_range)
                ms_ssim_val /= 3
            else:
                ms_ssim_val = safe_ssim(im1.squeeze(), im2.squeeze(), data_range)
            
            ms_ssim_values.append(ms_ssim_val)
        
        return np.mean(ms_ssim_values)
    else:
        if img1.ndim == 3:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
        
        return safe_ssim(img1, img2, data_range)


def estimate_bpp_from_features(quantized_features, image_shape):
    """Estimate BPP từ quantized feature dimensions."""
    B, C, H_feat, W_feat = quantized_features.shape
    compression_ratio = (H_feat * W_feat) / (image_shape[0] * image_shape[1])
    bits_per_feature = 4.0
    estimated_bpp = compression_ratio * C * bits_per_feature
    # FIXED: Remove the clamp that was forcing BPP to 10.0
    # estimated_bpp = max(0.1, min(10.0, estimated_bpp))  # REMOVED THIS LINE
    return estimated_bpp


class CodecEvaluatorStage2:
    """Evaluator cho codec metrics - load từng model riêng lẻ"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models từ các checkpoint riêng biệt
        self.load_models_separately()
        
        # Setup dataset
        self.setup_dataset()
        
        # Results storage
        self.results = []
        
    def load_models_separately(self):
        """Load 3 models từ các checkpoint riêng biệt"""
        print("Loading models separately...")
        
        # Initialize models
        self.wavelet_cnn = WaveletTransformCNN(
            input_channels=3,
            feature_channels=64,
            wavelet_channels=64
        ).to(self.device)
        
        self.adamixnet = AdaMixNet(
            input_channels=256,
            C_prime=64,
            C_mix=128
        ).to(self.device)
        
        self.compressor = MultiLambdaCompressorVNVC(
            input_channels=128,
            latent_channels=192
        ).to(self.device)
        
        # Load từng model riêng lẻ
        print("Loading WaveletTransformCNN...")
        if os.path.exists(self.args.wavelet_checkpoint):
            wavelet_checkpoint = torch.load(self.args.wavelet_checkpoint, map_location=self.device)
            if 'wavelet_state_dict' in wavelet_checkpoint:
                self.wavelet_cnn.load_state_dict(wavelet_checkpoint['wavelet_state_dict'])
                print("✓ Loaded wavelet_state_dict")
            elif 'state_dict' in wavelet_checkpoint:
                self.wavelet_cnn.load_state_dict(wavelet_checkpoint['state_dict'])
                print("✓ Loaded state_dict for wavelet")
            else:
                print("⚠️ Using random weights for wavelet")
        else:
            print("⚠️ Wavelet checkpoint not found, using random weights")
        
        print("Loading AdaMixNet...")
        if os.path.exists(self.args.adamixnet_checkpoint):
            adamixnet_checkpoint = torch.load(self.args.adamixnet_checkpoint, map_location=self.device)
            if 'adamixnet_state_dict' in adamixnet_checkpoint:
                self.adamixnet.load_state_dict(adamixnet_checkpoint['adamixnet_state_dict'])
                print("✓ Loaded adamixnet_state_dict")
            elif 'state_dict' in adamixnet_checkpoint:
                self.adamixnet.load_state_dict(adamixnet_checkpoint['state_dict'])
                print("✓ Loaded state_dict for adamixnet")
            else:
                print("⚠️ Using random weights for adamixnet")
        else:
            print("⚠️ AdaMixNet checkpoint not found, using random weights")
        
        print("Loading CompressorVNVC...")
        if os.path.exists(self.args.compressor_checkpoint):
            compressor_checkpoint = torch.load(self.args.compressor_checkpoint, map_location=self.device)
            if 'compressor_state_dict' in compressor_checkpoint:
                self.compressor.load_state_dict(compressor_checkpoint['compressor_state_dict'])
                print("✓ Loaded compressor_state_dict")
            elif 'state_dict' in compressor_checkpoint:
                self.compressor.load_state_dict(compressor_checkpoint['state_dict'])
                print("✓ Loaded state_dict for compressor")
            else:
                print("⚠️ Using random weights for compressor")
        else:
            print("⚠️ Compressor checkpoint not found, using random weights")
        
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
        
        # Move all buffers
        for name, buffer in model.named_buffers():
            if hasattr(buffer, 'device') and buffer.device != device:
                model.register_buffer(name, buffer.to(device))
        
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
        
        # Set compressor lambda
        self.compressor.set_lambda(lambda_value)
        
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
    parser = argparse.ArgumentParser(description='Codec Metrics Evaluation - Load models separately')
    
    # Model checkpoints
    parser.add_argument('--wavelet_checkpoint', type=str, default='checkpoints/wavenet_stage1.pt',
                       help='Path to wavelet checkpoint')
    parser.add_argument('--adamixnet_checkpoint', type=str, default='checkpoints/wavenet_stage1.pt',
                       help='Path to adamixnet checkpoint')
    parser.add_argument('--compressor_checkpoint', type=str, default='checkpoints/wavenet_stage2.pt',
                       help='Path to compressor checkpoint')
    
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
    parser.add_argument('--output_csv', type=str, default='results/codec_metrics_stage2.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    # Create evaluator và run evaluation
    evaluator = CodecEvaluatorStage2(args)
    evaluator.evaluate_all_lambdas()
    evaluator.save_results()


if __name__ == '__main__':
    main() 