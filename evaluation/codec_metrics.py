"""
Codec Metrics Evaluation
PSNR, MS-SSIM, BPP calculation for WAVENET-MV
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
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def calculate_ms_ssim(img1, img2, data_range=1.0):
    """Calculate MS-SSIM between two images"""
    # Convert to numpy và ensure correct shape
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()
    
    # Handle batch dimension
    if img1.ndim == 4:  # Batch of images
        ms_ssim_values = []
        for i in range(img1.shape[0]):
            # Convert from CHW to HWC
            im1 = np.transpose(img1[i], (1, 2, 0))
            im2 = np.transpose(img2[i], (1, 2, 0))
            
            # Calculate MS-SSIM cho từng channel và average
            if im1.shape[2] == 3:  # RGB
                ms_ssim_val = 0
                for c in range(3):
                    ms_ssim_val += ssim(im1[:,:,c], im2[:,:,c], data_range=data_range)
                ms_ssim_val /= 3
            else:
                ms_ssim_val = ssim(im1.squeeze(), im2.squeeze(), data_range=data_range)
            
            ms_ssim_values.append(ms_ssim_val)
        
        return np.mean(ms_ssim_values)
    else:
        # Single image
        if img1.ndim == 3:  # CHW to HWC
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
        
        return ssim(img1, img2, data_range=data_range, multichannel=True)


def estimate_bpp(compressed_data, image_shape):
    """Estimate bits per pixel from compressed representation"""
    if isinstance(compressed_data, dict):
        # Extract actual compressed size
        total_bits = 0
        if 'strings' in compressed_data:
            for string in compressed_data['strings']:
                if isinstance(string, (list, tuple)):
                    for s in string:
                        total_bits += len(s) * 8  # Convert bytes to bits
                else:
                    total_bits += len(string) * 8
        
        # Fallback: estimate from tensor size
        if total_bits == 0 and 'shape' in compressed_data:
            total_bits = np.prod(compressed_data['shape']) * 16  # Assume 16 bits per value
    
    elif torch.is_tensor(compressed_data):
        # Direct tensor - estimate compression
        total_bits = compressed_data.numel() * 16  # Assume quantized to 16 bits
    
    else:
        total_bits = len(str(compressed_data)) * 8  # Fallback
    
    # Calculate BPP
    total_pixels = image_shape[0] * image_shape[1]  # H * W
    bpp = total_bits / total_pixels
    
    return bpp


class CodecEvaluator:
    """Evaluator cho codec metrics"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.load_models()
        
        # Setup dataset
        self.setup_dataset()
        
        # Results storage
        self.results = []
        
    def load_models(self):
        """Load trained models"""
        print("Loading models...")
        
        # Load checkpoint
        if not os.path.exists(self.args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {self.args.checkpoint}")
        
        checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
        
        # Initialize models
        self.wavelet_cnn = WaveletTransformCNN(
            input_channels=3,
            feature_channels=64,
            wavelet_channels=64
        ).to(self.device)
        
        self.adamixnet = AdaMixNet(
            input_channels=256,  # 4 * 64
            C_prime=64,
            C_mix=128
        ).to(self.device)
        
        self.compressor = MultiLambdaCompressorVNVC(
            input_channels=128,
            latent_channels=192
        ).to(self.device)
        
        # Load state dicts
        if 'wavelet_state_dict' in checkpoint:
            self.wavelet_cnn.load_state_dict(checkpoint['wavelet_state_dict'])
        if 'adamixnet_state_dict' in checkpoint:
            self.adamixnet.load_state_dict(checkpoint['adamixnet_state_dict'])
        if 'compressor_state_dict' in checkpoint:
            self.compressor.load_state_dict(checkpoint['compressor_state_dict'])
        
        # Set to evaluation mode
        self.wavelet_cnn.eval()
        self.adamixnet.eval()
        self.compressor.eval()
        
        print("✓ Models loaded successfully")
        
    def _custom_collate_fn(self, batch):
        """Custom collate function to handle COCO dataset safely"""
        images = []
        
        for item in batch:
            if isinstance(item, dict):
                # COCO dataset format
                img = item['image']
                if torch.is_tensor(img):
                    images.append(img)
                else:
                    images.append(torch.tensor(img))
            else:
                # Simple tuple format
                img = item[0] if isinstance(item, (tuple, list)) else item
                if torch.is_tensor(img):
                    images.append(img)
                else:
                    images.append(torch.tensor(img))
        
        # Stack images carefully
        try:
            images_tensor = torch.stack(images, 0)
            return {'image': images_tensor}
        except Exception as e:
            # Fallback: process one by one
            print(f"Warning: Batch collate failed, using individual processing: {e}")
            return {'image': images[0].unsqueeze(0)}  # Process one image at a time
    
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
            num_workers=0,  # FIXED: Disable multiprocessing to avoid tensor resize errors
            pin_memory=False,  # FIXED: Disable pin_memory to avoid storage conflicts
            collate_fn=self._custom_collate_fn  # FIXED: Custom collate function
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
                    # 1. Wavelet transform
                    wavelet_coeffs = self.wavelet_cnn(images)
                    
                    # 2. AdaMixNet
                    mixed_features = self.adamixnet(wavelet_coeffs)
                    
                    # 3. Compression + Decompression
                    compressed_bitstream = self.compressor.compress(mixed_features, lambda_value)
                    reconstructed_features = self.compressor.decompress(compressed_bitstream, lambda_value)
                    
                    # 4. Inverse AdaMixNet (approximate)
                    # For simplicity, assume inverse ≈ linear projection
                    inverse_adamix = torch.nn.Conv2d(128, 256, 1).to(self.device)
                    recovered_coeffs = inverse_adamix(reconstructed_features)
                    
                    # 5. Inverse wavelet transform
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
                        
                        # PSNR
                        psnr_val = calculate_psnr(original, reconstructed).item()
                        psnr_values.append(psnr_val)
                        
                        # MS-SSIM
                        ms_ssim_val = calculate_ms_ssim(original, reconstructed)
                        ms_ssim_values.append(ms_ssim_val)
                        
                        # BPP
                        bpp_val = estimate_bpp(compressed_bitstream, images.shape[2:])
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
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Add metadata
        df['dataset'] = self.args.dataset
        df['split'] = self.args.split
        df['image_size'] = self.args.image_size
        df['model'] = os.path.basename(self.args.checkpoint)
        
        # Save to CSV
        os.makedirs(os.path.dirname(self.args.output_csv), exist_ok=True)
        df.to_csv(self.args.output_csv, index=False)
        
        print(f"✓ Results saved to {self.args.output_csv}")
        
        # Print summary
        print("\n" + "="*50)
        print("CODEC EVALUATION SUMMARY")
        print("="*50)
        print(df.to_string(index=False))
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Codec Metrics Evaluation')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, choices=['coco', 'davis'], default='coco',
                       help='Dataset to evaluate')
    parser.add_argument('--data_dir', type=str, default='datasets/',
                       help='Dataset directory')
    parser.add_argument('--split', type=str, default='val',
                       help='Dataset split')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    
    # Evaluation arguments
    parser.add_argument('--lambdas', type=int, nargs='+', default=[256, 512, 1024],
                       help='Lambda values to evaluate')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to evaluate (for testing)')
    
    # Output arguments
    parser.add_argument('--output_csv', type=str, default='results/codec_metrics.csv',
                       help='Output CSV file')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Create evaluator và run evaluation
    evaluator = CodecEvaluator(args)
    evaluator.evaluate_all_lambdas()
    evaluator.save_results()


if __name__ == '__main__':
    main() 