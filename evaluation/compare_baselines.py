"""
Baseline Comparison Script
So sánh WAVENET-MV với các phương pháp nén ảnh khác:
- JPEG (quality 10, 30, 50, 70, 90)
- WebP (quality 10, 30, 50, 70, 90) 
- PNG (lossless)
- Neural codecs (nếu có)
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.codec_metrics import CodecEvaluator, calculate_psnr, calculate_ms_ssim
from datasets.dataset_loaders import COCODatasetLoader


class BaselineComparator:
    """Compare WAVENET-MV with traditional codecs"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        
        # Setup dataset
        self.setup_dataset()
        
    def setup_dataset(self):
        """Setup evaluation dataset"""
        self.dataset = COCODatasetLoader(
            data_dir=self.args.data_dir,
            subset=self.args.split,
            image_size=self.args.image_size,
            augmentation=False
        )
        
        # Take subset for faster evaluation
        if self.args.max_samples:
            indices = list(range(min(self.args.max_samples, len(self.dataset))))
            self.dataset.image_ids = [self.dataset.image_ids[i] for i in indices]
        
        print(f"✓ Dataset loaded: {len(self.dataset)} images")
    
    def compress_jpeg(self, image_np, quality):
        """Compress image using JPEG"""
        # Convert to PIL Image
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image_np)
        
        # Compress to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        compressed_size = len(buffer.getvalue())
        
        # Decompress
        buffer.seek(0)
        decompressed = Image.open(buffer)
        decompressed_np = np.array(decompressed).astype(np.float32) / 255.0
        
        return decompressed_np, compressed_size
    
    def compress_webp(self, image_np, quality):
        """Compress image using WebP"""
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image_np)
        
        # Compress to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='WebP', quality=quality)
        compressed_size = len(buffer.getvalue())
        
        # Decompress
        buffer.seek(0)
        decompressed = Image.open(buffer)
        decompressed_np = np.array(decompressed).astype(np.float32) / 255.0
        
        return decompressed_np, compressed_size
    
    def compress_png(self, image_np):
        """Compress image using PNG (lossless)"""
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image_np)
        
        # Compress to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        compressed_size = len(buffer.getvalue())
        
        # PNG is lossless, so decompressed = original
        decompressed_np = image_np.astype(np.float32) / 255.0
        
        return decompressed_np, compressed_size
    
    def evaluate_baseline(self, method, quality=None):
        """Evaluate a baseline method"""
        print(f"\n🔄 Evaluating {method}" + (f" (quality={quality})" if quality else ""))
        
        psnr_values = []
        ms_ssim_values = []
        bpp_values = []
        
        for idx in tqdm(range(len(self.dataset)), desc=f"{method}"):
            try:
                # Get original image
                data = self.dataset[idx]
                original = data['image']  # [C, H, W]
                
                # Convert to numpy [H, W, C]
                if torch.is_tensor(original):
                    original_np = original.permute(1, 2, 0).cpu().numpy()
                else:
                    original_np = original
                
                # Denormalize if needed (ImageNet normalization)
                if original_np.min() < 0:  # Likely normalized
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    original_np = original_np * std + mean
                    original_np = np.clip(original_np, 0, 1)
                
                # Compress based on method
                if method == 'JPEG':
                    compressed_np, compressed_size = self.compress_jpeg(original_np, quality)
                elif method == 'WebP':
                    compressed_np, compressed_size = self.compress_webp(original_np, quality)
                elif method == 'PNG':
                    compressed_np, compressed_size = self.compress_png(original_np)
                else:
                    continue
                
                # Calculate metrics
                H, W = original_np.shape[:2]
                
                # PSNR
                psnr_val = calculate_psnr(
                    torch.from_numpy(original_np).permute(2, 0, 1),
                    torch.from_numpy(compressed_np).permute(2, 0, 1)
                ).item()
                psnr_values.append(psnr_val)
                
                # MS-SSIM
                ms_ssim_val = calculate_ms_ssim(original_np, compressed_np)
                ms_ssim_values.append(ms_ssim_val)
                
                # BPP
                total_bits = compressed_size * 8
                bpp = total_bits / (H * W)
                bpp_values.append(bpp)
                
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue
        
        # Calculate averages
        avg_psnr = np.mean(psnr_values) if psnr_values else 0
        avg_ms_ssim = np.mean(ms_ssim_values) if ms_ssim_values else 0
        avg_bpp = np.mean(bpp_values) if bpp_values else 0
        
        result = {
            'method': method,
            'quality': quality,
            'psnr_db': avg_psnr,
            'ms_ssim': avg_ms_ssim,
            'bpp': avg_bpp,
            'num_samples': len(psnr_values)
        }
        
        self.results.append(result)
        print(f"{method}: PSNR={avg_psnr:.2f}dB, MS-SSIM={avg_ms_ssim:.4f}, BPP={avg_bpp:.4f}")
        
        return result
    
    def evaluate_wavenet_mv(self):
        """Evaluate WAVENET-MV using existing codec_metrics"""
        print(f"\n🔄 Evaluating WAVENET-MV")
        
        # Use existing evaluator
        evaluator_args = argparse.Namespace(
            checkpoint=self.args.wavenet_checkpoint,
            dataset=self.args.dataset,
            data_dir=self.args.data_dir,
            split=self.args.split,
            image_size=self.args.image_size,
            lambdas=self.args.lambdas,
            batch_size=self.args.batch_size,
            max_samples=self.args.max_samples,
            output_csv='temp_wavenet_results.csv',
            num_workers=2,
            skip_entropy_update=True
        )
        
        evaluator = CodecEvaluator(evaluator_args)
        evaluator.evaluate_all_lambdas()
        
        # Add WAVENET-MV results
        for result in evaluator.results:
            wavenet_result = {
                'method': 'WAVENET-MV',
                'quality': result['lambda'],
                'psnr_db': result['psnr_db'],
                'ms_ssim': result['ms_ssim'],
                'bpp': result['bpp'],
                'num_samples': result['num_samples']
            }
            self.results.append(wavenet_result)
            print(f"WAVENET-MV (λ={result['lambda']}): PSNR={result['psnr_db']:.2f}dB, "
                  f"MS-SSIM={result['ms_ssim']:.4f}, BPP={result['bpp']:.4f}")
    
    def run_comparison(self):
        """Run full comparison"""
        print("🚀 Starting baseline comparison...")
        
        # Evaluate traditional codecs based on command line arguments
        if hasattr(self.args, 'methods') and self.args.methods:
            methods = self.args.methods
        else:
            methods = ['JPEG', 'WebP', 'PNG']
        
        if hasattr(self.args, 'qualities') and self.args.qualities:
            qualities = self.args.qualities
        else:
            qualities = [10, 30, 50, 70, 90]
        
        print(f"Methods: {methods}")
        print(f"Qualities: {qualities}")
        
        for method in methods:
            if method == 'PNG':
                # PNG is lossless, no quality parameter
                self.evaluate_baseline('PNG')
            else:
                # JPEG and WebP use quality parameter
                for quality in qualities:
                    self.evaluate_baseline(method, quality)
        
        # Evaluate WAVENET-MV
        if self.args.wavenet_checkpoint:
            self.evaluate_wavenet_mv()
    
    def save_results(self):
        """Save comparison results"""
        if not self.results:
            print("No results to save!")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Add metadata
        df['dataset'] = self.args.dataset
        df['split'] = self.args.split
        df['image_size'] = self.args.image_size
        
        # Save to CSV
        os.makedirs(os.path.dirname(self.args.output_csv), exist_ok=True)
        df.to_csv(self.args.output_csv, index=False)
        
        print(f"✓ Comparison results saved to {self.args.output_csv}")
        
        # Print summary
        print("\n" + "="*70)
        print("BASELINE COMPARISON SUMMARY")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
    
    def plot_rd_curves(self):
        """Plot Rate-Distortion curves"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PSNR vs BPP
        methods = df['method'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        
        for method, color in zip(methods, colors):
            method_data = df[df['method'] == method].sort_values('bpp')
            ax1.plot(method_data['bpp'], method_data['psnr_db'], 
                    'o-', label=method, color=color, linewidth=2, markersize=6)
        
        ax1.set_xlabel('BPP (bits per pixel)')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('Rate-Distortion: PSNR vs BPP')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MS-SSIM vs BPP
        for method, color in zip(methods, colors):
            method_data = df[df['method'] == method].sort_values('bpp')
            ax2.plot(method_data['bpp'], method_data['ms_ssim'], 
                    'o-', label=method, color=color, linewidth=2, markersize=6)
        
        ax2.set_xlabel('BPP (bits per pixel)')
        ax2.set_ylabel('MS-SSIM')
        ax2.set_title('Rate-Distortion: MS-SSIM vs BPP')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.args.output_csv.replace('.csv', '_rd_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ R-D curves saved to {plot_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Baseline Comparison')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='coco',
                       help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='datasets/COCO',
                       help='Dataset directory')
    parser.add_argument('--split', type=str, default='val',
                       help='Dataset split')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum samples for evaluation')
    
    # Baseline methods arguments
    parser.add_argument('--methods', type=str, nargs='+', 
                       choices=['JPEG', 'WebP', 'PNG'],
                       default=['JPEG', 'WebP', 'PNG'],
                       help='Methods to compare')
    parser.add_argument('--qualities', type=int, nargs='+', 
                       default=[10, 30, 50, 70, 90],
                       help='Quality values for JPEG/WebP')
    
    # WAVENET-MV arguments
    parser.add_argument('--wavenet_checkpoint', type=str,
                       help='WAVENET-MV checkpoint path')
    parser.add_argument('--lambdas', type=int, nargs='+', default=[256, 512, 1024],
                       help='Lambda values for WAVENET-MV')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for WAVENET-MV evaluation')
    
    # Output arguments
    parser.add_argument('--output_csv', type=str, default='results/baseline_comparison.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    # Run comparison
    comparator = BaselineComparator(args)
    comparator.run_comparison()
    comparator.save_results()
    comparator.plot_rd_curves()


if __name__ == '__main__':
    main() 