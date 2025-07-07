"""
Comprehensive Comparison Script
So sánh toàn diện WAVENET-MV với các phương pháp nén ảnh khác:
- JPEG (quality levels)
- WebP (quality levels) 
- PNG (lossless)
- WAVENET-MV (multiple lambda values)
- Analysis with/without Wavelet CNN
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
import warnings
warnings.filterwarnings('ignore')

# Fix OpenMP warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.codec_metrics import CodecEvaluator, calculate_psnr, calculate_ms_ssim, estimate_bpp_from_features
from evaluation.compare_baselines import BaselineComparator
from datasets.dataset_loaders import COCODatasetLoader
from models.wavelet_transform_cnn import WaveletTransformCNN
from models.adamixnet import AdaMixNet
from models.compressor_vnvc import MultiLambdaCompressorVNVC
from models.ai_heads import YOLOTinyHead, SegFormerLiteHead


class ComprehensiveEvaluator:
    """Comprehensive evaluation of WAVENET-MV vs baselines"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        
        # Setup dataset
        self.setup_dataset()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
    def setup_dataset(self):
        """Setup evaluation dataset"""
        try:
            self.dataset = COCODatasetLoader(
                data_dir=self.args.data_dir,
                subset='val',
                image_size=self.args.image_size,
                augmentation=False
            )
            
            # Take subset for faster evaluation
            if self.args.max_samples:
                indices = list(range(min(self.args.max_samples, len(self.dataset))))
                if hasattr(self.dataset, 'image_ids'):
                    self.dataset.image_ids = [self.dataset.image_ids[i] for i in indices]
                
            print(f"✓ Dataset loaded: {len(self.dataset)} images")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Fallback to mock dataset
            self.dataset = self.create_mock_dataset()
    
    def create_mock_dataset(self):
        """Create mock dataset for testing"""
        print("Creating mock dataset...")
        
        class MockDataset:
            def __init__(self, num_samples=10, image_size=256):
                self.num_samples = num_samples
                self.image_size = image_size
                
            def __len__(self):
                return self.num_samples
                
            def __getitem__(self, idx):
                # Generate random image
                image = torch.randn(3, self.image_size, self.image_size) * 0.5 + 0.5
                return {'image': image}
                
        return MockDataset(self.args.max_samples or 10, self.args.image_size)
    
    def load_wavenet_models(self):
        """Load WAVENET-MV models"""
        print("Loading WAVENET-MV models...")
        
        # WaveletTransformCNN
        self.wavelet_model = WaveletTransformCNN(
            input_channels=3,
            feature_channels=64,
            wavelet_channels=64
        ).to(self.device)
        
        # AdaMixNet
        self.adamix_model = AdaMixNet(
            input_channels=256,  # 4*64
            C_prime=64,
            C_mix=128,
            N=4
        ).to(self.device)
        
        # MultiLambdaCompressorVNVC
        self.compressor_model = MultiLambdaCompressorVNVC(
            input_channels=128,
            latent_channels=192
        ).to(self.device)
        
        # Load checkpoint if exists
        if os.path.exists(self.args.checkpoint):
            print(f"Loading checkpoint: {self.args.checkpoint}")
            checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
            
            # Try to load state dicts
            try:
                if 'wavelet_state_dict' in checkpoint:
                    self.wavelet_model.load_state_dict(checkpoint['wavelet_state_dict'])
                if 'adamix_state_dict' in checkpoint:
                    self.adamix_model.load_state_dict(checkpoint['adamix_state_dict'])
                if 'compressor_state_dict' in checkpoint:
                    self.compressor_model.load_state_dict(checkpoint['compressor_state_dict'])
                print("✓ Checkpoint loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Using randomly initialized models")
        else:
            print("No checkpoint found, using randomly initialized models")
        
        # Set to eval mode
        self.wavelet_model.eval()
        self.adamix_model.eval()
        self.compressor_model.eval()
    
    def evaluate_wavenet_mv(self, lambda_values=[64, 128, 256, 512, 1024]):
        """Evaluate WAVENET-MV at different lambda values"""
        print(f"\n🔄 Evaluating WAVENET-MV (λ={lambda_values})")
        
        self.load_wavenet_models()
        
        for lambda_val in lambda_values:
            print(f"\n--- Evaluating λ={lambda_val} ---")
            
            self.compressor_model.set_lambda(lambda_val)
            
            psnr_values = []
            ms_ssim_values = []
            bpp_values = []
            
            with torch.no_grad():
                for idx in tqdm(range(len(self.dataset)), desc=f"λ={lambda_val}"):
                    try:
                        # Get original image
                        data = self.dataset[idx]
                        original = data['image']  # [C, H, W]
                        
                        if torch.is_tensor(original):
                            original = original.to(self.device)
                        else:
                            original = torch.from_numpy(original).to(self.device)
                        
                        # Add batch dimension
                        if original.dim() == 3:
                            original = original.unsqueeze(0)
                        
                        # Forward pass
                        # Stage 1: Wavelet transform
                        wavelet_coeffs = self.wavelet_model(original)
                        
                        # Stage 2: Adaptive mixing
                        mixed_features = self.adamix_model(wavelet_coeffs)
                        
                        # Stage 3: Compression
                        compressed_features, likelihoods, quantized = self.compressor_model(mixed_features)
                        
                        # Reconstruction path: compressed -> mixed -> wavelet -> image
                        reconstructed_mixed = compressed_features
                        reconstructed_wavelet = self.adamix_model.inverse_transform(reconstructed_mixed)
                        reconstructed_image = self.wavelet_model.inverse_transform(reconstructed_wavelet)
                        
                        # Calculate metrics
                        H, W = original.shape[-2:]
                        
                        # PSNR
                        psnr_val = calculate_psnr(original, reconstructed_image)
                        psnr_values.append(psnr_val.item())
                        
                        # MS-SSIM
                        ms_ssim_val = calculate_ms_ssim(
                            original.cpu().numpy(),
                            reconstructed_image.cpu().numpy()
                        )
                        ms_ssim_values.append(ms_ssim_val)
                        
                        # BPP
                        bpp = estimate_bpp_from_features(quantized, (H, W))
                        bpp_values.append(bpp)
                        
                    except Exception as e:
                        print(f"Error processing image {idx}: {e}")
                        continue
            
            # Calculate averages
            avg_psnr = np.mean(psnr_values) if psnr_values else 0
            avg_ms_ssim = np.mean(ms_ssim_values) if ms_ssim_values else 0
            avg_bpp = np.mean(bpp_values) if bpp_values else 0
            
            result = {
                'method': 'WAVENET-MV',
                'lambda': lambda_val,
                'psnr_db': avg_psnr,
                'ms_ssim': avg_ms_ssim,
                'bpp': avg_bpp,
                'num_samples': len(psnr_values)
            }
            
            self.results.append(result)
            print(f"λ={lambda_val}: PSNR={avg_psnr:.2f}dB, MS-SSIM={avg_ms_ssim:.4f}, BPP={avg_bpp:.4f}")
    
    def evaluate_without_wavelet(self, lambda_values=[256, 512, 1024]):
        """Evaluate system without Wavelet CNN (direct compression)"""
        print(f"\n🔄 Evaluating without Wavelet CNN")
        
        # Load models (without wavelet)
        self.adamix_model = AdaMixNet(
            input_channels=192,  # Direct from RGB features
            C_prime=64,
            C_mix=128,
            N=4
        ).to(self.device)
        
        self.compressor_model = MultiLambdaCompressorVNVC(
            input_channels=128,
            latent_channels=192
        ).to(self.device)
        
        # Simple feature extractor to replace wavelet
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 192, 3, padding=1)
        ).to(self.device)
        
        self.adamix_model.eval()
        self.compressor_model.eval()
        self.feature_extractor.eval()
        
        for lambda_val in lambda_values:
            print(f"\n--- Evaluating without Wavelet λ={lambda_val} ---")
            
            self.compressor_model.set_lambda(lambda_val)
            
            psnr_values = []
            ms_ssim_values = []
            bpp_values = []
            
            with torch.no_grad():
                for idx in tqdm(range(len(self.dataset)), desc=f"No-Wavelet λ={lambda_val}"):
                    try:
                        # Get original image
                        data = self.dataset[idx]
                        original = data['image']  # [C, H, W]
                        
                        if torch.is_tensor(original):
                            original = original.to(self.device)
                        else:
                            original = torch.from_numpy(original).to(self.device)
                        
                        # Add batch dimension
                        if original.dim() == 3:
                            original = original.unsqueeze(0)
                        
                        # Forward pass without wavelet
                        # Direct feature extraction
                        features = self.feature_extractor(original)
                        
                        # Adaptive mixing
                        mixed_features = self.adamix_model(features)
                        
                        # Compression
                        compressed_features, likelihoods, quantized = self.compressor_model(mixed_features)
                        
                        # Simple reconstruction (just upsampling)
                        reconstructed_image = torch.nn.functional.interpolate(
                            compressed_features, 
                            size=original.shape[-2:], 
                            mode='bilinear', 
                            align_corners=False
                        )
                        
                        # Add simple conv to get back to RGB
                        if not hasattr(self, 'rgb_decoder'):
                            self.rgb_decoder = torch.nn.Conv2d(128, 3, 3, padding=1).to(self.device)
                        
                        reconstructed_image = torch.tanh(self.rgb_decoder(reconstructed_image))
                        
                        # Calculate metrics
                        H, W = original.shape[-2:]
                        
                        # PSNR
                        psnr_val = calculate_psnr(original, reconstructed_image)
                        psnr_values.append(psnr_val.item())
                        
                        # MS-SSIM
                        ms_ssim_val = calculate_ms_ssim(
                            original.cpu().numpy(),
                            reconstructed_image.cpu().numpy()
                        )
                        ms_ssim_values.append(ms_ssim_val)
                        
                        # BPP
                        bpp = estimate_bpp_from_features(quantized, (H, W))
                        bpp_values.append(bpp)
                        
                    except Exception as e:
                        print(f"Error processing image {idx}: {e}")
                        continue
            
            # Calculate averages
            avg_psnr = np.mean(psnr_values) if psnr_values else 0
            avg_ms_ssim = np.mean(ms_ssim_values) if ms_ssim_values else 0
            avg_bpp = np.mean(bpp_values) if bpp_values else 0
            
            result = {
                'method': 'WAVENET-MV (No Wavelet)',
                'lambda': lambda_val,
                'psnr_db': avg_psnr,
                'ms_ssim': avg_ms_ssim,
                'bpp': avg_bpp,
                'num_samples': len(psnr_values)
            }
            
            self.results.append(result)
            print(f"No-Wavelet λ={lambda_val}: PSNR={avg_psnr:.2f}dB, MS-SSIM={avg_ms_ssim:.4f}, BPP={avg_bpp:.4f}")
    
    def evaluate_baselines(self):
        """Evaluate traditional codec baselines"""
        print(f"\n🔄 Evaluating baseline methods")
        
        # JPEG qualities
        jpeg_qualities = [10, 30, 50, 70, 90]
        for quality in jpeg_qualities:
            self.evaluate_codec_baseline('JPEG', quality)
        
        # WebP qualities
        webp_qualities = [10, 30, 50, 70, 90]
        for quality in webp_qualities:
            self.evaluate_codec_baseline('WebP', quality)
        
        # PNG (lossless)
        self.evaluate_codec_baseline('PNG', None)
    
    def evaluate_codec_baseline(self, codec_type, quality):
        """Evaluate a specific codec baseline"""
        print(f"\n--- Evaluating {codec_type}" + (f" (Q={quality})" if quality else "") + " ---")
        
        psnr_values = []
        ms_ssim_values = []
        bpp_values = []
        
        for idx in tqdm(range(len(self.dataset)), desc=f"{codec_type}"):
            try:
                # Get original image
                data = self.dataset[idx]
                original = data['image']  # [C, H, W]
                
                # Convert to numpy [H, W, C]
                if torch.is_tensor(original):
                    original_np = original.permute(1, 2, 0).cpu().numpy()
                else:
                    original_np = original
                
                # Denormalize if needed
                if original_np.min() < 0:
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    original_np = original_np * std + mean
                    original_np = np.clip(original_np, 0, 1)
                
                # Compress based on codec
                if codec_type == 'JPEG':
                    compressed_np, compressed_size = self.compress_jpeg(original_np, quality)
                elif codec_type == 'WebP':
                    compressed_np, compressed_size = self.compress_webp(original_np, quality)
                elif codec_type == 'PNG':
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
            'method': codec_type,
            'quality': quality,
            'psnr_db': avg_psnr,
            'ms_ssim': avg_ms_ssim,
            'bpp': avg_bpp,
            'num_samples': len(psnr_values)
        }
        
        self.results.append(result)
        print(f"{codec_type}: PSNR={avg_psnr:.2f}dB, MS-SSIM={avg_ms_ssim:.4f}, BPP={avg_bpp:.4f}")
    
    def compress_jpeg(self, image_np, quality):
        """Compress image using JPEG"""
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
        """Compress image using PNG"""
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image_np)
        
        # Compress to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        compressed_size = len(buffer.getvalue())
        
        # PNG is lossless
        decompressed_np = image_np.astype(np.float32) / 255.0
        
        return decompressed_np, compressed_size
    
    def run_comprehensive_evaluation(self):
        """Run complete evaluation"""
        print("🚀 Starting Comprehensive Evaluation")
        
        # 1. Evaluate WAVENET-MV
        self.evaluate_wavenet_mv()
        
        # 2. Evaluate without Wavelet CNN
        self.evaluate_without_wavelet()
        
        # 3. Evaluate baselines
        self.evaluate_baselines()
        
        # 4. Save results
        self.save_results()
        
        # 5. Generate plots
        self.generate_plots()
        
        print("✅ Comprehensive evaluation completed!")
    
    def save_results(self):
        """Save results to CSV"""
        df = pd.DataFrame(self.results)
        output_path = os.path.join(self.args.output_dir, 'comprehensive_results.csv')
        df.to_csv(output_path, index=False)
        print(f"✓ Results saved to {output_path}")
        
        # Print summary
        print("\n📊 RESULTS SUMMARY:")
        print("=" * 80)
        
        for result in self.results:
            method = result['method']
            if 'lambda' in result:
                identifier = f"{method} (λ={result['lambda']})"
            elif 'quality' in result and result['quality'] is not None:
                identifier = f"{method} (Q={result['quality']})"
            else:
                identifier = method
            
            print(f"{identifier:30} | PSNR: {result['psnr_db']:6.2f}dB | MS-SSIM: {result['ms_ssim']:.4f} | BPP: {result['bpp']:6.4f}")
    
    def generate_plots(self):
        """Generate rate-distortion plots"""
        print("\n📈 Generating plots...")
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Rate-Distortion plot (PSNR vs BPP)
        plt.figure(figsize=(12, 8))
        
        # Plot each method
        methods = df['method'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        
        for method, color in zip(methods, colors):
            method_data = df[df['method'] == method]
            plt.plot(method_data['bpp'], method_data['psnr_db'], 
                    'o-', label=method, color=color, linewidth=2, markersize=6)
        
        plt.xlabel('Bits Per Pixel (BPP)', fontsize=12)
        plt.ylabel('PSNR (dB)', fontsize=12)
        plt.title('Rate-Distortion Comparison: PSNR vs BPP', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        rd_plot_path = os.path.join(self.args.output_dir, 'rate_distortion_psnr.png')
        plt.savefig(rd_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Rate-Distortion plot (MS-SSIM vs BPP)
        plt.figure(figsize=(12, 8))
        
        for method, color in zip(methods, colors):
            method_data = df[df['method'] == method]
            plt.plot(method_data['bpp'], method_data['ms_ssim'], 
                    'o-', label=method, color=color, linewidth=2, markersize=6)
        
        plt.xlabel('Bits Per Pixel (BPP)', fontsize=12)
        plt.ylabel('MS-SSIM', fontsize=12)
        plt.title('Rate-Distortion Comparison: MS-SSIM vs BPP', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        ssim_plot_path = os.path.join(self.args.output_dir, 'rate_distortion_msssim.png')
        plt.savefig(ssim_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Rate-distortion plots saved to {self.args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive WAVENET-MV Evaluation')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--max_samples', type=int, default=100, help='Max samples to evaluate')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(args)
    
    # Run evaluation
    evaluator.run_comprehensive_evaluation()


if __name__ == '__main__':
    main() 