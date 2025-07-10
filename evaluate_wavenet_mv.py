#!/usr/bin/env python3
"""
WAVENET-MV EVALUATION WITH AI ACCURACY
======================================
Script đánh giá WAVENET-MV model với AI task performance
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import argparse
import cv2

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Model imports
from models.wavelet_transform_cnn import WaveletTransformCNN
from models.compressor_improved import ImprovedCompressor
from models.ai_heads import AIHeads

# AI evaluation imports
try:
    from evaluate_ai_accuracy import AIAccuracyEvaluator
    AI_EVAL_AVAILABLE = True
except ImportError:
    print("⚠️ AI evaluation not available")
    AI_EVAL_AVAILABLE = False

class WavenetMVEvaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Load WAVENET-MV model
        self.model = self._load_model(model_path)
        
        # Load AI accuracy evaluator
        if AI_EVAL_AVAILABLE:
            self.ai_evaluator = AIAccuracyEvaluator()
        else:
            self.ai_evaluator = None
            print("⚠️ AI accuracy evaluation not available")
    
    def _load_model(self, model_path):
        """Load WAVENET-MV model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model components
            wavelet_transform = WaveletTransformCNN(
                in_channels=3,
                wavelet='db4',
                levels=3
            )
            
            compressor = ImprovedCompressor(
                latent_channels=64,
                num_layers=6,
                growth_rate=32
            )
            
            ai_heads = AIHeads(
                feature_dim=64,
                num_detection_classes=80,
                num_segmentation_classes=21
            )
            
            # Load state dicts
            if 'wavelet_transform' in checkpoint:
                wavelet_transform.load_state_dict(checkpoint['wavelet_transform'])
            if 'compressor' in checkpoint:
                compressor.load_state_dict(checkpoint['compressor'])
            if 'ai_heads' in checkpoint:
                ai_heads.load_state_dict(checkpoint['ai_heads'])
            
            # Create combined model
            model = WavenetMVModel(wavelet_transform, compressor, ai_heads)
            model.to(self.device)
            model.eval()
            
            print(f"✅ WAVENET-MV model loaded from: {model_path}")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def compress_and_decompress(self, image_path, lambda_val):
        """Compress and decompress image with WAVENET-MV"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Transform
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_tensor, lambda_val)
            
            # Extract reconstructed image
            reconstructed = outputs['reconstructed'].squeeze(0).cpu()
            
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            reconstructed = reconstructed * std + mean
            reconstructed = torch.clamp(reconstructed, 0, 1)
            
            # Convert to PIL
            reconstructed_pil = transforms.ToPILImage()(reconstructed)
            
            # Calculate compression metrics
            bpp = outputs.get('bpp', 0.0)
            
            return reconstructed_pil, bpp
            
        except Exception as e:
            print(f"❌ Compression failed: {e}")
            return None, 0.0
    
    def evaluate_image(self, image_path, lambda_val, temp_dir):
        """Evaluate single image with WAVENET-MV"""
        try:
            # Compress and decompress
            reconstructed, bpp = self.compress_and_decompress(image_path, lambda_val)
            
            if reconstructed is None:
                return None
            
            # Save reconstructed image
            temp_path = temp_dir / f"reconstructed_{lambda_val}.jpg"
            reconstructed.save(temp_path, 'JPEG', quality=95)
            
            # Calculate compression metrics
            original = cv2.imread(str(image_path))
            recon_cv = cv2.imread(str(temp_path))
            
            if original is None or recon_cv is None:
                return None
            
            # Resize if needed
            if original.shape != recon_cv.shape:
                recon_cv = cv2.resize(recon_cv, (original.shape[1], original.shape[0]))
            
            # Calculate PSNR
            mse = np.mean((original - recon_cv) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            
            # Calculate SSIM
            from skimage.metrics import structural_similarity as ssim
            ssim_score = ssim(original, recon_cv, multichannel=True, channel_axis=2, data_range=255)
            
            # Calculate AI accuracy
            ai_metrics = {'mAP': 0.0, 'mIoU': 0.0, 'pixel_accuracy': 0.0, 'num_objects': 0}
            if self.ai_evaluator:
                ai_metrics = self.ai_evaluator.evaluate_image(temp_path)
            
            # Combine results
            result = {
                'psnr': float(psnr),
                'ssim': float(ssim_score),
                'bpp': float(bpp),
                'lambda': lambda_val,
                **ai_metrics
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Image evaluation failed: {e}")
            return None

class WavenetMVModel(nn.Module):
    """Combined WAVENET-MV model"""
    def __init__(self, wavelet_transform, compressor, ai_heads):
        super().__init__()
        self.wavelet_transform = wavelet_transform
        self.compressor = compressor
        self.ai_heads = ai_heads
    
    def forward(self, x, lambda_val):
        # Wavelet transform
        wavelet_coeffs = self.wavelet_transform.forward_transform(x)
        
        # Compression
        compressed, bpp = self.compressor(wavelet_coeffs, lambda_val)
        
        # Decompression
        reconstructed_coeffs = self.compressor.decompress(compressed)
        
        # Inverse wavelet transform
        reconstructed = self.wavelet_transform.inverse_transform(reconstructed_coeffs)
        
        # AI heads (optional)
        ai_outputs = self.ai_heads(reconstructed_coeffs)
        
        return {
            'reconstructed': reconstructed,
            'bpp': bpp,
            'ai_outputs': ai_outputs
        }

def main():
    parser = argparse.ArgumentParser(description='Evaluate WAVENET-MV with AI accuracy')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to WAVENET-MV model checkpoint')
    parser.add_argument('--data_dir', type=str, default='datasets/COCO',
                       help='Path to COCO dataset')
    parser.add_argument('--max_images', type=int, default=50,
                       help='Maximum number of images to evaluate')
    parser.add_argument('--lambda_values', nargs='+', type=int,
                       default=[64, 128, 256, 512, 1024, 2048],
                       help='Lambda values to test')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--temp_dir', type=str, default='temp_wavenet',
                       help='Temporary directory for reconstructed images')
    
    args = parser.parse_args()
    
    print("🚀 WAVENET-MV EVALUATION WITH AI ACCURACY")
    print("=" * 50)
    
    # Check model file
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        print("Please train WAVENET-MV model first!")
        return
    
    # Check dataset
    image_dir = Path(args.data_dir) / "val2017"
    if not image_dir.exists():
        print(f"❌ Dataset not found: {image_dir}")
        return
    
    # Get images
    image_files = list(image_dir.glob("*.jpg"))[:args.max_images]
    print(f"📁 Found {len(image_files)} images to evaluate")
    
    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    try:
        evaluator = WavenetMVEvaluator(args.model_path)
    except Exception as e:
        print(f"❌ Failed to initialize evaluator: {e}")
        return
    
    # Results storage
    results = []
    
    # Evaluate each lambda value
    total_evaluations = len(args.lambda_values) * len(image_files)
    pbar = tqdm(total=total_evaluations, desc="Evaluating WAVENET-MV")
    
    for lambda_val in args.lambda_values:
        print(f"\n🔄 Evaluating λ={lambda_val}")
        
        lambda_results = []
        
        for image_path in image_files:
            result = evaluator.evaluate_image(image_path, lambda_val, temp_dir)
            
            if result is not None:
                result['image_path'] = str(image_path)
                results.append(result)
                lambda_results.append(result)
            
            pbar.update(1)
        
        # Print summary for this lambda
        if lambda_results:
            avg_psnr = np.mean([r['psnr'] for r in lambda_results if np.isfinite(r['psnr'])])
            avg_ssim = np.mean([r['ssim'] for r in lambda_results])
            avg_bpp = np.mean([r['bpp'] for r in lambda_results])
            avg_map = np.mean([r['mAP'] for r in lambda_results])
            avg_miou = np.mean([r['mIoU'] for r in lambda_results])
            
            print(f"  📊 λ={lambda_val}: PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.3f}, "
                  f"BPP={avg_bpp:.3f}, mAP={avg_map:.3f}, mIoU={avg_miou:.3f}")
    
    pbar.close()
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        output_file = output_dir / "wavenet_mv_evaluation.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
        
        # Generate summary
        summary_file = output_dir / "wavenet_mv_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("WAVENET-MV EVALUATION SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Images evaluated: {len(image_files)}\n")
            f.write(f"Lambda values: {args.lambda_values}\n\n")
            
            for lambda_val in args.lambda_values:
                lambda_data = df[df['lambda'] == lambda_val]
                if not lambda_data.empty:
                    f.write(f"λ={lambda_val}:\n")
                    
                    finite_psnr = lambda_data[np.isfinite(lambda_data['psnr'])]
                    if not finite_psnr.empty:
                        f.write(f"  PSNR: {finite_psnr['psnr'].mean():.2f} ± {finite_psnr['psnr'].std():.2f} dB\n")
                    f.write(f"  SSIM: {lambda_data['ssim'].mean():.4f} ± {lambda_data['ssim'].std():.4f}\n")
                    f.write(f"  BPP:  {lambda_data['bpp'].mean():.4f} ± {lambda_data['bpp'].std():.4f}\n")
                    f.write(f"  mAP:  {lambda_data['mAP'].mean():.4f} ± {lambda_data['mAP'].std():.4f}\n")
                    f.write(f"  mIoU: {lambda_data['mIoU'].mean():.4f} ± {lambda_data['mIoU'].std():.4f}\n\n")
        
        print(f"✅ Summary saved to: {summary_file}")
        
    else:
        print("❌ No results generated")
    
    # Cleanup
    import shutil
    try:
        shutil.rmtree(temp_dir)
        print(f"🗑️ Cleaned up temporary files")
    except:
        pass
    
    print("\n🎉 WAVENET-MV Evaluation Completed!")

if __name__ == "__main__":
    main() 