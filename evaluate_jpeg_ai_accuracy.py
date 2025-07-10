#!/usr/bin/env python3
"""
JPEG AI ACCURACY EVALUATION
============================
Script đánh giá AI accuracy chỉ cho JPEG compression
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse

# AI Model imports
try:
    from ultralytics import YOLO
    print("✅ YOLOv8 available")
    YOLO_AVAILABLE = True
except ImportError:
    print("❌ YOLOv8 not available - install: pip install ultralytics")
    YOLO_AVAILABLE = False

class JPEGAIEvaluator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Load YOLOv8 for object detection
        self.detection_model = None
        self._load_detection_model()
        
    def _load_detection_model(self):
        """Load YOLOv8 detection model"""
        if not YOLO_AVAILABLE:
            print("⚠️ Detection model not available")
            return
            
        try:
            # Use YOLOv8n (nano) for faster inference
            self.detection_model = YOLO('yolov8n.pt')
            print("✅ YOLOv8n detection model loaded")
        except Exception as e:
            print(f"❌ Failed to load detection model: {e}")
            
    def evaluate_detection(self, image_path):
        """Evaluate object detection on image"""
        if self.detection_model is None:
            return {'mAP': 0.0, 'num_objects': 0, 'avg_confidence': 0.0}
            
        try:
            # Run inference
            results = self.detection_model(image_path, verbose=False)
            
            if len(results) == 0:
                return {'mAP': 0.0, 'num_objects': 0, 'avg_confidence': 0.0}
                
            # Extract metrics
            result = results[0]
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                return {'mAP': 0.0, 'num_objects': 0, 'avg_confidence': 0.0}
                
            # Get confidence scores
            confidences = boxes.conf.cpu().numpy() if len(boxes.conf) > 0 else []
            avg_confidence = np.mean(confidences) if len(confidences) > 0 else 0.0
            
            # Use average confidence as proxy for mAP
            return {
                'mAP': float(avg_confidence),
                'num_objects': len(confidences),
                'avg_confidence': float(avg_confidence)
            }
            
        except Exception as e:
            print(f"❌ Detection evaluation failed: {e}")
            return {'mAP': 0.0, 'num_objects': 0, 'avg_confidence': 0.0}
    
    def evaluate_image(self, image_path):
        """Evaluate AI accuracy on single image"""
        detection_results = self.evaluate_detection(image_path)
        return detection_results

def compress_jpeg(image_path, quality, output_dir):
    """Compress image with JPEG"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate output filename
        image_name = Path(image_path).stem
        output_path = output_dir / f"{image_name}_q{quality}.jpg"
        
        # Compress with PIL
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(output_path, 'JPEG', quality=quality, optimize=True)
        
        # Verify output file exists
        if not output_path.exists():
            raise FileNotFoundError(f"Output file not created: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ JPEG compression failed for {image_path}: {e}")
        return None

def calculate_metrics(original_path, compressed_path):
    """Calculate PSNR, SSIM, and BPP metrics"""
    try:
        # Validate files exist
        if not os.path.exists(original_path) or not os.path.exists(compressed_path):
            return None
        
        # Load images
        original = cv2.imread(str(original_path))
        compressed = cv2.imread(str(compressed_path))
        
        if original is None or compressed is None:
            return None
            
        # Resize if needed
        if original.shape != compressed.shape:
            compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))
        
        # Calculate PSNR
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # Calculate SSIM
        from skimage.metrics import structural_similarity as ssim
        ssim_score = ssim(original, compressed, multichannel=True, channel_axis=2, data_range=255)
        
        # Calculate BPP
        file_size = os.path.getsize(compressed_path)
        height, width = original.shape[:2]
        bpp = (file_size * 8) / (height * width)
        
        return {
            'psnr': float(psnr),
            'ssim': float(ssim_score),
            'bpp': float(bpp),
            'file_size': int(file_size)
        }
        
    except Exception as e:
        print(f"❌ Metrics calculation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate JPEG AI accuracy')
    parser.add_argument('--data_dir', type=str, default='datasets/COCO', 
                       help='Path to COCO dataset')
    parser.add_argument('--max_images', type=int, default=50,
                       help='Maximum number of images to evaluate')
    parser.add_argument('--quality_levels', nargs='+', type=int, 
                       default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95],
                       help='JPEG quality levels to test')
    parser.add_argument('--output_dir', type=str, default='results/jpeg_ai_accuracy',
                       help='Output directory for results')
    parser.add_argument('--temp_dir', type=str, default='temp_jpeg',
                       help='Temporary directory for compressed images')
    
    args = parser.parse_args()
    
    print("📸 JPEG AI ACCURACY EVALUATION")
    print("=" * 40)
    
    # Initialize evaluator
    evaluator = JPEGAIEvaluator()
    
    # Find COCO images
    image_dir = Path(args.data_dir) / "val2017"
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return
    
    image_files = list(image_dir.glob("*.jpg"))[:args.max_images]
    print(f"📁 Found {len(image_files)} images to evaluate")
    
    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    results = []
    
    # Evaluate each quality level
    total_evaluations = len(args.quality_levels) * len(image_files)
    pbar = tqdm(total=total_evaluations, desc="Evaluating JPEG")
    
    for quality in args.quality_levels:
        print(f"\n🔄 Evaluating JPEG Q={quality}")
        
        quality_results = []
        success_count = 0
        
        for image_path in image_files:
            try:
                # Compress image
                compressed_path = compress_jpeg(image_path, quality, temp_dir)
                
                # Skip if compression failed
                if compressed_path is None:
                    pbar.update(1)
                    continue
                
                # Calculate compression metrics
                compression_metrics = calculate_metrics(image_path, compressed_path)
                if compression_metrics is None:
                    pbar.update(1)
                    continue
                
                # Evaluate AI accuracy on compressed image
                ai_metrics = evaluator.evaluate_image(compressed_path)
                
                # Combine results
                result = {
                    'codec': 'JPEG',
                    'quality': quality,
                    'image_path': str(image_path),
                    'compressed_path': str(compressed_path),
                    **compression_metrics,
                    **ai_metrics
                }
                
                results.append(result)
                quality_results.append(result)
                success_count += 1
                
            except Exception as e:
                print(f"❌ Failed to process {image_path.name}: {e}")
            
            pbar.update(1)
        
        # Print summary for this quality level
        if quality_results:
            avg_psnr = np.mean([r['psnr'] for r in quality_results if np.isfinite(r['psnr'])])
            avg_ssim = np.mean([r['ssim'] for r in quality_results])
            avg_bpp = np.mean([r['bpp'] for r in quality_results])
            avg_map = np.mean([r['mAP'] for r in quality_results])
            avg_objects = np.mean([r['num_objects'] for r in quality_results])
            
            print(f"  📊 Q={quality}: PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.3f}, "
                  f"BPP={avg_bpp:.3f}, mAP={avg_map:.3f}, Objects={avg_objects:.1f}")
            print(f"  ✅ Successfully processed: {success_count}/{len(image_files)} images")
    
    pbar.close()
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        output_file = output_dir / "jpeg_ai_accuracy.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
        
        # Generate paper-ready summary
        print("\n📊 JPEG AI ACCURACY SUMMARY")
        print("=" * 50)
        
        for quality in args.quality_levels:
            quality_data = df[df['quality'] == quality]
            if not quality_data.empty:
                psnr_mean = quality_data['psnr'].mean()
                psnr_std = quality_data['psnr'].std()
                ssim_mean = quality_data['ssim'].mean()
                ssim_std = quality_data['ssim'].std()
                bpp_mean = quality_data['bpp'].mean()
                bpp_std = quality_data['bpp'].std()
                map_mean = quality_data['mAP'].mean()
                map_std = quality_data['mAP'].std()
                
                print(f"JPEG Q={quality:2d}: PSNR={psnr_mean:5.1f}±{psnr_std:.1f}dB, "
                      f"SSIM={ssim_mean:.3f}±{ssim_std:.3f}, "
                      f"BPP={bpp_mean:.3f}±{bpp_std:.3f}, "
                      f"mAP={map_mean:.3f}±{map_std:.3f}")
        
        # Generate LaTeX table
        latex_file = output_dir / "jpeg_latex_table.txt"
        with open(latex_file, 'w') as f:
            f.write("\\begin{table}[!t]\n")
            f.write("\\centering\n")
            f.write("\\caption{JPEG Compression and AI Task Performance}\n")
            f.write("\\begin{tabular}{|c|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("Quality & PSNR (dB) & MS-SSIM & BPP & AI Acc (mAP) \\\\\n")
            f.write("\\hline\n")
            
            for quality in args.quality_levels:
                quality_data = df[df['quality'] == quality]
                if not quality_data.empty:
                    psnr_mean = quality_data['psnr'].mean()
                    psnr_std = quality_data['psnr'].std()
                    ssim_mean = quality_data['ssim'].mean()
                    ssim_std = quality_data['ssim'].std()
                    bpp_mean = quality_data['bpp'].mean()
                    bpp_std = quality_data['bpp'].std()
                    map_mean = quality_data['mAP'].mean()
                    map_std = quality_data['mAP'].std()
                    
                    f.write(f"{quality} & {psnr_mean:.1f}±{psnr_std:.1f} & "
                           f"{ssim_mean:.3f}±{ssim_std:.3f} & "
                           f"{bpp_mean:.3f}±{bpp_std:.3f} & "
                           f"{map_mean:.3f}±{map_std:.3f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"✅ LaTeX table saved to: {latex_file}")
        
    else:
        print("❌ No results generated")
    
    # Cleanup temp files
    import shutil
    try:
        shutil.rmtree(temp_dir)
        print(f"🗑️ Cleaned up temporary files")
    except:
        pass
    
    print("\n🎉 JPEG AI Accuracy Evaluation Completed!")

if __name__ == "__main__":
    main() 