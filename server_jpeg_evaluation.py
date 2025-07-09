#!/usr/bin/env python3
"""
JPEG/JPEG2000 Baseline Evaluation Script for Server
Đánh giá JPEG và JPEG2000 compression với các quality levels khác nhau
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from PIL import Image
import pillow_heif
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from datasets.dataset_loaders import COCODatasetLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_metrics(original, compressed):
    """
    Calculate PSNR, SSIM, and MS-SSIM between original and compressed images
    """
    # Convert to numpy arrays
    if isinstance(original, Image.Image):
        original = np.array(original)
    if isinstance(compressed, Image.Image):
        compressed = np.array(compressed)
    
    # Ensure same dtype and range
    original = original.astype(np.float32)
    compressed = compressed.astype(np.float32)
    
    # Calculate PSNR
    psnr_value = psnr(original, compressed, data_range=255)
    
    # Calculate SSIM
    if len(original.shape) == 3:
        ssim_value = ssim(original, compressed, data_range=255, channel_axis=2)
    else:
        ssim_value = ssim(original, compressed, data_range=255)
    
    return psnr_value, ssim_value


def evaluate_jpeg(image_path, quality, output_dir):
    """
    Evaluate JPEG compression for a single image
    """
    try:
        # Load original image
        original = cv2.imread(image_path)
        if original is None:
            return None
        
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Compress with JPEG
        temp_path = os.path.join(output_dir, f'temp_jpeg_{quality}.jpg')
        cv2.imwrite(temp_path, original, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        # Load compressed image
        compressed = cv2.imread(temp_path)
        compressed_rgb = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
        
        # Calculate metrics
        psnr_value, ssim_value = calculate_metrics(original_rgb, compressed_rgb)
        
        # Calculate file size and BPP
        file_size = os.path.getsize(temp_path)
        H, W = original.shape[:2]
        bpp = (file_size * 8) / (H * W)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            'codec': 'JPEG',
            'quality': quality,
            'psnr': psnr_value,
            'ssim': ssim_value,
            'bpp': bpp,
            'file_size': file_size,
            'image_path': image_path
        }
        
    except Exception as e:
        print(f"Error processing JPEG {image_path}: {e}")
        return None


def evaluate_jpeg2000(image_path, quality, output_dir):
    """
    Evaluate JPEG2000 compression for a single image
    """
    try:
        # Load original image
        original = cv2.imread(image_path)
        if original is None:
            return None
        
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Compress with JPEG2000 using OpenCV
        temp_path = os.path.join(output_dir, f'temp_jp2_{quality}.jp2')
        
        # Quality mapping for JPEG2000 (0-100 to compression ratio)
        # Higher quality = lower compression ratio
        compression_ratio = max(1, int(100 - quality + 1))
        
        cv2.imwrite(temp_path, original, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, compression_ratio * 1000])
        
        # Load compressed image
        compressed = cv2.imread(temp_path)
        compressed_rgb = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
        
        # Calculate metrics
        psnr_value, ssim_value = calculate_metrics(original_rgb, compressed_rgb)
        
        # Calculate file size and BPP
        file_size = os.path.getsize(temp_path)
        H, W = original.shape[:2]
        bpp = (file_size * 8) / (H * W)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            'codec': 'JPEG2000',
            'quality': quality,
            'psnr': psnr_value,
            'ssim': ssim_value,
            'bpp': bpp,
            'file_size': file_size,
            'image_path': image_path
        }
        
    except Exception as e:
        print(f"Error processing JPEG2000 {image_path}: {e}")
        return None


def process_image(args):
    """
    Process a single image with both JPEG and JPEG2000
    """
    image_path, quality_levels, output_dir = args
    results = []
    
    for quality in quality_levels:
        # JPEG
        jpeg_result = evaluate_jpeg(image_path, quality, output_dir)
        if jpeg_result:
            results.append(jpeg_result)
        
        # JPEG2000
        jp2_result = evaluate_jpeg2000(image_path, quality, output_dir)
        if jp2_result:
            results.append(jp2_result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='JPEG/JPEG2000 Baseline Evaluation')
    parser.add_argument('--data_dir', type=str, default='datasets/COCO_Official', 
                        help='Path to COCO dataset')
    parser.add_argument('--subset', type=str, default='val', choices=['train', 'val'],
                        help='Dataset subset to evaluate')
    parser.add_argument('--max_images', type=int, default=100,
                        help='Maximum number of images to evaluate')
    parser.add_argument('--quality_levels', type=int, nargs='+', 
                        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95],
                        help='Quality levels to test')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--output_file', type=str, default='jpeg_baseline_results.csv',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"🔄 Starting JPEG/JPEG2000 Baseline Evaluation")
    print(f"📂 Dataset: {args.data_dir}")
    print(f"🎯 Max images: {args.max_images}")
    print(f"⚙️ Quality levels: {args.quality_levels}")
    print(f"🔧 Workers: {args.num_workers}")
    
    # Load dataset
    try:
        if args.subset == 'val':
            image_dir = os.path.join(args.data_dir, 'val2017')
        else:
            image_dir = os.path.join(args.data_dir, 'train2017')
        
        if not os.path.exists(image_dir):
            print(f"❌ Image directory not found: {image_dir}")
            return
        
        # Get image list
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(image_dir).glob(ext))
        
        if not image_files:
            print(f"❌ No images found in {image_dir}")
            return
        
        # Limit number of images
        image_files = image_files[:args.max_images]
        print(f"📊 Processing {len(image_files)} images...")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Prepare arguments for multiprocessing
    process_args = [(str(img_path), args.quality_levels, temp_dir) for img_path in image_files]
    
    # Process images in parallel
    all_results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        with tqdm(total=len(image_files), desc="Processing images") as pbar:
            futures = [executor.submit(process_image, arg) for arg in process_args]
            
            for future in futures:
                try:
                    results = future.result()
                    all_results.extend(results)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing image: {e}")
                    pbar.update(1)
    
    # Clean up temp directory
    try:
        for temp_file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, temp_file))
        os.rmdir(temp_dir)
    except:
        pass
    
    # Create DataFrame and save results
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Calculate statistics
        stats = df.groupby(['codec', 'quality']).agg({
            'psnr': ['mean', 'std', 'min', 'max'],
            'ssim': ['mean', 'std', 'min', 'max'],
            'bpp': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        # Save detailed results
        output_path = os.path.join(args.output_dir, args.output_file)
        df.to_csv(output_path, index=False)
        
        # Save summary statistics
        stats_path = os.path.join(args.output_dir, 'jpeg_baseline_stats.csv')
        stats.to_csv(stats_path)
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\n✅ Evaluation completed in {elapsed_time:.2f} seconds")
        print(f"📊 Processed {len(image_files)} images")
        print(f"💾 Results saved to: {output_path}")
        print(f"📈 Statistics saved to: {stats_path}")
        
        # Print quick summary
        print("\n📋 QUICK SUMMARY:")
        print("=" * 60)
        for codec in ['JPEG', 'JPEG2000']:
            codec_data = df[df['codec'] == codec]
            if not codec_data.empty:
                print(f"\n{codec}:")
                print(f"  PSNR range: {codec_data['psnr'].min():.2f} - {codec_data['psnr'].max():.2f} dB")
                print(f"  SSIM range: {codec_data['ssim'].min():.4f} - {codec_data['ssim'].max():.4f}")
                print(f"  BPP range:  {codec_data['bpp'].min():.4f} - {codec_data['bpp'].max():.4f}")
        
    else:
        print("❌ No results obtained")


if __name__ == "__main__":
    main() 