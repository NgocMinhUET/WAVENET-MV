#!/usr/bin/env python3
"""
Official COCO Dataset Setup Script for WAVENET-MV
Based on https://cocodataset.org/#download and community best practices
Supports both full dataset và minimal validation set for testing
"""

import os
import sys
import urllib.request
import zipfile
import argparse
from pathlib import Path
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# COCO Dataset URLs - Official Microsoft URLs
COCO_URLS = {
    'train2017': 'http://images.cocodataset.org/zips/train2017.zip',  # 19GB, 118K images
    'val2017': 'http://images.cocodataset.org/zips/val2017.zip',      # 1GB, 5K images
    'test2017': 'http://images.cocodataset.org/zips/test2017.zip',    # 7GB, 41K images
    'unlabeled2017': 'http://images.cocodataset.org/zips/unlabeled2017.zip',  # 19GB
    
    # Annotations
    'annotations_trainval2017': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',  # 241MB
    'stuff_annotations_trainval2017': 'http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip',  # 401MB  
    'image_info_test2017': 'http://images.cocodataset.org/annotations/image_info_test2017.zip',  # 1MB
    'image_info_unlabeled2017': 'http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip'  # 4MB
}

# File sizes for verification (approx)
EXPECTED_SIZES = {
    'train2017.zip': 19_000_000_000,  # 19GB
    'val2017.zip': 1_000_000_000,     # 1GB
    'test2017.zip': 7_000_000_000,    # 7GB
    'annotations_trainval2017.zip': 241_000_000,  # 241MB
}

def print_header():
    """Print setup header"""
    print("🔥" + "="*60 + "🔥")
    print("  OFFICIAL COCO DATASET SETUP FOR WAVENET-MV")
    print("  Based on: https://cocodataset.org/#download")
    print("🔥" + "="*60 + "🔥")
    print()

def download_with_progress(url, filepath, desc="Downloading"):
    """Download file với progress bar"""
    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                sys.stdout.write(f"\r{desc}: {percent:3d}% ({block_num * block_size // 1024 // 1024:,}MB/{total_size // 1024 // 1024:,}MB)")
                sys.stdout.flush()
        
        print(f"📥 {desc}: {url}")
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print()  # New line after progress
        
        # Verify file size
        file_size = os.path.getsize(filepath)
        print(f"✅ Downloaded: {os.path.basename(filepath)} ({file_size // 1024 // 1024:,}MB)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def extract_with_progress(zip_path, extract_dir, desc="Extracting"):
    """Extract ZIP file với progress tracking"""
    try:
        print(f"📦 {desc}: {os.path.basename(zip_path)}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            for i, member in enumerate(members):
                zip_ref.extract(member, extract_dir)
                if i % 100 == 0:  # Update every 100 files
                    percent = (i * 100) // len(members)
                    sys.stdout.write(f"\r{desc}: {percent:3d}% ({i+1:,}/{len(members):,} files)")
                    sys.stdout.flush()
        
        print(f"\n✅ Extracted: {len(members):,} files")
        return True
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        return False

def setup_coco_structure(base_dir):
    """Setup COCO directory structure"""
    print("📁 Setting up COCO directory structure...")
    
    directories = [
        'images/train2017',
        'images/val2017', 
        'images/test2017',
        'images/unlabeled2017',
        'annotations',
        'labels/train2017',
        'labels/val2017'
    ]
    
    for dir_path in directories:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        
    print(f"✅ Created directory structure in {base_dir}")

def verify_coco_structure(base_dir):
    """Verify COCO dataset structure"""
    print("\n🔍 Verifying COCO dataset structure...")
    
    required_files = {
        'images/val2017': '*.jpg',
        'annotations/instances_val2017.json': None,
        'annotations/instances_train2017.json': None,
    }
    
    issues = []
    
    for path, pattern in required_files.items():
        full_path = os.path.join(base_dir, path)
        
        if pattern is None:  # Single file
            if os.path.exists(full_path):
                size = os.path.getsize(full_path)
                print(f"✅ {path} ({size // 1024 // 1024:,}MB)")
            else:
                issues.append(f"❌ Missing: {path}")
        else:  # Directory with files
            if os.path.exists(full_path):
                if pattern == '*.jpg':
                    jpg_files = [f for f in os.listdir(full_path) if f.endswith('.jpg')]
                    print(f"✅ {path} ({len(jpg_files):,} images)")
                    if len(jpg_files) == 0:
                        issues.append(f"⚠️ Empty directory: {path}")
            else:
                issues.append(f"❌ Missing directory: {path}")
    
    if issues:
        print("\n⚠️ Issues found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n🎉 COCO dataset structure verified successfully!")
        return True

def create_dataset_info(base_dir, datasets_downloaded):
    """Create dataset information file"""
    info = {
        "dataset": "COCO 2017",
        "source": "https://cocodataset.org/",
        "setup_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets_downloaded": datasets_downloaded,
        "directory_structure": {
            "images/": "Image files organized by split",
            "annotations/": "JSON annotation files",  
            "labels/": "YOLO format labels (if converted)"
        },
        "usage": {
            "wavenet_mv": "Compatible với WAVENET-MV COCODatasetLoader",
            "training": "Ready for 3-stage training pipeline",
            "evaluation": "Use val2017 for validation"
        },
        "statistics": {}
    }
    
    # Count files if available
    for split in ['train2017', 'val2017', 'test2017']:
        img_dir = os.path.join(base_dir, 'images', split)
        if os.path.exists(img_dir):
            jpg_count = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
            info["statistics"][split] = f"{jpg_count:,} images"
    
    # Save info file
    info_path = os.path.join(base_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"📄 Dataset info saved: {info_path}")

def download_dataset(url_key, base_dir, force_download=False):
    """Download và extract single dataset"""
    if url_key not in COCO_URLS:
        print(f"❌ Unknown dataset: {url_key}")
        return False
        
    url = COCO_URLS[url_key]
    filename = url.split('/')[-1]
    zip_path = os.path.join(base_dir, filename)
    
    # Check if already exists
    if os.path.exists(zip_path) and not force_download:
        print(f"⏭️  Already exists: {filename}")
    else:
        # Download
        if not download_with_progress(url, zip_path, f"Downloading {filename}"):
            return False
    
    # Determine extraction directory
    if 'images' in filename:
        extract_dir = os.path.join(base_dir, 'images')
    elif 'annotations' in filename:
        extract_dir = os.path.join(base_dir, 'annotations')
    else:
        extract_dir = base_dir
    
    # Extract
    if not extract_with_progress(zip_path, extract_dir, f"Extracting {filename}"):
        return False
    
    # Clean up ZIP file to save space
    try:
        os.remove(zip_path)
        print(f"🗑️  Cleaned up: {filename}")
    except:
        pass
        
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Official COCO Dataset Setup for WAVENET-MV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download validation set only (recommended for testing)
  python setup_coco_official.py --minimal
  
  # Download full dataset (119GB total)
  python setup_coco_official.py --full
  
  # Download specific datasets
  python setup_coco_official.py --datasets val2017 annotations_trainval2017
  
  # Custom download directory
  python setup_coco_official.py --minimal --dir /path/to/coco
        """)
    
    parser.add_argument('--dir', default='COCO', 
                       help='Download directory (default: COCO)')
    parser.add_argument('--minimal', action='store_true',
                       help='Download minimal dataset for testing (val2017 + annotations)')
    parser.add_argument('--full', action='store_true', 
                       help='Download full dataset (train+val+test+annotations)')
    parser.add_argument('--datasets', nargs='+', choices=list(COCO_URLS.keys()),
                       help='Specific datasets to download')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download existing files')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing dataset structure')
    
    args = parser.parse_args()
    
    print_header()
    
    base_dir = os.path.abspath(args.dir)
    print(f"📁 COCO Directory: {base_dir}")
    
    # Verify only mode
    if args.verify_only:
        return verify_coco_structure(base_dir)
    
    # Create directory structure
    setup_coco_structure(base_dir)
    
    # Determine what to download
    datasets_to_download = []
    
    if args.minimal:
        datasets_to_download = ['val2017', 'annotations_trainval2017']
        print("📦 Minimal setup: val2017 + annotations (~1.2GB)")
    elif args.full:
        datasets_to_download = ['train2017', 'val2017', 'test2017', 'annotations_trainval2017', 'stuff_annotations_trainval2017']
        print("📦 Full setup: All datasets (~46GB)")
    elif args.datasets:
        datasets_to_download = args.datasets
        print(f"📦 Custom setup: {', '.join(datasets_to_download)}")
    else:
        # Default: minimal
        datasets_to_download = ['val2017', 'annotations_trainval2017'] 
        print("📦 Default: Minimal setup (use --full for complete dataset)")
    
    print(f"🎯 Target directory: {base_dir}")
    print(f"📋 Datasets to download: {', '.join(datasets_to_download)}")
    print()
    
    # Download datasets
    success_count = 0
    for dataset in datasets_to_download:
        print(f"\n{'='*20} {dataset.upper()} {'='*20}")
        if download_dataset(dataset, base_dir, args.force):
            success_count += 1
        else:
            print(f"❌ Failed to download: {dataset}")
    
    # Final verification
    print(f"\n{'='*60}")
    print(f"📊 DOWNLOAD SUMMARY: {success_count}/{len(datasets_to_download)} successful")
    
    if success_count == len(datasets_to_download):
        # Verify structure
        if verify_coco_structure(base_dir):
            # Create info file
            create_dataset_info(base_dir, datasets_to_download)
            
            print(f"\n🎉 COCO DATASET SETUP COMPLETED SUCCESSFULLY!")
            print(f"📁 Location: {base_dir}")
            print(f"🚀 Ready for WAVENET-MV training!")
            
            # Usage instructions
            print(f"\n📋 Next Steps:")
            print(f"1. Update dataset path in COCODatasetLoader: data_dir='{base_dir}'")
            print(f"2. Test loading: python -c \"from datasets.dataset_loaders import COCODatasetLoader; dataset = COCODatasetLoader(data_dir='{base_dir}', subset='val')\"")
            print(f"3. Start Stage 1 training: python training/stage1_train_wavelet.py")
            
            return True
        else:
            print(f"\n⚠️ Setup completed với issues - please check above")
            return False
    else:
        print(f"\n❌ Setup failed - {len(datasets_to_download) - success_count} datasets failed to download")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 