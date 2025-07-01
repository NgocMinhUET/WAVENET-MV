#!/usr/bin/env python3
"""
Debug script for COCO dataset issues on server
"""

import os
import sys

def check_dataset_structure(data_dir='datasets/COCO_Official'):
    """Check COCO dataset structure and files"""
    print("🔍 COCO Dataset Structure Check")
    print("=" * 50)
    
    # Check base directory
    print(f"📁 Base directory: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"❌ ERROR: {data_dir} does not exist!")
        return False
    else:
        print(f"✅ {data_dir} exists")
    
    # List contents
    print(f"\n📂 Contents of {data_dir}:")
    try:
        contents = os.listdir(data_dir)
        for item in sorted(contents):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
            else:
                print(f"  📄 {item}")
    except Exception as e:
        print(f"❌ Error listing contents: {e}")
        return False
    
    # Check required directories
    required_dirs = ['val2017', 'annotations']
    for req_dir in required_dirs:
        dir_path = os.path.join(data_dir, req_dir)
        print(f"\n📁 Checking {req_dir}:")
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path} exists")
            
            # Count files
            try:
                files = os.listdir(dir_path)
                print(f"  📊 Contains {len(files)} items")
                
                # Show first few files
                print(f"  📋 First 5 items:")
                for i, file in enumerate(sorted(files)[:5]):
                    print(f"    - {file}")
                    
            except Exception as e:
                print(f"  ❌ Error reading directory: {e}")
        else:
            print(f"  ❌ {dir_path} does not exist!")
    
    # Check specific annotation files
    print(f"\n📋 Checking annotation files:")
    annotation_files = [
        'annotations/instances_val2017.json',
        'annotations/instances_train2017.json'
    ]
    
    for ann_file in annotation_files:
        ann_path = os.path.join(data_dir, ann_file)
        if os.path.exists(ann_path):
            file_size = os.path.getsize(ann_path)
            print(f"  ✅ {ann_file} - Size: {file_size/1024/1024:.1f}MB")
        else:
            print(f"  ❌ {ann_file} - Not found!")
    
    return True

def check_alternative_paths():
    """Check alternative dataset paths"""
    print("\n🔍 Checking Alternative Paths")
    print("=" * 50)
    
    possible_paths = [
        'datasets/COCO_Official',
        'datasets/COCO',
        'COCO_Official',
        'COCO',
        '../COCO_Official',
        '../datasets/COCO_Official'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found: {path}")
            check_dataset_structure(path)
            return path
        else:
            print(f"❌ Not found: {path}")
    
    return None

def suggest_fixes():
    """Suggest fixes based on findings"""
    print("\n🔧 Suggested Fixes")
    print("=" * 50)
    
    print("1. Check if COCO dataset was downloaded:")
    print("   ls -la datasets/")
    print("   ls -la datasets/COCO_Official/")
    
    print("\n2. If dataset missing, run setup script:")
    print("   python datasets/setup_coco_official.py")
    
    print("\n3. If annotations missing, check structure:")
    print("   find datasets/COCO_Official -name '*.json' -type f")
    
    print("\n4. Manual verification:")
    print("   cd datasets/COCO_Official")
    print("   ls -la")
    print("   ls -la val2017/ | head -10")
    print("   ls -la annotations/")

if __name__ == "__main__":
    print("🚀 COCO Dataset Debug Tool")
    print("=" * 50)
    
    # Check current working directory
    cwd = os.getcwd()
    print(f"📍 Current directory: {cwd}")
    
    # Check dataset structure
    if not check_dataset_structure():
        # Try alternative paths
        found_path = check_alternative_paths()
        if not found_path:
            print("\n❌ No valid COCO dataset found!")
    
    suggest_fixes()
    
    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Run this script on server to identify exact issue")
    print("2. Follow suggested fixes")
    print("3. Verify with: python -c \"from datasets.dataset_loaders import COCODatasetLoader; d=COCODatasetLoader('datasets/COCO_Official'); print(f'Dataset size: {len(d)}')\"") 