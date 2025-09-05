#!/usr/bin/env python3
"""
LARGE-SCALE EVALUATION SETUP
============================
Script này setup large-scale evaluation để đáp ứng yêu cầu của reviewers:
- COCO val2017: 1000+ ảnh (từ 50 ảnh hiện tại)
- Statistical significance: N≥1000 cho p<0.05
- Multiple datasets: Cityscapes, ADE20K (future)
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def setup_coco_large_scale(data_dir, size=1000, seed=42):
    """Setup COCO dataset for large-scale evaluation"""
    print(f"\n🔧 Setting up COCO large-scale evaluation (N={size})")
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Check if COCO dataset exists
    coco_dir = Path(data_dir) / "COCO"
    val_dir = coco_dir / "val2017"
    ann_file = coco_dir / "annotations" / "instances_val2017.json"
    
    if not val_dir.exists():
        print(f"❌ COCO val2017 directory not found: {val_dir}")
        print("Please run: python datasets/setup_coco_official.py --minimal")
        return False
    
    if not ann_file.exists():
        print(f"❌ COCO annotations not found: {ann_file}")
        print("Please run: python datasets/setup_coco_official.py --minimal")
        return False
    
    # Load COCO annotations
    print("📊 Loading COCO annotations...")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get all image files
    all_images = list(val_dir.glob("*.jpg"))
    print(f"📁 Found {len(all_images)} total images in val2017")
    
    # Filter images that have annotations
    annotated_image_ids = set([img['id'] for img in coco_data['images']])
    annotated_images = []
    
    for img_path in all_images:
        # Extract image ID from filename (e.g., 000000000139.jpg -> 139)
        img_id = int(img_path.stem)
        if img_id in annotated_image_ids:
            annotated_images.append(img_path)
    
    print(f"📊 Found {len(annotated_images)} images with annotations")
    
    # Sample subset for evaluation
    if size > len(annotated_images):
        print(f"⚠️ Requested size ({size}) > available images ({len(annotated_images)})")
        selected_size = len(annotated_images)
    else:
        selected_size = size
    
    selected_images = random.sample(annotated_images, selected_size)
    selected_image_ids = [int(img.stem) for img in selected_images]
    
    print(f"✅ Selected {selected_size} images for evaluation")
    
    # Create evaluation subset directory
    eval_dir = Path("evaluation_datasets") / f"COCO_eval_{selected_size}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symlinks or copy images (using symlinks for efficiency)
    images_dir = eval_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print("🔗 Creating image links...")
    for img_path in tqdm(selected_images, desc="Linking images"):
        link_path = images_dir / img_path.name
        if link_path.exists():
            link_path.unlink()
        
        try:
            # Try symlink first (more efficient)
            link_path.symlink_to(img_path.absolute())
        except OSError:
            # Fallback to copy if symlink fails (Windows)
            shutil.copy2(img_path, link_path)
    
    # Filter annotations for selected images
    print("📝 Filtering annotations...")
    
    # Filter images metadata
    filtered_images = [img for img in coco_data['images'] if img['id'] in selected_image_ids]
    
    # Filter annotations
    filtered_annotations = [ann for ann in coco_data['annotations'] 
                          if ann['image_id'] in selected_image_ids]
    
    # Create filtered annotation file
    filtered_coco_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'categories': coco_data['categories'],
        'images': filtered_images,
        'annotations': filtered_annotations
    }
    
    # Save filtered annotations
    ann_output_path = eval_dir / "annotations.json"
    with open(ann_output_path, 'w') as f:
        json.dump(filtered_coco_data, f)
    
    # Create evaluation config
    config = {
        'dataset': 'COCO',
        'subset_size': selected_size,
        'total_available': len(annotated_images),
        'images_dir': str(images_dir.absolute()),
        'annotations_file': str(ann_output_path.absolute()),
        'selected_image_ids': selected_image_ids[:100],  # Store first 100 for verification
        'random_seed': seed,
        'setup_date': pd.Timestamp.now().isoformat()
    }
    
    config_path = eval_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create evaluation stats
    stats = {
        'total_images': selected_size,
        'total_annotations': len(filtered_annotations),
        'avg_annotations_per_image': len(filtered_annotations) / selected_size,
        'categories_count': len(coco_data['categories']),
        'statistical_power': 'adequate' if selected_size >= 1000 else 'limited'
    }
    
    stats_path = eval_dir / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Large-scale evaluation dataset created:")
    print(f"   📁 Dataset directory: {eval_dir}")
    print(f"   🖼️ Images: {selected_size}")
    print(f"   📊 Annotations: {len(filtered_annotations)}")
    print(f"   📈 Avg annotations/image: {len(filtered_annotations)/selected_size:.2f}")
    print(f"   🎯 Statistical power: {'✅ Adequate' if selected_size >= 1000 else '⚠️ Limited'}")
    
    return eval_dir

def setup_cityscapes_subset(data_dir, size=500):
    """Setup Cityscapes dataset subset (placeholder for future)"""
    print(f"\n🔧 Cityscapes setup (N={size}) - PLACEHOLDER")
    print("⚠️ Cityscapes dataset not implemented yet")
    print("Future work: Download from https://www.cityscapes-dataset.com/")
    return None

def setup_ade20k_subset(data_dir, size=500):
    """Setup ADE20K dataset subset (placeholder for future)"""
    print(f"\n🔧 ADE20K setup (N={size}) - PLACEHOLDER")
    print("⚠️ ADE20K dataset not implemented yet")
    print("Future work: Download from http://data.csail.mit.edu/places/ADEchallenge/")
    return None

def create_evaluation_script(eval_dir, dataset_name):
    """Create evaluation script for the large-scale dataset"""
    script_content = f'''#!/usr/bin/env python3
"""
LARGE-SCALE EVALUATION SCRIPT
Auto-generated for {dataset_name} evaluation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from evaluate_ai_accuracy import AIAccuracyEvaluator, compress_with_codec, calculate_metrics
import json
import pandas as pd
from tqdm import tqdm

def run_large_scale_evaluation():
    """Run large-scale evaluation on {dataset_name}"""
    
    # Load config
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"🔧 Large-scale evaluation: {{config['dataset']}} (N={{config['subset_size']}})")
    
    # Initialize evaluator
    evaluator = AIAccuracyEvaluator()
    
    # Get image list
    images_dir = Path(config['images_dir'])
    image_files = list(images_dir.glob("*.jpg"))
    
    print(f"📁 Found {{len(image_files)}} images for evaluation")
    
    # Evaluation parameters
    codecs = ['JPEG']
    quality_levels = [10, 30, 50, 70, 90]
    
    results = []
    total = len(codecs) * len(quality_levels) * len(image_files)
    
    with tqdm(total=total, desc="Large-scale evaluation") as pbar:
        for codec in codecs:
            for quality in quality_levels:
                for img_path in image_files:
                    try:
                        # Compress
                        compressed_path = compress_with_codec(
                            img_path, codec, quality, 
                            Path("temp_large_scale") / codec / str(quality)
                        )
                        
                        if compressed_path:
                            # Calculate metrics
                            comp_metrics = calculate_metrics(img_path, compressed_path)
                            ai_metrics = evaluator.evaluate_image(compressed_path)
                            
                            if comp_metrics:
                                result = {{
                                    'codec': codec,
                                    'quality': quality,
                                    'image_id': int(img_path.stem),
                                    **comp_metrics,
                                    **ai_metrics
                                }}
                                results.append(result)
                    
                    except Exception as e:
                        print(f"❌ Error processing {{img_path.name}}: {{e}}")
                    
                    pbar.update(1)
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        output_path = Path(__file__).parent / "large_scale_results.csv"
        df.to_csv(output_path, index=False)
        
        # Generate summary statistics
        summary = {{}}
        for codec in codecs:
            codec_data = df[df['codec'] == codec]
            summary[codec] = {{
                'mean_psnr': codec_data['psnr'].mean(),
                'std_psnr': codec_data['psnr'].std(),
                'mean_mAP': codec_data['mAP'].mean(),
                'std_mAP': codec_data['mAP'].std(),
                'mean_bpp': codec_data['bpp'].mean(),
                'sample_size': len(codec_data)
            }}
        
        # Save summary
        summary_path = Path(__file__).parent / "statistical_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Results saved: {{output_path}}")
        print(f"✅ Summary saved: {{summary_path}}")
        print(f"📊 Total evaluations: {{len(results)}}")
        
        # Statistical power analysis
        n = len(results) // len(quality_levels)  # Per quality level
        power_status = "✅ Adequate (N≥1000)" if n >= 1000 else f"⚠️ Limited (N={{n}})"
        print(f"📈 Statistical power: {{power_status}}")

if __name__ == "__main__":
    run_large_scale_evaluation()
'''
    
    script_path = eval_dir / "run_evaluation.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_path.chmod(0o755)
    
    print(f"✅ Evaluation script created: {script_path}")
    return script_path

def main():
    parser = argparse.ArgumentParser(description='Setup Large-Scale Evaluation Datasets')
    
    parser.add_argument('--dataset', type=str, choices=['coco', 'cityscapes', 'ade20k'], 
                       default='coco', help='Dataset to setup')
    parser.add_argument('--size', type=int, default=1000,
                       help='Number of images for evaluation')
    parser.add_argument('--data_dir', type=str, default='datasets',
                       help='Base data directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("🚀 LARGE-SCALE EVALUATION SETUP")
    print("=" * 50)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Size: {args.size} images")
    print(f"Data directory: {args.data_dir}")
    print(f"Random seed: {args.seed}")
    
    # Setup based on dataset
    eval_dir = None
    
    if args.dataset == 'coco':
        eval_dir = setup_coco_large_scale(args.data_dir, args.size, args.seed)
    elif args.dataset == 'cityscapes':
        eval_dir = setup_cityscapes_subset(args.data_dir, args.size)
    elif args.dataset == 'ade20k':
        eval_dir = setup_ade20k_subset(args.data_dir, args.size)
    
    # Create evaluation script
    if eval_dir:
        create_evaluation_script(eval_dir, args.dataset.upper())
        
        print("\n🎯 NEXT STEPS:")
        print(f"1. cd {eval_dir}")
        print("2. python run_evaluation.py")
        print("3. Check large_scale_results.csv")
        print("4. Review statistical_summary.json")
        
        # Statistical power guidance
        if args.size >= 1000:
            print(f"\n✅ Statistical Power: ADEQUATE (N={args.size})")
            print("   - Sufficient for p<0.05 significance testing")
            print("   - Can calculate 95% confidence intervals")
            print("   - Addresses Reviewer concerns about sample size")
        else:
            print(f"\n⚠️ Statistical Power: LIMITED (N={args.size})")
            print("   - Consider increasing to N≥1000 for publication")
            print("   - Current size may not satisfy reviewers")
    
    print("\n🎉 Large-scale evaluation setup completed!")

if __name__ == "__main__":
    main() 