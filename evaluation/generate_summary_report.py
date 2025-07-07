#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Summary Report
-----------------------
Script này tổng hợp kết quả từ các file đánh giá riêng lẻ thành một báo cáo tổng hợp
"""

import os
import argparse
import pandas as pd
import glob
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Generate summary report from evaluation results')
    parser.add_argument('--input_dir', type=str, default='results', 
                        help='Directory containing evaluation result files')
    parser.add_argument('--output_file', type=str, default='results/wavenet_mv_comprehensive_results.csv',
                        help='Output file for the summary report')
    parser.add_argument('--format', type=str, choices=['csv', 'json'], default='csv',
                        help='Output format (csv or json)')
    return parser.parse_args()

def find_evaluation_files(input_dir):
    """Find all evaluation result files in the input directory"""
    codec_files = glob.glob(os.path.join(input_dir, '*codec_metrics.csv'))
    ai_files = glob.glob(os.path.join(input_dir, '*ai_metrics.csv'))
    
    # Group files by lambda value
    lambda_groups = {}
    
    for file in codec_files:
        # Extract lambda value from filename
        filename = os.path.basename(file)
        if 'lambda' in filename:
            lambda_value = filename.split('lambda')[1].split('_')[0]
            if lambda_value not in lambda_groups:
                lambda_groups[lambda_value] = {'codec': None, 'ai': None}
            lambda_groups[lambda_value]['codec'] = file
    
    for file in ai_files:
        # Extract lambda value from filename
        filename = os.path.basename(file)
        if 'lambda' in filename:
            lambda_value = filename.split('lambda')[1].split('_')[0]
            if lambda_value not in lambda_groups:
                lambda_groups[lambda_value] = {'codec': None, 'ai': None}
            lambda_groups[lambda_value]['ai'] = file
    
    return lambda_groups

def merge_results(lambda_groups):
    """Merge codec and AI metrics for each lambda value"""
    results = []
    
    for lambda_value, files in lambda_groups.items():
        codec_file = files.get('codec')
        ai_file = files.get('ai')
        
        if codec_file and os.path.exists(codec_file):
            try:
                codec_df = pd.read_csv(codec_file)
                # Take average of all rows if multiple rows exist
                codec_metrics = codec_df.mean(numeric_only=True).to_dict()
                
                # Basic metrics
                result = {
                    'method': 'WAVENET-MV',
                    'lambda': int(lambda_value),
                    'psnr_db': round(codec_metrics.get('psnr', 0.0), 2),
                    'ms_ssim': round(codec_metrics.get('ms_ssim', 0.0), 4),
                    'bpp': round(codec_metrics.get('bpp', 0.0), 2)
                }
                
                # Add AI metrics if available
                if ai_file and os.path.exists(ai_file):
                    try:
                        ai_df = pd.read_csv(ai_file)
                        ai_metrics = ai_df.mean(numeric_only=True).to_dict()
                        
                        result['detection_map'] = round(ai_metrics.get('detection_map', 0.0), 3)
                        result['segmentation_miou'] = round(ai_metrics.get('segmentation_miou', 0.0), 3)
                        result['ai_accuracy'] = round((result['detection_map'] + result['segmentation_miou']) / 2, 3)
                    except Exception as e:
                        print(f"Warning: Could not process AI metrics file {ai_file}: {e}")
                
                results.append(result)
            except Exception as e:
                print(f"Warning: Could not process codec metrics file {codec_file}: {e}")
    
    # Sort results by lambda value
    results.sort(key=lambda x: x['lambda'])
    return results

def add_baseline_results(results, input_dir):
    """Add baseline results if available"""
    baseline_file = os.path.join(input_dir, 'baseline_comparison.csv')
    if os.path.exists(baseline_file):
        try:
            baseline_df = pd.read_csv(baseline_file)
            for _, row in baseline_df.iterrows():
                baseline_result = {
                    'method': row['method'],
                    'lambda': None,
                    'psnr_db': round(row['psnr'], 2),
                    'ms_ssim': round(row['ms_ssim'], 4),
                    'bpp': round(row['bpp'], 2),
                    'ai_accuracy': round(row['ai_accuracy'], 3) if 'ai_accuracy' in row else None
                }
                
                # Add quality parameter if available
                if 'quality' in row:
                    baseline_result['quality'] = row['quality']
                
                results.append(baseline_result)
        except Exception as e:
            print(f"Warning: Could not process baseline file {baseline_file}: {e}")
    
    return results

def save_results(results, output_file, format='csv'):
    """Save results to a file in the specified format"""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if format == 'csv':
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"✅ Results saved to {output_file}")
    elif format == 'json':
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to {output_file}")
    else:
        print(f"❌ Unsupported format: {format}")

def main():
    args = parse_args()
    
    print(f"🔍 Looking for evaluation files in {args.input_dir}...")
    lambda_groups = find_evaluation_files(args.input_dir)
    
    if not lambda_groups:
        print("❌ No evaluation files found!")
        return
    
    print(f"✅ Found evaluation files for {len(lambda_groups)} lambda values")
    
    print("🔄 Merging results...")
    results = merge_results(lambda_groups)
    
    print("🔄 Adding baseline results if available...")
    results = add_baseline_results(results, args.input_dir)
    
    print(f"🔄 Saving results to {args.output_file}...")
    
    # Determine output format from file extension if not specified
    format = args.format
    if format == 'csv' and args.output_file.endswith('.json'):
        format = 'json'
    elif format == 'json' and args.output_file.endswith('.csv'):
        format = 'csv'
    
    save_results(results, args.output_file, format)
    
    # Also save in the other format
    if format == 'csv':
        json_output = os.path.splitext(args.output_file)[0] + '.json'
        save_results(results, json_output, 'json')
    else:
        csv_output = os.path.splitext(args.output_file)[0] + '.csv'
        save_results(results, csv_output, 'csv')
    
    print("✅ Summary report generation completed!")

if __name__ == "__main__":
    main() 