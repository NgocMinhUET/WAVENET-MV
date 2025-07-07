#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Statistical Analysis
-------------------
Script này phân tích thống kê kết quả và đóng góp của các thành phần
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description='Statistical analysis of evaluation results')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input CSV file with comprehensive results')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file for analysis results')
    parser.add_argument('--analysis_type', type=str, choices=['wavelet', 'lambda', 'all'], default='all',
                        help='Type of analysis to perform')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    return parser.parse_args()

def load_data(input_file):
    """Load data from input file"""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.json'):
        with open(input_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {input_file}")
    
    return df

def analyze_wavelet_contribution(df):
    """Analyze the contribution of wavelet transform"""
    # Filter WAVENET-MV results
    wavenet_df = df[df['method'] == 'WAVENET-MV'].copy()
    
    # Check if we have data for WAVENET-MV without wavelet
    no_wavelet_df = df[df['method'] == 'WAVENET-MV (No Wavelet)'].copy() if 'WAVENET-MV (No Wavelet)' in df['method'].values else None
    
    results = []
    
    if no_wavelet_df is not None and not no_wavelet_df.empty:
        # For each lambda value in WAVENET-MV, find the closest BPP in No Wavelet
        for _, row in wavenet_df.iterrows():
            lambda_val = row['lambda']
            bpp = row['bpp']
            psnr = row['psnr_db']
            ms_ssim = row['ms_ssim']
            ai_acc = row['ai_accuracy'] if 'ai_accuracy' in row else None
            
            # Find the closest BPP in No Wavelet
            if not no_wavelet_df.empty:
                no_wavelet_df['bpp_diff'] = abs(no_wavelet_df['bpp'] - bpp)
                closest_row = no_wavelet_df.loc[no_wavelet_df['bpp_diff'].idxmin()]
                
                # Calculate improvements
                psnr_gain = psnr - closest_row['psnr_db']
                ms_ssim_gain = ms_ssim - closest_row['ms_ssim']
                ai_gain = (ai_acc - closest_row['ai_accuracy']) if ai_acc is not None and 'ai_accuracy' in closest_row else None
                
                result = {
                    'lambda': lambda_val,
                    'bpp': bpp,
                    'psnr_gain': round(psnr_gain, 2),
                    'ms_ssim_gain': round(ms_ssim_gain, 4),
                    'psnr_gain_percent': round((psnr_gain / closest_row['psnr_db']) * 100, 1),
                    'ms_ssim_gain_percent': round((ms_ssim_gain / closest_row['ms_ssim']) * 100, 1)
                }
                
                if ai_gain is not None:
                    result['ai_gain'] = round(ai_gain, 3)
                    result['ai_gain_percent'] = round((ai_gain / closest_row['ai_accuracy']) * 100, 1)
                
                results.append(result)
    else:
        # If we don't have No Wavelet data, just report the performance
        for _, row in wavenet_df.iterrows():
            lambda_val = row['lambda']
            bpp = row['bpp']
            psnr = row['psnr_db']
            ms_ssim = row['ms_ssim']
            ai_acc = row['ai_accuracy'] if 'ai_accuracy' in row else None
            
            result = {
                'lambda': lambda_val,
                'bpp': bpp,
                'psnr': psnr,
                'ms_ssim': ms_ssim
            }
            
            if ai_acc is not None:
                result['ai_accuracy'] = ai_acc
            
            results.append(result)
    
    # Calculate average gains
    if results and 'psnr_gain' in results[0]:
        avg_psnr_gain = sum(r['psnr_gain'] for r in results) / len(results)
        avg_ms_ssim_gain = sum(r['ms_ssim_gain'] for r in results) / len(results)
        avg_psnr_percent = sum(r['psnr_gain_percent'] for r in results) / len(results)
        avg_ms_ssim_percent = sum(r['ms_ssim_gain_percent'] for r in results) / len(results)
        
        summary = {
            'avg_psnr_gain': round(avg_psnr_gain, 2),
            'avg_ms_ssim_gain': round(avg_ms_ssim_gain, 4),
            'avg_psnr_gain_percent': round(avg_psnr_percent, 1),
            'avg_ms_ssim_gain_percent': round(avg_ms_ssim_percent, 1)
        }
        
        if 'ai_gain' in results[0]:
            avg_ai_gain = sum(r['ai_gain'] for r in results) / len(results)
            avg_ai_percent = sum(r['ai_gain_percent'] for r in results) / len(results)
            summary['avg_ai_gain'] = round(avg_ai_gain, 3)
            summary['avg_ai_gain_percent'] = round(avg_ai_percent, 1)
        
        results.append(summary)
    
    return results

def analyze_lambda_impact(df):
    """Analyze the impact of lambda value on rate-distortion performance"""
    # Filter WAVENET-MV results
    wavenet_df = df[df['method'] == 'WAVENET-MV'].copy()
    
    if wavenet_df.empty:
        return []
    
    # Sort by lambda value
    wavenet_df = wavenet_df.sort_values('lambda')
    
    results = []
    
    # Calculate BD-Rate and BD-PSNR between consecutive lambda values
    prev_row = None
    for _, row in wavenet_df.iterrows():
        lambda_val = row['lambda']
        bpp = row['bpp']
        psnr = row['psnr_db']
        ms_ssim = row['ms_ssim']
        
        result = {
            'lambda': lambda_val,
            'bpp': bpp,
            'psnr_db': psnr,
            'ms_ssim': ms_ssim
        }
        
        if prev_row is not None:
            # Calculate rate change
            bpp_change = bpp - prev_row['bpp']
            bpp_change_percent = (bpp_change / prev_row['bpp']) * 100
            
            # Calculate distortion change
            psnr_change = psnr - prev_row['psnr_db']
            ms_ssim_change = ms_ssim - prev_row['ms_ssim']
            
            # Calculate rate-distortion efficiency
            rd_efficiency_psnr = psnr_change / bpp_change if bpp_change != 0 else float('inf')
            rd_efficiency_ms_ssim = ms_ssim_change / bpp_change if bpp_change != 0 else float('inf')
            
            result['bpp_change'] = round(bpp_change, 2)
            result['bpp_change_percent'] = round(bpp_change_percent, 1)
            result['psnr_change'] = round(psnr_change, 2)
            result['ms_ssim_change'] = round(ms_ssim_change, 4)
            result['rd_efficiency_psnr'] = round(rd_efficiency_psnr, 2)
            result['rd_efficiency_ms_ssim'] = round(rd_efficiency_ms_ssim, 4)
        
        results.append(result)
        prev_row = row
    
    return results

def visualize_results(df, analysis_type, output_dir):
    """Generate visualization plots for the results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    if analysis_type in ['wavelet', 'all']:
        # Filter WAVENET-MV results
        wavenet_df = df[df['method'] == 'WAVENET-MV'].copy()
        no_wavelet_df = df[df['method'] == 'WAVENET-MV (No Wavelet)'].copy() if 'WAVENET-MV (No Wavelet)' in df['method'].values else None
        
        if not wavenet_df.empty:
            # Rate-distortion curve
            plt.figure(figsize=(10, 6))
            plt.plot(wavenet_df['bpp'], wavenet_df['psnr_db'], 'o-', label='WAVENET-MV', linewidth=2)
            
            if no_wavelet_df is not None and not no_wavelet_df.empty:
                plt.plot(no_wavelet_df['bpp'], no_wavelet_df['psnr_db'], 's--', label='WAVENET-MV (No Wavelet)', linewidth=2)
            
            plt.xlabel('Bits per Pixel (BPP)')
            plt.ylabel('PSNR (dB)')
            plt.title('Rate-Distortion Performance')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'wavelet_rd_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # MS-SSIM curve
            plt.figure(figsize=(10, 6))
            plt.plot(wavenet_df['bpp'], wavenet_df['ms_ssim'], 'o-', label='WAVENET-MV', linewidth=2)
            
            if no_wavelet_df is not None and not no_wavelet_df.empty:
                plt.plot(no_wavelet_df['bpp'], no_wavelet_df['ms_ssim'], 's--', label='WAVENET-MV (No Wavelet)', linewidth=2)
            
            plt.xlabel('Bits per Pixel (BPP)')
            plt.ylabel('MS-SSIM')
            plt.title('Rate-MS-SSIM Performance')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'wavelet_ms_ssim_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    if analysis_type in ['lambda', 'all']:
        # Filter WAVENET-MV results
        wavenet_df = df[df['method'] == 'WAVENET-MV'].copy()
        
        if not wavenet_df.empty:
            # Lambda vs. BPP
            plt.figure(figsize=(10, 6))
            plt.plot(wavenet_df['lambda'], wavenet_df['bpp'], 'o-', linewidth=2)
            plt.xlabel('Lambda Value')
            plt.ylabel('Bits per Pixel (BPP)')
            plt.title('Impact of Lambda on Bit Rate')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'lambda_vs_bpp.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Lambda vs. PSNR
            plt.figure(figsize=(10, 6))
            plt.plot(wavenet_df['lambda'], wavenet_df['psnr_db'], 'o-', linewidth=2)
            plt.xlabel('Lambda Value')
            plt.ylabel('PSNR (dB)')
            plt.title('Impact of Lambda on PSNR')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'lambda_vs_psnr.png'), dpi=300, bbox_inches='tight')
            plt.close()

def save_results(results, output_file):
    """Save analysis results to a file"""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if output_file.endswith('.csv'):
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
    elif output_file.endswith('.json'):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        # Default to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
    
    print(f"✅ Analysis results saved to {output_file}")

def main():
    args = parse_args()
    
    print(f"🔍 Loading data from {args.input_file}...")
    df = load_data(args.input_file)
    
    results = []
    
    if args.analysis_type in ['wavelet', 'all']:
        print("🔄 Analyzing wavelet contribution...")
        wavelet_results = analyze_wavelet_contribution(df)
        if wavelet_results:
            results = wavelet_results
    
    if args.analysis_type in ['lambda', 'all']:
        print("🔄 Analyzing lambda impact...")
        lambda_results = analyze_lambda_impact(df)
        if lambda_results and not results:
            results = lambda_results
    
    print(f"🔄 Saving results to {args.output_file}...")
    save_results(results, args.output_file)
    
    if args.visualize:
        output_dir = os.path.dirname(args.output_file)
        print("🔄 Generating visualization plots...")
        visualize_results(df, args.analysis_type, output_dir)
    
    print("✅ Statistical analysis completed!")

if __name__ == "__main__":
    main() 