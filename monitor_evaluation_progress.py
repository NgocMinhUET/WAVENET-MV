"""
Monitor Full Lambda Evaluation Progress
Hiển thị progress và partial results trong quá trình evaluation
"""

import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_partial_results():
    """Load và combine tất cả partial results hiện có"""
    results_dir = Path('results')
    if not results_dir.exists():
        return None
    
    # Find all lambda evaluation files
    lambda_files = list(results_dir.glob('wavenet_mv_lambda*_evaluation.csv'))
    if not lambda_files:
        return None
    
    # Load and combine
    dfs = []
    for file in lambda_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except:
            continue
    
    if not dfs:
        return None
    
    return pd.concat(dfs, ignore_index=True)

def plot_partial_rd_curve(df):
    """Plot partial rate-distortion curve"""
    if df is None or df.empty:
        return
    
    plt.figure(figsize=(10, 6))
    
    # Sort by lambda for proper line
    df_sorted = df.sort_values('lambda')
    
    # Plot points
    plt.plot(df_sorted['bpp'], df_sorted['psnr_db'], 
             'o-', linewidth=2, markersize=8, label='WAVENET-MV')
    
    # Add lambda labels
    for _, row in df_sorted.iterrows():
        plt.annotate(f'λ={int(row["lambda"])}', 
                    (row['bpp'], row['psnr_db']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Rate (BPP)')
    plt.ylabel('PSNR (dB)')
    plt.title('Partial Rate-Distortion Curve')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('results/partial_rd_curve.png')
    plt.close()

def create_progress_summary(df):
    """Create progress summary"""
    if df is None:
        print("\n❌ No results found yet")
        return
    
    print("\n📊 EVALUATION PROGRESS")
    print("=" * 50)
    
    # Expected lambda values
    expected = {64, 128, 256, 512, 1024}
    completed = set(df['lambda'].unique())
    remaining = expected - completed
    
    print(f"✅ Completed λ values: {sorted(completed)}")
    print(f"⏳ Remaining λ values: {sorted(remaining)}")
    print(f"📈 Progress: {len(completed)}/{len(expected)} ({len(completed)*100/len(expected):.1f}%)")
    
    if not df.empty:
        print("\n📋 Current Results:")
        summary = df.groupby('lambda').agg({
            'psnr_db': 'mean',
            'ms_ssim': 'mean',
            'bpp': 'mean'
        }).round(3)
        print(summary)

def main():
    print("🔍 MONITORING FULL LAMBDA EVALUATION PROGRESS")
    print("============================================")
    
    try:
        while True:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Load current results
            df = load_partial_results()
            
            # Create summary
            create_progress_summary(df)
            
            # Update plot if we have data
            if df is not None and not df.empty:
                plot_partial_rd_curve(df)
                print("\n📈 Rate-distortion plot updated: results/partial_rd_curve.png")
            
            # Check if all lambdas completed
            if df is not None and len(df['lambda'].unique()) == 5:
                print("\n🎉 EVALUATION COMPLETED!")
                break
            
            print("\n⏳ Waiting for more results... (Press Ctrl+C to stop)")
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")
        sys.exit(0)

if __name__ == '__main__':
    main() 