#!/usr/bin/env python3
"""
Vẽ đồ thị AI Accuracy vs BPP từ dữ liệu bảng đã cập nhật
WAVENET-MV Research Team
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Dữ liệu từ bảng đã cập nhật
jpeg_data = {
    'quality': [10, 30, 50, 70, 90, 95],
    'psnr': [30.6, 32.2, 33.0, 33.9, 37.4, 40.1],
    'ssim': [0.734, 0.847, 0.884, 0.912, 0.960, 0.979],
    'bpp': [0.31, 0.71, 1.01, 1.42, 2.56, 3.67],
    'ai_accuracy': [0.587, 0.638, 0.666, 0.698, 0.770, 0.803]
}

wavenet_data = {
    'lambda': [64, 128, 256, 512, 1024, 2048],
    'psnr': [22.8, 26.2, 27.4, 30.6, 31.1, 33.8],
    'ssim': [0.692, 0.728, 0.776, 0.812, 0.845, 0.873],
    'bpp': [0.54, 0.89, 1.34, 1.78, 2.51, 3.42],
    'ai_accuracy': [0.698, 0.779, 0.792, 0.808, 0.821, 0.845]
}

# Tạo DataFrame
jpeg_df = pd.DataFrame(jpeg_data)
wavenet_df = pd.DataFrame(wavenet_data)

# Cài đặt matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

def create_accuracy_bpp_plot():
    """Tạo đồ thị AI Accuracy vs BPP"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Vẽ đường cho JPEG
    ax.plot(jpeg_df['bpp'], jpeg_df['ai_accuracy'], 
            marker='o', linewidth=2.5, markersize=8,
            color='#1f77b4', label='JPEG Baseline',
            markerfacecolor='white', markeredgewidth=2)
    
    # Vẽ đường cho WAVENET-MV
    ax.plot(wavenet_df['bpp'], wavenet_df['ai_accuracy'], 
            marker='s', linewidth=2.5, markersize=8,
            color='#ff7f0e', label='WAVENET-MV',
            markerfacecolor='white', markeredgewidth=2)
    
    # Annotation cho các điểm quan trọng
    # JPEG points
    for i, (bpp, acc, q) in enumerate(zip(jpeg_df['bpp'], jpeg_df['ai_accuracy'], jpeg_df['quality'])):
        ax.annotate(f'Q={q}', (bpp, acc), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, alpha=0.7)
    
    # WAVENET-MV points
    for i, (bpp, acc, lam) in enumerate(zip(wavenet_df['bpp'], wavenet_df['ai_accuracy'], wavenet_df['lambda'])):
        ax.annotate(f'λ={lam}', (bpp, acc), 
                   xytext=(5, -15), textcoords='offset points',
                   fontsize=10, alpha=0.7)
    
    # Tính toán cải thiện
    improvements = []
    for i, wb in enumerate(wavenet_df['bpp']):
        # Tìm JPEG point gần nhất
        closest_idx = np.argmin(np.abs(jpeg_df['bpp'] - wb))
        jpeg_acc = jpeg_df['ai_accuracy'].iloc[closest_idx]
        wavenet_acc = wavenet_df['ai_accuracy'].iloc[i]
        improvement = (wavenet_acc - jpeg_acc) * 100
        improvements.append(improvement)
    
    avg_improvement = np.mean(improvements)
    
    # Thiết lập trục
    ax.set_xlabel('Bits Per Pixel (BPP)', fontweight='bold')
    ax.set_ylabel('AI Accuracy (mAP)', fontweight='bold')
    ax.set_title('AI Accuracy vs BPP: WAVENET-MV vs JPEG Baseline', 
                fontweight='bold', pad=20)
    
    # Thêm grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Thiết lập range
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0.55, 0.87)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # Thêm text box với thông tin cải thiện
    textstr = f'Avg Improvement: +{avg_improvement:.1f}% mAP'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Lưu đồ thị
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'ai_accuracy_vs_bpp_updated.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ai_accuracy_vs_bpp_updated.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'ai_accuracy_vs_bpp_updated.svg', bbox_inches='tight')
    
    print(f"✅ Đồ thị đã được lưu tại {output_dir}/")
    return fig

def create_comparison_table():
    """Tạo bảng so sánh chi tiết"""
    
    # Tạo bảng so sánh
    comparison_data = []
    
    for i, (wb, wa) in enumerate(zip(wavenet_df['bpp'], wavenet_df['ai_accuracy'])):
        # Tìm JPEG point gần nhất
        closest_idx = np.argmin(np.abs(jpeg_df['bpp'] - wb))
        jpeg_bpp = jpeg_df['bpp'].iloc[closest_idx]
        jpeg_acc = jpeg_df['ai_accuracy'].iloc[closest_idx]
        jpeg_q = jpeg_df['quality'].iloc[closest_idx]
        
        # Tính toán cải thiện
        acc_improvement = (wa - jpeg_acc) * 100
        bpp_overhead = ((wb - jpeg_bpp) / jpeg_bpp) * 100
        
        comparison_data.append({
            'WAVENET_Lambda': wavenet_df['lambda'].iloc[i],
            'WAVENET_BPP': wb,
            'WAVENET_Accuracy': wa,
            'JPEG_Quality': jpeg_q,
            'JPEG_BPP': jpeg_bpp,
            'JPEG_Accuracy': jpeg_acc,
            'Accuracy_Improvement_Pct': acc_improvement,
            'BPP_Overhead_Pct': bpp_overhead
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Lưu bảng
    output_dir = Path('tables')
    output_dir.mkdir(exist_ok=True)
    
    comparison_df.to_csv(output_dir / 'accuracy_bpp_comparison_updated.csv', index=False)
    
    print(f"✅ Bảng so sánh đã được lưu tại {output_dir}/")
    
    # In thống kê
    print("\n🔍 THỐNG KÊ CẢI THIỆN:")
    print(f"   • Trung bình cải thiện AI accuracy: {comparison_df['Accuracy_Improvement_Pct'].mean():.1f}%")
    print(f"   • Trung bình overhead BPP: {comparison_df['BPP_Overhead_Pct'].mean():.1f}%")
    print(f"   • Cải thiện cao nhất: {comparison_df['Accuracy_Improvement_Pct'].max():.1f}%")
    print(f"   • Cải thiện thấp nhất: {comparison_df['Accuracy_Improvement_Pct'].min():.1f}%")
    
    return comparison_df

def create_detailed_analysis():
    """Tạo phân tích chi tiết"""
    
    print("\n📊 PHÂN TÍCH CHI TIẾT AI ACCURACY vs BPP:")
    print("=" * 60)
    
    print("\n🔵 JPEG Baseline:")
    for i, row in jpeg_df.iterrows():
        print(f"   Q={int(row['quality']):2d}: BPP={row['bpp']:.2f}, Accuracy={row['ai_accuracy']:.3f}")
    
    print("\n🔶 WAVENET-MV:")
    for i, row in wavenet_df.iterrows():
        print(f"   λ={int(row['lambda']):4d}: BPP={row['bpp']:.2f}, Accuracy={row['ai_accuracy']:.3f}")
    
    print("\n📈 CẢI THIỆN THEO TỪNG KHOẢNG BPP:")
    print("=" * 60)
    
    # Phân tích theo khoảng BPP
    bpp_ranges = [
        (0.5, 1.0, "Low BPP"),
        (1.0, 2.0, "Medium BPP"), 
        (2.0, 4.0, "High BPP")
    ]
    
    for bpp_min, bpp_max, label in bpp_ranges:
        jpeg_subset = jpeg_df[(jpeg_df['bpp'] >= bpp_min) & (jpeg_df['bpp'] < bpp_max)]
        wavenet_subset = wavenet_df[(wavenet_df['bpp'] >= bpp_min) & (wavenet_df['bpp'] < bpp_max)]
        
        if len(jpeg_subset) > 0 and len(wavenet_subset) > 0:
            jpeg_avg = jpeg_subset['ai_accuracy'].mean()
            wavenet_avg = wavenet_subset['ai_accuracy'].mean()
            improvement = (wavenet_avg - jpeg_avg) * 100
            
            print(f"\n{label} ({bpp_min}-{bpp_max}):")
            print(f"   • JPEG avg accuracy: {jpeg_avg:.3f}")
            print(f"   • WAVENET-MV avg accuracy: {wavenet_avg:.3f}")
            print(f"   • Improvement: +{improvement:.1f}%")

if __name__ == "__main__":
    print("🎨 Tạo đồ thị AI Accuracy vs BPP từ dữ liệu đã cập nhật...")
    
    # Tạo đồ thị
    fig = create_accuracy_bpp_plot()
    
    # Tạo bảng so sánh
    comparison_df = create_comparison_table()
    
    # Phân tích chi tiết
    create_detailed_analysis()
    
    print("\n✅ Hoàn thành tất cả!")
    print(f"   📁 Đồ thị: figures/ai_accuracy_vs_bpp_updated.png")
    print(f"   📁 Bảng so sánh: tables/accuracy_bpp_comparison_updated.csv")
    
    # Hiển thị đồ thị
    plt.show() 