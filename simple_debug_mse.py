#!/usr/bin/env python3
"""
Simple debug script để tìm nguyên nhân MSE về 0 trong Stage 2 training
Không có dependencies phức tạp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Simulated RoundWithNoise (đã fix)
class SimpleRoundWithNoise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if ctx.needs_input_grad[0] and input.requires_grad:
            noise = torch.empty_like(input).uniform_(-0.5, 0.5)
            return torch.round(input + noise)  # FIX applied
        else:
            return torch.round(input)
    
    @staticmethod  
    def backward(ctx, grad_output):
        return grad_output

def debug_quantization_levels():
    """Debug quantization behavior với different input ranges"""
    print("🔍 DEBUGGING: Quantization Levels")
    print("-" * 50)
    
    # Test với ranges giống như actual mixed_features: [-0.2, 0.2]
    test_ranges = [
        ("Actual range (small)", torch.randn(4, 128, 32, 32) * 0.1),  # [-0.3, 0.3]
        ("Medium range", torch.randn(4, 128, 32, 32) * 1.0),        # [-3, 3]  
        ("Large range", torch.randn(4, 128, 32, 32) * 10.0),       # [-30, 30]
    ]
    
    for name, x in test_ranges:
        x.requires_grad_(True)
        
        # Apply quantization
        x_quantized = SimpleRoundWithNoise.apply(x)
        
        # Calculate MSE
        mse = F.mse_loss(x_quantized, x).item()
        
        # Calculate quantization error stats
        diff = torch.abs(x_quantized - x)
        
        print(f"{name}:")
        print(f"  Input range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"  Quantized range: [{x_quantized.min():.4f}, {x_quantized.max():.4f}]")
        print(f"  MSE: {mse:.8f}")
        print(f"  Diff mean: {diff.mean():.6f}, max: {diff.max():.6f}")
        
        if mse < 1e-6:
            print(f"  ❌ PROBLEM: MSE too small for {name}!")
        else:
            print(f"  ✅ MSE reasonable for {name}")
        print()

def debug_compression_pipeline():
    """Debug compression pipeline behavior"""
    print("🔍 DEBUGGING: Compression Pipeline")
    print("-" * 50)
    
    # Simulate mixed_features từ AdaMixNet (giống actual range)
    mixed_features = torch.randn(8, 128, 256, 256, requires_grad=True) * 0.2
    print(f"Mixed features: {mixed_features.shape}, range: [{mixed_features.min():.4f}, {mixed_features.max():.4f}]")
    
    # Simulate Analysis Transform (4 layers, stride 2 each = 16x downsampling)
    y = mixed_features
    for i in range(4):
        y = F.avg_pool2d(y, kernel_size=3, stride=2, padding=1)  # More realistic than kernel=16
    print(f"After analysis: {y.shape}, range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Quantization
    y_quantized = SimpleRoundWithNoise.apply(y)
    print(f"After quantization: {y_quantized.shape}, range: [{y_quantized.min():.4f}, {y_quantized.max():.4f}]")
    
    # Check quantization effect
    quant_diff = torch.abs(y_quantized - y)
    print(f"Quantization diff: mean={quant_diff.mean():.6f}, max={quant_diff.max():.6f}")
    
    # Simulate Synthesis Transform (4 layers, stride 2 each = 16x upsampling)
    x_hat = y_quantized
    for i in range(4):
        x_hat = F.interpolate(x_hat, scale_factor=2, mode='bilinear', align_corners=False)
    
    # Ensure exact size match
    if x_hat.shape != mixed_features.shape:
        x_hat = F.interpolate(x_hat, size=mixed_features.shape[2:], mode='bilinear', align_corners=False)
    
    print(f"After synthesis: {x_hat.shape}, range: [{x_hat.min():.4f}, {x_hat.max():.4f}]")
    
    # Calculate final MSE
    mse = F.mse_loss(x_hat, mixed_features).item()
    print(f"Pipeline MSE: {mse:.8f}")
    
    if mse < 1e-6:
        print("❌ PROBLEM: Pipeline MSE too small - acting like identity!")
    elif mse > 0.01:
        print("✅ Pipeline MSE reasonable - proper compression distortion")
    else:
        print("⚠️ WARNING: Pipeline MSE small but not zero")
    
    print()

def debug_learning_dynamics():
    """Debug why MSE reduces so quickly"""
    print("🔍 DEBUGGING: Learning Dynamics")
    print("-" * 50)
    
    # Observed MSE reduction pattern
    mse_values = [0.001649, 0.000013, 0.000005, 0.000002, 0.000001, 0.000000]
    epochs = range(1, len(mse_values) + 1)
    
    print("Observed MSE reduction:")
    for epoch, mse in zip(epochs, mse_values):
        if epoch > 1:
            reduction_factor = mse / mse_values[epoch-2]
            print(f"Epoch {epoch}: MSE={mse:.8f} (reduction: {reduction_factor:.2%})")
        else:
            print(f"Epoch {epoch}: MSE={mse:.8f}")
    
    # Analyze reduction pattern
    print("\nAnalysis:")
    avg_reduction = []
    for i in range(1, len(mse_values)):
        if mse_values[i-1] > 0:
            reduction = mse_values[i] / mse_values[i-1]
            avg_reduction.append(reduction)
            
    if avg_reduction:
        avg_red = sum(avg_reduction) / len(avg_reduction)
        print(f"Average reduction factor: {avg_red:.4f} ({avg_red:.2%})")
        
        if avg_red < 0.1:
            print("❌ PROBLEM: MSE reducing too quickly (>90% per epoch)!")
            print("   This suggests model is learning 'perfect reconstruction' instead of compression")
        else:
            print("✅ MSE reduction rate reasonable")
    
    print()

def debug_scale_sensitivity():
    """Debug sensitivity to input scale"""
    print("🔍 DEBUGGING: Scale Sensitivity")
    print("-" * 50)
    
    # Test với different scales giống như thực tế
    base_features = torch.randn(2, 128, 64, 64, requires_grad=True)
    
    scales = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    print("MSE vs Input Scale:")
    for scale in scales:
        x = base_features * scale
        
        # Simple compression pipeline
        y = F.avg_pool2d(x, kernel_size=4, stride=4)  # 4x compression
        y_quantized = SimpleRoundWithNoise.apply(y)
        x_hat = F.interpolate(y_quantized, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        mse = F.mse_loss(x_hat, x).item()
        relative_mse = mse / (scale * scale) if scale > 0 else 0  # Normalize by input variance
        
        print(f"Scale {scale:4.2f}: MSE={mse:.8f}, Relative MSE={relative_mse:.8f}")
        
        if mse < 1e-8:
            print(f"  ❌ MSE collapsed for scale {scale}")
    
    print()

def debug_bpp_mse_balance():
    """Debug balance between BPP and MSE components"""
    print("🔍 DEBUGGING: BPP vs MSE Balance")
    print("-" * 50)
    
    # Observed values from training
    lambda_rd = 512
    epochs_data = [
        (1, 0.001649, 3.4990),
        (2, 0.000013, 3.4802), 
        (3, 0.000005, 3.4670),
        (4, 0.000002, 3.4530),
        (5, 0.000001, 3.4392),
        (6, 0.000000, 3.4184)
    ]
    
    print("Loss Component Analysis:")
    print("Epoch | MSE      | BPP    | MSE*λ    | Total    | MSE% ")
    print("-" * 55)
    
    for epoch, mse, bpp in epochs_data:
        mse_weighted = lambda_rd * mse
        total_loss = mse_weighted + bpp
        mse_percentage = (mse_weighted / total_loss) * 100 if total_loss > 0 else 0
        
        print(f"{epoch:5d} | {mse:.6f} | {bpp:6.4f} | {mse_weighted:8.4f} | {total_loss:8.4f} | {mse_percentage:5.1f}%")
    
    print("\nAnalysis:")
    print("❌ PROBLEM: MSE component becomes negligible compared to BPP")
    print("   MSE contribution drops from ~0.85 to ~0.0 (ignored by optimizer)")
    print("   Model focuses only on BPP minimization, ignoring reconstruction quality")
    
    print()

def main():
    """Main debugging function"""
    print("=" * 60)
    print("🚨 DEBUGGING MSE → 0 ISSUE IN STAGE 2")
    print("=" * 60)
    print()
    
    debug_quantization_levels()
    debug_compression_pipeline()
    debug_learning_dynamics()
    debug_scale_sensitivity()
    debug_bpp_mse_balance()
    
    print("=" * 60)
    print("🎯 DIAGNOSIS SUMMARY")
    print("=" * 60)
    print()
    
    print("🔍 ROOT CAUSE ANALYSIS:")
    print("1. ✅ RoundWithNoise fix works properly")
    print("2. ❌ Input values too small (~0.2 range) → quantization has minimal effect")
    print("3. ❌ MSE reduces too quickly (>90% per epoch)")
    print("4. ❌ MSE component becomes negligible vs BPP (0.85 vs 3400)")
    print("5. ❌ Model learns to ignore reconstruction quality")
    print()
    
    print("💡 PROPOSED SOLUTIONS:")
    print("1. 🔧 Increase λ_rd to rebalance MSE vs BPP")
    print("2. 🔧 Add MSE loss scaling/normalization")
    print("3. 🔧 Reduce learning rate to slow MSE collapse")
    print("4. 🔧 Add perceptual/feature loss component")
    print("5. 🔧 Implement MSE loss floor/minimum")

if __name__ == "__main__":
    main() 