"""
Advanced Quantizer Fix - Comprehensive solution for small input ranges
Based on test results: Scale factor 4.0 not enough, need adaptive scaling
"""

import torch
import torch.nn as nn
from models.compressor_vnvc import CompressorVNVC
from models.adamixnet import AdaMixNet
from models.wavelet_transform_cnn import WaveletTransformCNN

def analyze_pipeline_ranges():
    """Analyze ranges through entire pipeline to find bottleneck"""
    
    print("🔬 COMPREHENSIVE PIPELINE ANALYSIS")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create full pipeline
    wavelet_cnn = WaveletTransformCNN(input_channels=3, wavelet_channels=64).to(device)
    adamixnet = AdaMixNet(input_channels=256, output_channels=128).to(device)  # 4*64=256
    compressor = CompressorVNVC(input_channels=128, latent_channels=192, lambda_rd=128).to(device)
    
    # Set to eval mode
    wavelet_cnn.eval()
    adamixnet.eval() 
    compressor.eval()
    
    # Test input
    test_input = torch.randn(1, 3, 256, 256).to(device)
    print(f"Original input: [{test_input.min():.6f}, {test_input.max():.6f}]")
    
    with torch.no_grad():
        # Step 1: Wavelet
        wavelet_out = wavelet_cnn(test_input)
        print(f"After Wavelet: {wavelet_out.shape}, [{wavelet_out.min():.6f}, {wavelet_out.max():.6f}]")
        
        # Step 2: AdaMixNet  
        mixed_out = adamixnet(wavelet_out)
        print(f"After AdaMixNet: {mixed_out.shape}, [{mixed_out.min():.6f}, {mixed_out.max():.6f}]")
        
        # Step 3: Analysis Transform
        analysis_out = compressor.analysis_transform(mixed_out)
        print(f"After Analysis: {analysis_out.shape}, [{analysis_out.min():.6f}, {analysis_out.max():.6f}]")
        
        # Step 4: Quantizer
        quantized_out = compressor.quantizer(analysis_out)
        print(f"After Quantizer: {quantized_out.shape}, [{quantized_out.min():.6f}, {quantized_out.max():.6f}]")
        print(f"Quantized unique: {torch.unique(quantized_out).numel()}")
        print(f"Quantized non-zero: {(quantized_out != 0).sum()}/{quantized_out.numel()}")
        
        # Identify bottleneck
        ranges = {
            'wavelet': wavelet_out.abs().max().item(),
            'adamix': mixed_out.abs().max().item(), 
            'analysis': analysis_out.abs().max().item(),
            'quantized': quantized_out.abs().max().item()
        }
        
        print(f"\n📊 RANGE ANALYSIS:")
        for stage, max_val in ranges.items():
            print(f"  {stage}: max_abs = {max_val:.6f}")
            
        # Find where range collapses
        if ranges['analysis'] < 0.1:
            print(f"🚨 BOTTLENECK FOUND: Analysis transform collapses range to {ranges['analysis']:.6f}")
            return 'analysis_transform'
        elif ranges['quantized'] == 0:
            print(f"🚨 BOTTLENECK FOUND: Quantizer kills all values")
            return 'quantizer'
        else:
            print(f"✅ Pipeline preserves ranges adequately")
            return 'none'

def create_adaptive_quantizer():
    """Create improved quantizer with adaptive scaling"""
    
    class AdaptiveRoundWithNoise(torch.autograd.Function):
        """Adaptive quantizer that scales based on input range"""
        
        @staticmethod
        def forward(ctx, input):
            # Calculate adaptive scale factor based on input range
            input_range = input.abs().max()
            
            if input_range < 0.01:      # Very tiny values
                scale_factor = 20.0
            elif input_range < 0.1:     # Small values  
                scale_factor = 10.0
            elif input_range < 1.0:     # Medium values
                scale_factor = 4.0
            else:                       # Large values
                scale_factor = 1.0
                
            # Store scale factor for backward
            ctx.scale_factor = scale_factor
            
            # Scale up
            scaled_input = input * scale_factor
            
            # Quantize
            if ctx.needs_input_grad[0] and input.requires_grad:
                noise = torch.empty_like(scaled_input).uniform_(-0.5, 0.5)
                quantized = torch.round(scaled_input + noise)
            else:
                quantized = torch.round(scaled_input)
                
            # Scale back down
            return quantized / scale_factor
        
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output
    
    class AdaptiveQuantizerVNVC(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x):
            return AdaptiveRoundWithNoise.apply(x)
            
        def quantize(self, x):
            # Same adaptive logic for inference
            input_range = x.abs().max()
            if input_range < 0.01:
                scale_factor = 20.0
            elif input_range < 0.1:
                scale_factor = 10.0  
            elif input_range < 1.0:
                scale_factor = 4.0
            else:
                scale_factor = 1.0
                
            scaled = x * scale_factor
            quantized = torch.round(scaled)
            return quantized / scale_factor
    
    return AdaptiveQuantizerVNVC()

def test_adaptive_quantizer():
    """Test adaptive quantizer on problematic ranges"""
    
    print(f"\n🧪 TESTING ADAPTIVE QUANTIZER")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adaptive_quantizer = create_adaptive_quantizer().to(device)
    
    test_cases = [
        ('Ultra Tiny', 0.005),
        ('Tiny', 0.01), 
        ('Small', 0.05),
        ('Medium', 0.2),
        ('Normal', 1.0),
        ('Large', 5.0)
    ]
    
    for name, scale in test_cases:
        test_input = torch.randn(1, 192, 8, 8).to(device) * scale
        
        with torch.no_grad():
            quantized = adaptive_quantizer(test_input)
            
            print(f"\n📊 {name} (scale={scale}):")
            print(f"  Input: [{test_input.min():.6f}, {test_input.max():.6f}]")
            print(f"  Output: [{quantized.min():.6f}, {quantized.max():.6f}]") 
            print(f"  Unique: {torch.unique(quantized).numel()}")
            print(f"  Non-zero: {(quantized != 0).float().mean():.4f}")
            
            if (quantized == 0).all():
                print("  ❌ Still all zeros")
            elif (quantized != 0).float().mean() > 0.5:
                print("  ✅ Good preservation")
            else:
                print("  ⚠️ Needs improvement")

def propose_analysis_transform_fix():
    """Propose fix for analysis transform if it's the bottleneck"""
    
    print(f"\n💡 ANALYSIS TRANSFORM FIX PROPOSAL:")
    print("="*50)
    
    print("Current Analysis Transform:")
    print("  conv(128→192, k=5, s=2) → /2")  
    print("  conv(192→192, k=5, s=2) → /4")
    print("  conv(192→192, k=3, s=1)")
    print("  conv(192→192, k=3, s=1)")
    print("  → Result: /4 spatial, range collapse")
    
    print(f"\nProposed Fix:")
    print("1. Add BatchNorm sau mỗi conv để preserve range")
    print("2. Thêm skip connections để maintain information")  
    print("3. Reduce depth từ 4 layers xuống 3 layers")
    print("4. Use GroupNorm thay vì no normalization")
    
    print(f"\nCode changes needed:")
    print("- Add nn.GroupNorm(8, channels) sau conv layers")
    print("- Implement residual connections")
    print("- Adjust quantizer scale based on analysis output")

if __name__ == "__main__":
    # Step 1: Find bottleneck
    bottleneck = analyze_pipeline_ranges()
    
    # Step 2: Test adaptive quantizer
    test_adaptive_quantizer()
    
    # Step 3: Propose comprehensive fix
    if bottleneck == 'analysis_transform':
        propose_analysis_transform_fix()
    elif bottleneck == 'quantizer':  
        print(f"\n✅ Use adaptive quantizer as primary fix")
    else:
        print(f"\n🤔 Need further investigation")
    
    print(f"\n📋 RECOMMENDED ACTION:")
    print("1. Implement adaptive quantizer with scale factors 1.0-20.0")
    print("2. Add normalization to analysis transform") 
    print("3. Test full pipeline với fixed components")
    print("4. Retrain Stage 2 nếu architectural changes")
    
    print(f"\n✅ Advanced analysis completed") 