"""
Fix Quantizer Issue - Investigate why quantized features are all zeros
Based on debug output: y_quantized range: [-0.0000, -0.0000]
"""

import torch
import torch.nn.functional as F
import numpy as np

# Import models
from models.wavelet_transform_cnn import WaveletTransformCNN
from models.adamixnet import AdaMixNet
from models.compressor_vnvc import MultiLambdaCompressorVNVC, CompressorVNVC

def investigate_quantizer_issue():
    """Investigate why quantized features are all zeros"""
    
    print("🔍 INVESTIGATING QUANTIZER ISSUE")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test input with varied ranges
    test_inputs = {
        'small': torch.randn(1, 128, 64, 64).to(device) * 0.1,  # Small values
        'normal': torch.randn(1, 128, 64, 64).to(device) * 1.0,  # Normal values  
        'large': torch.randn(1, 128, 64, 64).to(device) * 10.0,  # Large values
    }
    
    # Test individual compressor
    print("\n🔧 TESTING SINGLE COMPRESSOR:")
    compressor_single = CompressorVNVC(input_channels=128, latent_channels=192, lambda_rd=128).to(device)
    compressor_single.eval()
    
    for name, test_input in test_inputs.items():
        print(f"\n📊 TEST {name.upper()}:")
        print(f"Input range: [{test_input.min():.4f}, {test_input.max():.4f}]")
        print(f"Input mean: {test_input.mean():.4f}, std: {test_input.std():.4f}")
        
        with torch.no_grad():
            # Forward pass
            x_hat, likelihoods, y_quantized = compressor_single(test_input)
            
            # Check each component
            print(f"After analysis: [{compressor_single.analysis_transform(test_input).min():.4f}, {compressor_single.analysis_transform(test_input).max():.4f}]")
            
            # Check quantizer specifically
            y_before_quant = compressor_single.analysis_transform(test_input)
            y_after_quant = compressor_single.quantizer(y_before_quant)
            
            print(f"Before quantization: [{y_before_quant.min():.4f}, {y_before_quant.max():.4f}]")
            print(f"After quantization: [{y_after_quant.min():.4f}, {y_after_quant.max():.4f}]")
            print(f"Quantization diff: {(y_after_quant - y_before_quant).abs().mean():.6f}")
            
            print(f"y_quantized unique values: {torch.unique(y_quantized).numel()}")
            print(f"y_quantized non-zero: {(y_quantized != 0).sum()}/{y_quantized.numel()}")
            
            if (y_quantized == 0).all():
                print("❌ All quantized values are zero!")
                # Check if it's a range issue
                if y_before_quant.abs().max() < 0.5:
                    print("   → Input range too small for quantizer")
                else:
                    print("   → Quantizer implementation issue")
    
    # Test with checkpoint loading
    print("\n🔧 TESTING WITH CHECKPOINT:")
    try:
        checkpoint_path = "checkpoints/stage3_ai_heads_coco_best.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        compressor_multi = MultiLambdaCompressorVNVC(input_channels=128, latent_channels=192).to(device)
        if 'compressor_state_dict' in checkpoint:
            compressor_multi.load_state_dict(checkpoint['compressor_state_dict'])
            print("✓ Checkpoint loaded successfully")
        else:
            print("⚠️ No compressor_state_dict in checkpoint")
        
        compressor_multi.eval()
        compressor_multi.set_lambda(128)
        
        # Test with checkpoint
        test_input = torch.randn(1, 128, 64, 64).to(device)
        print(f"\nCheckpoint test input: [{test_input.min():.4f}, {test_input.max():.4f}]")
        
        with torch.no_grad():
            x_hat, likelihoods, y_quantized = compressor_multi(test_input)
            print(f"Checkpoint y_quantized: [{y_quantized.min():.4f}, {y_quantized.max():.4f}]")
            print(f"Checkpoint unique values: {torch.unique(y_quantized).numel()}")
            
            if (y_quantized == 0).all():
                print("❌ Checkpoint also produces all zeros!")
                print("   → Model not properly trained or saved")
            else:
                print("✓ Checkpoint produces non-zero quantized values")
                
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
    
    # Propose fixes
    print(f"\n💡 PROPOSED FIXES:")
    print("1. Check if Stage 2 training actually saved compressor weights")
    print("2. Verify quantizer implementation in RoundWithNoise")  
    print("3. Check if model was trained with proper learning rate")
    print("4. Verify analysis transform produces reasonable outputs")
    print("5. Consider retraining Stage 2 with proper compression")
    
    print(f"\n✅ Investigation completed")

if __name__ == "__main__":
    investigate_quantizer_issue() 