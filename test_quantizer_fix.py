"""
Test Quantizer Fix - Verify improved quantizer works with small ranges
"""

import torch
from models.compressor_vnvc import CompressorVNVC, QuantizerVNVC

def test_quantizer_fix():
    """Test improved quantizer với different input ranges"""
    
    print("🧪 TESTING QUANTIZER FIX")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test individual quantizer first
    print("\n🔧 TESTING QUANTIZER MODULE:")
    quantizer = QuantizerVNVC(scale_factor=4.0).to(device)
    
    test_ranges = {
        'tiny': 0.01,    # Very small
        'small': 0.1,    # Small  
        'normal': 1.0,   # Normal
        'large': 10.0    # Large
    }
    
    for name, scale in test_ranges.items():
        test_input = torch.randn(1, 192, 8, 8).to(device) * scale
        print(f"\n📊 TEST {name.upper()} (scale={scale}):")
        print(f"Input range: [{test_input.min():.4f}, {test_input.max():.4f}]")
        
        with torch.no_grad():
            quantized = quantizer(test_input)
            
            print(f"Output range: [{quantized.min():.4f}, {quantized.max():.4f}]")
            print(f"Unique values: {torch.unique(quantized).numel()}")
            print(f"Non-zero ratio: {(quantized != 0).float().mean():.4f}")
            
            if (quantized == 0).all():
                print("❌ Still all zeros!")
            elif torch.unique(quantized).numel() > 10:
                print("✅ Good diversity!")
            else:
                print("⚠️ Limited diversity")
    
    # Test full compressor
    print(f"\n🔧 TESTING FULL COMPRESSOR:")
    compressor = CompressorVNVC(input_channels=128, latent_channels=192, lambda_rd=128).to(device)
    compressor.eval()
    
    # Test với realistic AdaMixNet output ranges
    realistic_input = torch.randn(1, 128, 64, 64).to(device) * 0.1  # Typical AdaMixNet output
    print(f"\nRealistic input range: [{realistic_input.min():.4f}, {realistic_input.max():.4f}]")
    
    with torch.no_grad():
        x_hat, likelihoods, y_quantized = compressor(realistic_input)
        
        print(f"Quantized range: [{y_quantized.min():.4f}, {y_quantized.max():.4f}]")
        print(f"Quantized unique: {torch.unique(y_quantized).numel()}")
        print(f"Quantized non-zero: {(y_quantized != 0).sum()}/{y_quantized.numel()}")
        
        # Check reconstruction quality
        mse = torch.nn.functional.mse_loss(x_hat, realistic_input)
        print(f"Reconstruction MSE: {mse.item():.6f}")
        
        if (y_quantized == 0).all():
            print("❌ Fix FAILED - still all zeros")
            return False
        elif torch.unique(y_quantized).numel() < 5:
            print("⚠️ Fix PARTIAL - limited quantization")
            return False
        else:
            print("✅ Fix SUCCESS - proper quantization!")
            return True

def compare_old_vs_new():
    """Compare old vs new quantizer behavior"""
    
    print(f"\n🔬 COMPARING OLD VS NEW QUANTIZER:")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Old quantizer (no scaling)
    class OldQuantizer(torch.nn.Module):
        def forward(self, x):
            return torch.round(x + torch.empty_like(x).uniform_(-0.5, 0.5))
    
    old_quantizer = OldQuantizer().to(device)
    new_quantizer = QuantizerVNVC(scale_factor=4.0).to(device)
    
    # Test with small input (typical problem case)
    test_input = torch.randn(1, 192, 8, 8).to(device) * 0.1
    print(f"Test input range: [{test_input.min():.4f}, {test_input.max():.4f}]")
    
    with torch.no_grad():
        old_output = old_quantizer(test_input)
        new_output = new_quantizer(test_input)
        
        print(f"\n📊 OLD QUANTIZER:")
        print(f"  Output range: [{old_output.min():.4f}, {old_output.max():.4f}]")
        print(f"  Unique values: {torch.unique(old_output).numel()}")
        print(f"  Non-zero ratio: {(old_output != 0).float().mean():.4f}")
        
        print(f"\n📊 NEW QUANTIZER:")
        print(f"  Output range: [{new_output.min():.4f}, {new_output.max():.4f}]")
        print(f"  Unique values: {torch.unique(new_output).numel()}")
        print(f"  Non-zero ratio: {(new_output != 0).float().mean():.4f}")
        
        improvement = (new_output != 0).float().mean() - (old_output != 0).float().mean()
        print(f"\n📈 IMPROVEMENT: {improvement:.4f} (+{improvement*100:.1f}% non-zero values)")

if __name__ == "__main__":
    success = test_quantizer_fix()
    compare_old_vs_new()
    
    if success:
        print(f"\n🎉 QUANTIZER FIX VERIFIED!")
        print("Ready to test full pipeline!")
    else:
        print(f"\n❌ QUANTIZER FIX NEEDS MORE WORK!")
    
    print(f"\n✅ Testing completed") 