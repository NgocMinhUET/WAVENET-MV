"""
Test Improved vs Old Compressor - Side-by-side comparison
Based on bottleneck analysis: Analysis transform collapses 0.22 → 0.04 → all zeros
"""

import torch
import torch.nn.functional as F

# Handle import paths
try:
    from models.compressor_vnvc import CompressorVNVC
    from models.compressor_improved import ImprovedCompressorVNVC
    from models.wavelet_transform_cnn import WaveletTransformCNN
    from models.adamixnet import AdaMixNet
except ModuleNotFoundError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    from compressor_vnvc import CompressorVNVC
    from compressor_improved import ImprovedCompressorVNVC
    from wavelet_transform_cnn import WaveletTransformCNN
    from adamixnet import AdaMixNet

def compare_compressors():
    """Compare old vs improved compressor performance"""
    
    print("🔬 OLD vs IMPROVED COMPRESSOR COMPARISON")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create both compressors
    old_comp = CompressorVNVC(input_channels=128, latent_channels=192, lambda_rd=128).to(device)
    improved_comp = ImprovedCompressorVNVC(input_channels=128, latent_channels=192, lambda_rd=128).to(device)
    
    old_comp.eval()
    improved_comp.eval()
    
    # Test scenarios matching real pipeline
    test_cases = [
        ("Realistic AdaMixNet Output", torch.randn(1, 128, 64, 64) * 0.2),  # Our problematic case
        ("Small Range", torch.randn(1, 128, 64, 64) * 0.1),
        ("Normal Range", torch.randn(1, 128, 64, 64) * 1.0),
    ]
    
    for name, test_input in test_cases:
        test_input = test_input.to(device)
        
        print(f"\n📊 TEST CASE: {name}")
        print(f"Input range: [{test_input.min():.6f}, {test_input.max():.6f}]")
        print("-" * 40)
        
        with torch.no_grad():
            # OLD COMPRESSOR
            print("🔧 OLD COMPRESSOR:")
            try:
                # Analysis transform
                old_analysis = old_comp.analysis_transform(test_input)
                print(f"  Analysis range: [{old_analysis.min():.6f}, {old_analysis.max():.6f}]")
                
                # Full forward
                old_x_hat, old_likelihoods, old_y_quantized = old_comp(test_input)
                
                print(f"  Quantized range: [{old_y_quantized.min():.6f}, {old_y_quantized.max():.6f}]")
                print(f"  Quantized unique: {torch.unique(old_y_quantized).numel()}")
                print(f"  Non-zero ratio: {(old_y_quantized != 0).float().mean():.4f}")
                
                old_mse = F.mse_loss(old_x_hat, test_input)
                print(f"  Reconstruction MSE: {old_mse.item():.6f}")
                
                # Status
                if (old_y_quantized == 0).all():
                    old_status = "❌ All zeros"
                elif torch.unique(old_y_quantized).numel() < 5:
                    old_status = "⚠️ Limited diversity"
                else:
                    old_status = "✅ Good diversity"
                print(f"  Status: {old_status}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                old_y_quantized = torch.zeros_like(test_input[:, :192, :16, :16])
                old_mse = torch.tensor(float('inf'))
                old_status = "❌ Failed"
        
            # IMPROVED COMPRESSOR
            print("\n🔧 IMPROVED COMPRESSOR:")
            try:
                # Analysis transform
                improved_analysis = improved_comp.analysis_transform(test_input)
                print(f"  Analysis range: [{improved_analysis.min():.6f}, {improved_analysis.max():.6f}]")
                
                # Full forward
                improved_x_hat, improved_likelihoods, improved_y_quantized = improved_comp(test_input)
                
                print(f"  Quantized range: [{improved_y_quantized.min():.6f}, {improved_y_quantized.max():.6f}]")
                print(f"  Quantized unique: {torch.unique(improved_y_quantized).numel()}")
                print(f"  Non-zero ratio: {(improved_y_quantized != 0).float().mean():.4f}")
                
                improved_mse = F.mse_loss(improved_x_hat, test_input)
                print(f"  Reconstruction MSE: {improved_mse.item():.6f}")
                
                # Status
                if (improved_y_quantized == 0).all():
                    improved_status = "❌ All zeros"
                elif torch.unique(improved_y_quantized).numel() < 5:
                    improved_status = "⚠️ Limited diversity"
                else:
                    improved_status = "✅ Good diversity"
                print(f"  Status: {improved_status}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                improved_y_quantized = torch.zeros_like(test_input[:, :192, :16, :16])
                improved_mse = torch.tensor(float('inf'))
                improved_status = "❌ Failed"
        
            # COMPARISON SUMMARY
            print(f"\n📈 COMPARISON SUMMARY:")
            
            # Non-zero ratio improvement
            old_nonzero = (old_y_quantized != 0).float().mean()
            improved_nonzero = (improved_y_quantized != 0).float().mean()
            nonzero_improvement = improved_nonzero - old_nonzero
            
            print(f"  Non-zero ratio: {old_nonzero:.4f} → {improved_nonzero:.4f} ({nonzero_improvement:+.4f})")
            
            # Unique values improvement
            old_unique = torch.unique(old_y_quantized).numel()
            improved_unique = torch.unique(improved_y_quantized).numel()
            unique_improvement = improved_unique - old_unique
            
            print(f"  Unique values: {old_unique} → {improved_unique} ({unique_improvement:+d})")
            
            # MSE comparison
            if old_mse.item() != float('inf') and improved_mse.item() != float('inf'):
                mse_improvement = old_mse.item() - improved_mse.item()
                print(f"  MSE: {old_mse.item():.6f} → {improved_mse.item():.6f} ({mse_improvement:+.6f})")
            
            # Overall assessment
            if improved_nonzero > old_nonzero and improved_unique > old_unique:
                print(f"  ✅ IMPROVED - Better quantization preservation")
            elif improved_nonzero > old_nonzero or improved_unique > old_unique:
                print(f"  ⚠️ PARTIAL IMPROVEMENT")
            else:
                print(f"  ❌ NO IMPROVEMENT")

def test_full_pipeline_comparison():
    """Test trong context của full pipeline"""
    
    print(f"\n🔬 FULL PIPELINE COMPARISON")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Pipeline components already imported above
    
    # Create pipeline
    wavelet_cnn = WaveletTransformCNN(input_channels=3, wavelet_channels=64).to(device)
    adamixnet = AdaMixNet(input_channels=256, C_prime=64, C_mix=128).to(device)
    
    old_comp = CompressorVNVC(input_channels=128, latent_channels=192, lambda_rd=128).to(device)
    improved_comp = ImprovedCompressorVNVC(input_channels=128, latent_channels=192, lambda_rd=128).to(device)
    
    # Set eval mode
    wavelet_cnn.eval()
    adamixnet.eval()
    old_comp.eval()
    improved_comp.eval()
    
    # Test input
    test_input = torch.randn(1, 3, 256, 256).to(device)
    print(f"Original input: [{test_input.min():.6f}, {test_input.max():.6f}]")
    
    with torch.no_grad():
        # Shared pipeline steps
        wavelet_out = wavelet_cnn(test_input)
        mixed_out = adamixnet(wavelet_out)
        print(f"AdaMixNet output: [{mixed_out.min():.6f}, {mixed_out.max():.6f}]")
        
        # OLD COMPRESSOR PATH
        print(f"\n🔧 OLD COMPRESSOR PIPELINE:")
        old_x_hat, old_likelihoods, old_y_quantized = old_comp(mixed_out)
        print(f"  Quantized: [{old_y_quantized.min():.6f}, {old_y_quantized.max():.6f}]")
        print(f"  Non-zero: {(old_y_quantized != 0).float().mean():.4f}")
        
        # IMPROVED COMPRESSOR PATH  
        print(f"\n🔧 IMPROVED COMPRESSOR PIPELINE:")
        improved_x_hat, improved_likelihoods, improved_y_quantized = improved_comp(mixed_out)
        print(f"  Quantized: [{improved_y_quantized.min():.6f}, {improved_y_quantized.max():.6f}]")
        print(f"  Non-zero: {(improved_y_quantized != 0).float().mean():.4f}")
        
        # Final comparison
        print(f"\n📊 PIPELINE COMPARISON:")
        old_nonzero = (old_y_quantized != 0).float().mean()
        improved_nonzero = (improved_y_quantized != 0).float().mean()
        
        if improved_nonzero > 0.1 and old_nonzero < 0.01:
            print("✅ MAJOR IMPROVEMENT - Fixed quantization collapse!")
        elif improved_nonzero > old_nonzero:
            print("⚠️ IMPROVEMENT - Better but still needs work")
        else:
            print("❌ NO IMPROVEMENT - Need different approach")

if __name__ == "__main__":
    compare_compressors()
    test_full_pipeline_comparison()
    
    print(f"\n🎯 CONCLUSION:")
    print("If improved compressor shows >50% non-zero ratio on realistic inputs,")
    print("then the fix is working and we can proceed with full integration!")
    
    print(f"\n✅ Comparison completed") 