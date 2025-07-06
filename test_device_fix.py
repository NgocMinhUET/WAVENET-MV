"""
Test script để kiểm tra việc sửa device mismatch
"""

import torch
import sys
from pathlib import Path

# Thêm directory gốc vào path
sys.path.append(str(Path(__file__).parent))

from models.wavelet_transform_cnn import WaveletTransformCNN
from models.adamixnet import AdaMixNet  
from models.compressor_vnvc import MultiLambdaCompressorVNVC


def test_device_fix():
    """Test việc sửa device mismatch"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing device: {device}")
    
    # Tạo models
    wavelet_cnn = WaveletTransformCNN(
        input_channels=3,
        feature_channels=64,
        wavelet_channels=64
    )
    
    adamixnet = AdaMixNet(
        input_channels=256,
        C_prime=64,
        C_mix=128
    )
    
    compressor = MultiLambdaCompressorVNVC(
        input_channels=128,
        latent_channels=192
    )
    
    # Test input
    test_input = torch.randn(1, 3, 256, 256).to(device)
    print(f"Test input device: {test_input.device}")
    
    # Di chuyển models đến device
    print("\n=== Testing WaveletTransformCNN ===")
    wavelet_cnn = wavelet_cnn.to(device)
    
    print("\n=== Testing AdaMixNet ===")
    adamixnet = adamixnet.to(device)
    
    print("\n=== Testing Compressor ===")
    compressor = compressor.to(device)
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    try:
        with torch.no_grad():
            # Wavelet
            wavelet_out = wavelet_cnn(test_input)
            print(f"Wavelet output: {wavelet_out.shape}, device: {wavelet_out.device}")
            
            # AdaMixNet
            mixed_out = adamixnet(wavelet_out)
            print(f"AdaMixNet output: {mixed_out.shape}, device: {mixed_out.device}")
            
            # Compressor
            x_hat, likelihoods, y_quantized = compressor(mixed_out)
            print(f"Compressor output: {x_hat.shape}, device: {x_hat.device}")
            
        print("✅ Forward pass successful!")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = test_device_fix()
    if success:
        print("\n🎉 Device fix test PASSED!")
    else:
        print("\n💥 Device fix test FAILED!") 