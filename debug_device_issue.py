"""
Debug script để kiểm tra device mismatch chi tiết
"""

import torch
import sys
from pathlib import Path

# Thêm directory gốc vào path
sys.path.append(str(Path(__file__).parent))

from models.wavelet_transform_cnn import WaveletTransformCNN
from models.adamixnet import AdaMixNet  
from models.compressor_improved import ImprovedCompressorVNVC


def check_model_devices(model, device, name="Model"):
    """Kiểm tra tất cả parameters và buffers trong model"""
    print(f"\n=== Checking {name} ===")
    
    # Kiểm tra parameters
    param_devices = set()
    for param_name, param in model.named_parameters():
        param_devices.add(str(param.device))
        # Chỉ báo lỗi nếu device type khác nhau
        if param.device.type != device.type:
            print(f"❌ Parameter {param_name}: {param.device} (expected {device.type})")
    
    # Kiểm tra buffers
    buffer_devices = set()
    for buffer_name, buffer in model.named_buffers():
        if hasattr(buffer, 'device'):
            buffer_devices.add(str(buffer.device))
            # Chỉ báo lỗi nếu device type khác nhau
            if buffer.device.type != device.type:
                print(f"❌ Buffer {buffer_name}: {buffer.device} (expected {device.type})")
    
    print(f"Parameter devices: {param_devices}")
    print(f"Buffer devices: {buffer_devices}")
    
    # Kiểm tra tất cả parameters và buffers có cùng device type không
    all_same_type = True
    
    # Kiểm tra parameters
    if param_devices:
        param_device_types = {str(param.device.type) for param in model.parameters()}
        if len(param_device_types) > 1 or (param_device_types and device.type not in param_device_types):
            all_same_type = False
            print(f"❌ Parameter device types inconsistent: {param_device_types}")
    
    # Kiểm tra buffers
    if buffer_devices:
        buffer_device_types = {str(buffer.device.type) for buffer in model.buffers() if hasattr(buffer, 'device')}
        if len(buffer_device_types) > 1 or (buffer_device_types and device.type not in buffer_device_types):
            all_same_type = False
            print(f"❌ Buffer device types inconsistent: {buffer_device_types}")
    
    if all_same_type:
        print("✅ All parameters and buffers are on the expected device type")
    
    return all_same_type


def test_device_consistency():
    """Test device consistency của toàn bộ pipeline"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing device: {device}")
    
    # Tạo models
    wavelet_cnn = WaveletTransformCNN(
        input_channels=3,
        feature_channels=64,
        wavelet_channels=64
    ).to(device)
    
    adamixnet = AdaMixNet(
        input_channels=256,
        C_prime=64,
        C_mix=128
    ).to(device)
    
    compressor = ImprovedCompressorVNVC(
        input_channels=128,
        latent_channels=192,
        lambda_rd=128
    ).to(device)
    
    # Test input
    test_input = torch.randn(1, 3, 256, 256).to(device)
    print(f"Test input device: {test_input.device}")
    
    # Kiểm tra device của từng model
    models = [
        ('WaveletTransformCNN', wavelet_cnn),
        ('AdaMixNet', adamixnet),
        ('ImprovedCompressorVNVC', compressor)
    ]
    
    all_consistent = True
    for name, model in models:
        if not check_model_devices(model, device, name):
            all_consistent = False
    
    if not all_consistent:
        print("\n❌ Device inconsistency detected!")
        return False
    
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
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False


if __name__ == "__main__":
    success = test_device_consistency()
    if success:
        print("\n🎉 Device consistency test PASSED!")
    else:
        print("\n💥 Device consistency test FAILED!") 