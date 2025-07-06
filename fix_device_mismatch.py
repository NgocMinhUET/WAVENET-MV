"""
WAVENET-MV Device Mismatch Fixer
Sửa lỗi "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same"

Cách dùng: python fix_device_mismatch.py --checkpoint path/to/checkpoint.pt --output path/to/fixed_checkpoint.pt
"""

import argparse
import torch
import os
import sys
from pathlib import Path

# Thêm directory gốc vào path
sys.path.append(str(Path(__file__).parent))

# Import models
from models.wavelet_transform_cnn import WaveletTransformCNN
from models.adamixnet import AdaMixNet  
from models.compressor_vnvc import MultiLambdaCompressorVNVC


def fix_device_mismatch(model, target_device):
    """
    Đảm bảo tất cả tham số và buffers đều nằm trên cùng một device
    
    Args:
        model: PyTorch model
        target_device: device cần chuyển đến (thường là 'cuda')
    """
    print(f"Đang di chuyển {model.__class__.__name__} đến {target_device}...")
    
    # Đảm bảo model ở đúng device
    model = model.to(target_device)
    
    # Kiểm tra từng module con
    for name, module in model.named_modules():
        # Kiểm tra parameters
        for param_name, param in module.named_parameters(recurse=False):
            if param.device != target_device:
                print(f"  - Di chuyển parameter {name}.{param_name} từ {param.device} đến {target_device}")
                param.data = param.data.to(target_device)
        
        # Kiểm tra buffers
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer.device != target_device:
                print(f"  - Di chuyển buffer {name}.{buffer_name} từ {buffer.device} đến {target_device}")
                module.register_buffer(buffer_name, buffer.to(target_device))
    
    return model


def check_model_devices(model, name="Model"):
    """
    Kiểm tra tất cả parameters và buffers trong model có cùng device không
    
    Args:
        model: PyTorch model
        name: Tên của model để hiển thị
        
    Returns:
        bool: True nếu tất cả cùng device, False nếu có sự khác biệt
    """
    devices = set()
    
    # Kiểm tra parameters
    for name, param in model.named_parameters():
        devices.add(str(param.device))
    
    # Kiểm tra buffers
    for name, buffer in model.named_buffers():
        devices.add(str(buffer.device))
    
    print(f"{name} đang sử dụng các devices: {devices}")
    
    return len(devices) == 1


def fix_checkpoint(checkpoint_path, output_path, device_str="cuda"):
    """
    Sửa checkpoint để đảm bảo tất cả parameters và buffers đều ở cùng device
    
    Args:
        checkpoint_path: Đường dẫn đến checkpoint gốc
        output_path: Đường dẫn lưu checkpoint sau khi sửa
        device_str: Tên device muốn chuyển đến ("cuda" hoặc "cpu")
    """
    print(f"Đang sửa checkpoint: {checkpoint_path}")
    print(f"Device đích: {device_str}")
    
    # Xác định device
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"Sử dụng device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("Đã load checkpoint")
    
    # Khởi tạo models
    wavelet_cnn = WaveletTransformCNN(
        input_channels=3,
        feature_channels=64,
        wavelet_channels=64
    )
    
    adamixnet = AdaMixNet(
        input_channels=256,  # 4 * 64
        C_prime=64,
        C_mix=128
    )
    
    compressor = MultiLambdaCompressorVNVC(
        input_channels=128,
        latent_channels=192
    )
    
    # Load state dicts
    print("Đang load state dictionaries...")
    if 'wavelet_state_dict' in checkpoint:
        wavelet_cnn.load_state_dict(checkpoint['wavelet_state_dict'])
        print("✓ Đã load wavelet_state_dict")
    
    if 'adamixnet_state_dict' in checkpoint:
        adamixnet.load_state_dict(checkpoint['adamixnet_state_dict'])
        print("✓ Đã load adamixnet_state_dict")
    
    if 'compressor_state_dict' in checkpoint:
        compressor.load_state_dict(checkpoint['compressor_state_dict'])
        print("✓ Đã load compressor_state_dict")
    
    # Sửa device mismatch
    print("\nĐang sửa device mismatch...")
    wavelet_cnn = fix_device_mismatch(wavelet_cnn, device)
    adamixnet = fix_device_mismatch(adamixnet, device)
    compressor = fix_device_mismatch(compressor, device)
    
    # Kiểm tra kết quả
    print("\nKiểm tra sau khi sửa:")
    check_model_devices(wavelet_cnn, "WaveletTransformCNN")
    check_model_devices(adamixnet, "AdaMixNet")
    check_model_devices(compressor, "CompressorVNVC")
    
    # Cập nhật state dicts trong checkpoint
    checkpoint['wavelet_state_dict'] = wavelet_cnn.state_dict()
    checkpoint['adamixnet_state_dict'] = adamixnet.state_dict()
    checkpoint['compressor_state_dict'] = compressor.state_dict()
    
    # Lưu checkpoint đã sửa
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"\n✓ Đã lưu checkpoint đã sửa tại: {output_path}")
    print(f"Dùng checkpoint này để chạy evaluation sẽ không còn lỗi device mismatch")


def main():
    parser = argparse.ArgumentParser(description="Fix device mismatch issues in WAVENET-MV checkpoints")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Đường dẫn đến checkpoint gốc")
    parser.add_argument("--output", type=str, required=True,
                       help="Đường dẫn lưu checkpoint đã sửa")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="Device muốn sử dụng (cuda hoặc cpu)")
    
    args = parser.parse_args()
    
    fix_checkpoint(args.checkpoint, args.output, args.device)


if __name__ == "__main__":
    main() 