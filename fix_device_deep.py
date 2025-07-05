"""
Script để sửa lỗi device mismatch giữa CUDA và CPU trong các module PyTorch
Lỗi: "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same"
"""

import os
import sys
import torch
import torch.nn as nn
import importlib
import argparse

def fix_device_mismatch(model, device=None, verbose=True):
    """
    Đảm bảo tất cả các tham số và buffers của model đều ở cùng một device
    
    Args:
        model: Mô hình PyTorch cần sửa
        device: Device đích (cuda hoặc cpu), nếu None sẽ tự động chọn cuda nếu có
        verbose: In thông báo chi tiết
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"🔧 Đang chuyển model sang device {device}...")
    
    # Chuyển toàn bộ model sang device
    model.to(device)
    
    # Kiểm tra từng module con
    for name, module in model.named_modules():
        # Đảm bảo tất cả parameters đều ở đúng device
        for param_name, param in module.named_parameters(recurse=False):
            if param.device != device:
                if verbose:
                    print(f"⚠️ Parameter {name}.{param_name} đang ở {param.device}, chuyển sang {device}")
                param.data = param.data.to(device)
        
        # Đảm bảo tất cả buffers đều ở đúng device
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer.device != device:
                if verbose:
                    print(f"⚠️ Buffer {name}.{buffer_name} đang ở {buffer.device}, chuyển sang {device}")
                module.register_buffer(buffer_name, buffer.to(device))
    
    # Kiểm tra lại sau khi sửa
    mismatched_params = []
    mismatched_buffers = []
    
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param.device != device:
                mismatched_params.append(f"{name}.{param_name}")
        
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer.device != device:
                mismatched_buffers.append(f"{name}.{buffer_name}")
    
    if len(mismatched_params) == 0 and len(mismatched_buffers) == 0:
        if verbose:
            print(f"✅ Tất cả parameters và buffers đã được chuyển sang {device}")
        return True
    else:
        if verbose:
            if mismatched_params:
                print(f"❌ Các parameters sau vẫn chưa ở {device}: {mismatched_params}")
            if mismatched_buffers:
                print(f"❌ Các buffers sau vẫn chưa ở {device}: {mismatched_buffers}")
        return False

def fix_model_from_checkpoint(checkpoint_path, device=None, save_fixed=True):
    """
    Sửa lỗi device mismatch cho một checkpoint
    
    Args:
        checkpoint_path: Đường dẫn đến file checkpoint
        device: Device đích (cuda hoặc cpu)
        save_fixed: Lưu checkpoint đã sửa
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔍 Đang kiểm tra checkpoint: {checkpoint_path}")
    
    try:
        # Tải checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✅ Đã tải checkpoint thành công")
        
        # Kiểm tra cấu trúc checkpoint
        if isinstance(checkpoint, dict):
            fixed_count = 0
            
            # Xử lý từng state_dict trong checkpoint
            for key in checkpoint.keys():
                if isinstance(checkpoint[key], dict) and any(k.endswith('.weight') or k.endswith('.bias') for k in checkpoint[key].keys()):
                    print(f"🔧 Đang xử lý state_dict: {key}")
                    state_dict = checkpoint[key]
                    
                    # Kiểm tra và sửa device cho từng tensor
                    for param_name, param in state_dict.items():
                        if isinstance(param, torch.Tensor) and param.device != device:
                            state_dict[param_name] = param.to(device)
                            fixed_count += 1
                    
                    print(f"✅ Đã sửa {fixed_count} tensors trong {key}")
            
            if fixed_count > 0 and save_fixed:
                # Tạo tên file mới
                base, ext = os.path.splitext(checkpoint_path)
                fixed_path = f"{base}_fixed{ext}"
                
                # Lưu checkpoint đã sửa
                torch.save(checkpoint, fixed_path)
                print(f"💾 Đã lưu checkpoint đã sửa: {fixed_path}")
            
            return fixed_count
        else:
            print("❌ Checkpoint không có cấu trúc dictionary như mong đợi")
            return 0
    except Exception as e:
        print(f"❌ Lỗi khi xử lý checkpoint: {e}")
        return 0

def fix_model_in_evaluation(model_name, checkpoint_path, device=None):
    """
    Tải và sửa model để sử dụng trong evaluation
    
    Args:
        model_name: Tên của class model (ví dụ: "WaveletTransformCNN")
        checkpoint_path: Đường dẫn đến file checkpoint
        device: Device đích (cuda hoặc cpu)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Tìm module chứa model
        module_found = False
        model_class = None
        
        for module_name in ['models.wavelet_transform_cnn', 'models.adamixnet', 'models.compressor_vnvc', 'models.compressor_improved', 'models.ai_heads']:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, model_name):
                    model_class = getattr(module, model_name)
                    module_found = True
                    print(f"✅ Tìm thấy class {model_name} trong module {module_name}")
                    break
            except ImportError:
                continue
        
        if not module_found:
            print(f"❌ Không tìm thấy class {model_name} trong các module")
            return None
        
        # Tạo instance của model
        model = model_class()
        
        # Tải checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Tìm state_dict phù hợp
        state_dict_key = None
        for key in checkpoint.keys():
            if key.lower().endswith('state_dict') and model_name.lower() in key.lower():
                state_dict_key = key
                break
        
        if state_dict_key is None:
            # Thử tìm state_dict bất kỳ
            for key in checkpoint.keys():
                if key.lower().endswith('state_dict'):
                    state_dict_key = key
                    break
        
        if state_dict_key is not None:
            # Load state_dict
            model.load_state_dict(checkpoint[state_dict_key])
            print(f"✅ Đã tải state_dict từ key: {state_dict_key}")
        else:
            print("❌ Không tìm thấy state_dict phù hợp trong checkpoint")
            return None
        
        # Sửa device mismatch
        fix_device_mismatch(model, device)
        
        return model
    except Exception as e:
        print(f"❌ Lỗi khi tải và sửa model: {e}")
        return None

def fix_all_models_in_checkpoint(checkpoint_path, device=None):
    """
    Sửa tất cả các models trong một checkpoint
    
    Args:
        checkpoint_path: Đường dẫn đến file checkpoint
        device: Device đích (cuda hoặc cpu)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Import các models
        from models.wavelet_transform_cnn import WaveletTransformCNN
        from models.adamixnet import AdaMixNet
        try:
            from models.compressor_improved import ImprovedMultiLambdaCompressorVNVC as MultiLambdaCompressorVNVC
        except ImportError:
            from models.compressor_vnvc import MultiLambdaCompressorVNVC
        
        # Tải checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Đã tải checkpoint: {checkpoint_path}")
        
        # Khởi tạo models
        wavelet_cnn = WaveletTransformCNN(input_channels=3, feature_channels=64, wavelet_channels=64)
        adamixnet = AdaMixNet(input_channels=256, C_prime=64, C_mix=128)
        compressor = MultiLambdaCompressorVNVC(input_channels=128, latent_channels=192)
        
        # Load state dicts
        if 'wavelet_state_dict' in checkpoint:
            wavelet_cnn.load_state_dict(checkpoint['wavelet_state_dict'])
            print("✅ Đã tải wavelet_state_dict")
        
        if 'adamixnet_state_dict' in checkpoint:
            adamixnet.load_state_dict(checkpoint['adamixnet_state_dict'])
            print("✅ Đã tải adamixnet_state_dict")
        
        if 'compressor_state_dict' in checkpoint:
            compressor.load_state_dict(checkpoint['compressor_state_dict'])
            print("✅ Đã tải compressor_state_dict")
        
        # Sửa device mismatch
        print("\n🔧 Đang sửa WaveletTransformCNN...")
        fix_device_mismatch(wavelet_cnn, device)
        
        print("\n🔧 Đang sửa AdaMixNet...")
        fix_device_mismatch(adamixnet, device)
        
        print("\n🔧 Đang sửa MultiLambdaCompressorVNVC...")
        fix_device_mismatch(compressor, device)
        
        # Lưu checkpoint đã sửa
        checkpoint['wavelet_state_dict'] = wavelet_cnn.state_dict()
        checkpoint['adamixnet_state_dict'] = adamixnet.state_dict()
        checkpoint['compressor_state_dict'] = compressor.state_dict()
        
        # Tạo tên file mới
        base, ext = os.path.splitext(checkpoint_path)
        fixed_path = f"{base}_fixed_deep{ext}"
        
        # Lưu checkpoint đã sửa
        torch.save(checkpoint, fixed_path)
        print(f"\n💾 Đã lưu checkpoint đã sửa: {fixed_path}")
        
        return fixed_path
    except Exception as e:
        print(f"❌ Lỗi khi sửa models: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Sửa lỗi device mismatch trong checkpoint PyTorch")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn đến file checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device đích (cuda hoặc cpu)")
    parser.add_argument("--mode", type=str, default="deep", choices=["simple", "deep"], help="Chế độ sửa (simple: chỉ sửa tensors, deep: sửa toàn bộ models)")
    
    args = parser.parse_args()
    
    # Xác định device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 Bắt đầu sửa lỗi device mismatch với device={device}, mode={args.mode}")
    
    if args.mode == "simple":
        # Chế độ đơn giản: chỉ sửa tensors
        fixed_count = fix_model_from_checkpoint(args.checkpoint, device)
        print(f"✅ Đã sửa {fixed_count} tensors")
    else:
        # Chế độ sâu: sửa toàn bộ models
        fixed_path = fix_all_models_in_checkpoint(args.checkpoint, device)
        if fixed_path:
            print(f"✅ Đã sửa toàn bộ models và lưu vào: {fixed_path}")
            print(f"\n🚀 Đề xuất chạy lệnh sau để kiểm tra:")
            print(f"python evaluation/codec_metrics.py --checkpoint {fixed_path} --dataset coco --data_dir datasets/COCO --split val --lambdas 128 --batch_size 1 --max_samples 10 --output_csv results/test_fixed.csv")
        else:
            print("❌ Không thể sửa models")

if __name__ == "__main__":
    main() 