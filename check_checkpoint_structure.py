"""
Script để kiểm tra cấu trúc checkpoint Stage 3
"""

import torch
import sys
from pathlib import Path

def check_checkpoint_structure(checkpoint_path):
    """Kiểm tra cấu trúc của checkpoint"""
    
    print(f"🔍 Kiểm tra checkpoint: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"\n📋 Cấu trúc checkpoint:")
        print(f"Type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Keys: {list(checkpoint.keys())}")
            
            for key, value in checkpoint.items():
                if isinstance(value, dict):
                    print(f"  {key}: dict with {len(value)} items")
                    if key in ['wavelet_state_dict', 'adamixnet_state_dict', 'compressor_state_dict']:
                        print(f"    - Contains model parameters")
                elif hasattr(value, 'shape'):
                    print(f"  {key}: tensor with shape {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
        
        # Kiểm tra xem có phải là checkpoint Stage 3 không
        if 'wavelet_state_dict' in checkpoint and 'adamixnet_state_dict' in checkpoint and 'compressor_state_dict' in checkpoint:
            print(f"\n✅ Đây là checkpoint đầy đủ (Stage 2 hoặc Stage 3)")
            print(f"   - WaveletTransformCNN: ✓")
            print(f"   - AdaMixNet: ✓") 
            print(f"   - CompressorVNVC: ✓")
            
            # Kiểm tra có AI heads không
            ai_heads_keys = [k for k in checkpoint.keys() if 'ai' in k.lower() or 'head' in k.lower() or 'yolo' in k.lower() or 'segformer' in k.lower()]
            if ai_heads_keys:
                print(f"   - AI Heads: ✓ ({ai_heads_keys})")
                print(f"✅ Đây là checkpoint Stage 3 (AI Heads)")
            else:
                print(f"   - AI Heads: ✗")
                print(f"✅ Đây là checkpoint Stage 2 (Compressor)")
                
        else:
            print(f"\n❓ Cấu trúc checkpoint không rõ ràng")
            print(f"   Có thể là checkpoint của một model riêng lẻ")
            
    except Exception as e:
        print(f"❌ Lỗi khi load checkpoint: {e}")


if __name__ == "__main__":
    checkpoint_path = "checkpoints/stage3_ai_heads_coco_best.pth"
    check_checkpoint_structure(checkpoint_path) 