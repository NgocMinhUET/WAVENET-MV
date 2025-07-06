#!/usr/bin/env python3
"""
Debug script để kiểm tra cấu trúc checkpoint Stage 3
"""

import torch
import os
import sys

def debug_checkpoint(checkpoint_path):
    """Debug checkpoint structure"""
    print(f"🔍 Debugging checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint không tồn tại: {checkpoint_path}")
        return
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"✅ Checkpoint loaded successfully")
        print(f"📊 Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
            
            # Check for AI heads state dicts
            if 'yolo_head_state_dict' in checkpoint:
                print(f"✅ Found yolo_head_state_dict with {len(checkpoint['yolo_head_state_dict'])} keys")
                print(f"   First 5 keys: {list(checkpoint['yolo_head_state_dict'].keys())[:5]}")
            else:
                print(f"❌ yolo_head_state_dict not found")
            
            if 'segformer_head_state_dict' in checkpoint:
                print(f"✅ Found segformer_head_state_dict with {len(checkpoint['segformer_head_state_dict'])} keys")
                print(f"   First 5 keys: {list(checkpoint['segformer_head_state_dict'].keys())[:5]}")
            else:
                print(f"❌ segformer_head_state_dict not found")
            
            # Check for other common keys
            common_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'loss', 'best_loss']
            for key in common_keys:
                if key in checkpoint:
                    print(f"✅ Found {key}")
                else:
                    print(f"❌ {key} not found")
            
            # Show all keys for debugging
            print(f"\n📋 All checkpoint keys:")
            for i, key in enumerate(checkpoint.keys()):
                value_type = type(checkpoint[key]).__name__
                if isinstance(checkpoint[key], dict):
                    value_info = f"dict with {len(checkpoint[key])} keys"
                elif isinstance(checkpoint[key], torch.Tensor):
                    value_info = f"tensor shape {checkpoint[key].shape}"
                else:
                    value_info = str(checkpoint[key])[:50] + "..." if len(str(checkpoint[key])) > 50 else str(checkpoint[key])
                print(f"  {i+1:2d}. {key}: {value_type} - {value_info}")
        
        else:
            print(f"⚠️ Checkpoint is not a dict, it's {type(checkpoint)}")
            print(f"   Content: {checkpoint}")
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    checkpoint_path = "checkpoints/stage3_ai_heads_coco_best.pth"
    
    print("🚀 Debugging Stage 3 Checkpoint Structure")
    print("=" * 50)
    
    debug_checkpoint(checkpoint_path)
    
    print("\n" + "=" * 50)
    print("💡 Suggestions:")
    print("1. Nếu checkpoint không có yolo_head_state_dict/segformer_head_state_dict")
    print("   → Cần train lại Stage 3 với script đúng")
    print("2. Nếu checkpoint có format khác")
    print("   → Cần sửa code để load đúng format")
    print("3. Nếu checkpoint bị corrupt")
    print("   → Cần train lại từ đầu")

if __name__ == "__main__":
    main() 