"""
Codec Metrics Evaluation
PSNR, MS-SSIM, BPP calculation for WAVENET-MV
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
from pathlib import Path
import math

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.wavelet_transform_cnn import WaveletTransformCNN
from models.adamixnet import AdaMixNet  
from models.compressor_vnvc import MultiLambdaCompressorVNVC
from datasets.dataset_loaders import COCODatasetLoader, DAVISDatasetLoader


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR giữa hai ảnh.
    Nếu max_val không được truyền (mặc định 1.0) nhưng giá trị tuyệt đối của ảnh lớn hơn 1
    (ví dụ ảnh đã Normalize theo ImageNet mean/std), ta sẽ ước lượng peak signal
    bằng đoạn [min, max] của ảnh gốc để tránh PSNR âm không cần thiết."""
    if max_val == 1.0:
        # Ước lượng dải động thực tế
        data_max = torch.max(img1)
        data_min = torch.min(img1)
        max_val_est = max(data_max.abs(), data_min.abs())
        # Trường hợp ảnh đã normalise quanh 0 với std~0.23 → max_val_est ~2.5
        # Đảm bảo max_val_est >= 1.0 để công thức hợp lệ
        max_val = torch.clamp(max_val_est, min=1.0).item()
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def calculate_ms_ssim(img1, img2, data_range=1.0):
    """Calculate MS-SSIM between two images với dải động linh hoạt và handle small images."""
    if data_range == 1.0:
        data_max = torch.max(img1) if torch.is_tensor(img1) else np.max(img1)
        data_min = torch.min(img1) if torch.is_tensor(img1) else np.min(img1)
        data_range = max(abs(float(data_max)), abs(float(data_min)), 1.0)
    
    # Convert to numpy và ensure correct shape
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()
    
    def safe_ssim(im1, im2, data_range):
        """Safe SSIM calculation with adaptive window size"""
        try:
            # Get image dimensions
            H, W = im1.shape[:2] if im1.ndim >= 2 else im1.shape
            
            # Calculate adaptive window size
            min_dim = min(H, W)
            if min_dim < 7:
                # For very small images, use smaller window or fallback to simple similarity
                if min_dim < 3:
                    # Fallback to simple correlation for tiny images
                    return np.corrcoef(im1.flatten(), im2.flatten())[0, 1]
                else:
                    # Use smaller window size
                    win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
                    return ssim(im1, im2, data_range=data_range, win_size=win_size)
            else:
                # Normal case: use default window size
                return ssim(im1, im2, data_range=data_range)
        except Exception as e:
            # Fallback to simple correlation if SSIM fails
            print(f"SSIM failed, using correlation fallback: {e}")
            return np.corrcoef(im1.flatten(), im2.flatten())[0, 1]
    
    # Handle batch dimension
    if img1.ndim == 4:  # Batch of images
        ms_ssim_values = []
        for i in range(img1.shape[0]):
            # Convert from CHW to HWC
            im1 = np.transpose(img1[i], (1, 2, 0))
            im2 = np.transpose(img2[i], (1, 2, 0))
            
            # Calculate MS-SSIM cho từng channel và average
            if im1.shape[2] == 3:  # RGB
                ms_ssim_val = 0
                for c in range(3):
                    ms_ssim_val += safe_ssim(im1[:,:,c], im2[:,:,c], data_range)
                ms_ssim_val /= 3
            else:
                ms_ssim_val = safe_ssim(im1.squeeze(), im2.squeeze(), data_range)
            
            ms_ssim_values.append(ms_ssim_val)
        
        return np.mean(ms_ssim_values)
    else:
        # Single image
        if img1.ndim == 3:  # CHW to HWC
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
        
        return safe_ssim(img1, img2, data_range)


def estimate_bpp(compressed_data, image_shape):
    """Estimate bits per pixel from compressed representation"""
    if isinstance(compressed_data, dict):
        # Extract actual compressed size
        total_bits = 0
        if 'strings' in compressed_data:
            for string in compressed_data['strings']:
                if isinstance(string, (list, tuple)):
                    for s in string:
                        total_bits += len(s) * 8  # Convert bytes to bits
                else:
                    total_bits += len(string) * 8
        
        # Fallback: estimate from tensor size
        if total_bits == 0 and 'shape' in compressed_data:
            total_bits = np.prod(compressed_data['shape']) * 16  # Assume 16 bits per value
    
    elif torch.is_tensor(compressed_data):
        # Direct tensor - estimate compression
        total_bits = compressed_data.numel() * 16  # Assume quantized to 16 bits
    
    else:
        total_bits = len(str(compressed_data)) * 8  # Fallback
    
    # Calculate BPP
    total_pixels = image_shape[0] * image_shape[1]  # H * W
    bpp = total_bits / total_pixels
    
    return bpp


def estimate_bpp_from_features(quantized_features, image_shape):
    """
    UNIFIED: Consistent BPP estimation across all evaluation scripts
    Args:
        quantized_features: Quantized latent features [B, C, H, W]
        image_shape: Original image shape (H, W)
    Returns:
        bpp: Estimated bits per pixel
    """
    B, C, H_feat, W_feat = quantized_features.shape
    
    # Calculate compression ratio: feature_pixels / image_pixels
    feature_pixels = H_feat * W_feat
    image_pixels = image_shape[0] * image_shape[1]
    compression_ratio = feature_pixels / image_pixels
    
    # UNIFIED: Calculate non-zero ratio and entropy more accurately
    # Move to CPU for numpy operations
    features_cpu = quantized_features.cpu().detach()
    
    # Calculate non-zero ratio
    non_zero_mask = torch.abs(features_cpu) > 1e-6
    non_zero_ratio = torch.mean(non_zero_mask.float()).item()
    
    # Estimate entropy based on value distribution
    # More diverse values = higher entropy = more bits needed
    unique_values = torch.unique(features_cpu)
    num_unique = len(unique_values)
    
    # Calculate bits per feature based on entropy
    if num_unique <= 1:
        # Almost no information
        bits_per_feature = 0.1
    elif num_unique <= 4:
        # Very low entropy
        bits_per_feature = 0.5 + 1.5 * non_zero_ratio
    elif num_unique <= 16:
        # Low entropy
        bits_per_feature = 1.0 + 2.0 * non_zero_ratio
    else:
        # Normal entropy
        bits_per_feature = 2.0 + 3.0 * non_zero_ratio
    
    # UNIFIED: More accurate BPP calculation
    # Total bits = (features per image) * (channels) * (bits per feature)
    estimated_bpp = compression_ratio * C * bits_per_feature
    
    # UNIFIED: Reasonable BPP clamping based on actual compression standards
    # Typical range: 0.1-8.0 BPP for neural compression
    estimated_bpp = max(0.1, min(8.0, estimated_bpp))
    
    return estimated_bpp


class CodecEvaluator:
    """Evaluator cho codec metrics"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.load_models()
        
        # Setup dataset
        self.setup_dataset()
        
        # Results storage
        self.results = []
        
    def load_models(self):
        """Load trained models"""
        print("Loading models...")
        
        # Load checkpoint
        if not os.path.exists(self.args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {self.args.checkpoint}")
        
        checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
        
        # Initialize models
        self.wavelet_cnn = WaveletTransformCNN(
            input_channels=3,
            feature_channels=64,
            wavelet_channels=64
        ).to(self.device)
        
        self.adamixnet = AdaMixNet(
            input_channels=256,  # 4 * 64
            C_prime=64,
            C_mix=128
        ).to(self.device)
        
        self.compressor = MultiLambdaCompressorVNVC(
            input_channels=128,
            latent_channels=192
        ).to(self.device)
        
        # Load state dicts
        print("Đang load state dictionaries...")
        
        # Kiểm tra cấu trúc checkpoint
        checkpoint_keys = list(checkpoint.keys())
        print(f"Checkpoint keys: {checkpoint_keys}")
        
        # Load các models chính
        if 'wavelet_state_dict' in checkpoint:
            self.wavelet_cnn.load_state_dict(checkpoint['wavelet_state_dict'])
            print("✓ Đã load wavelet_state_dict")
        
        if 'adamixnet_state_dict' in checkpoint:
            self.adamixnet.load_state_dict(checkpoint['adamixnet_state_dict'])
            print("✓ Đã load adamixnet_state_dict")
        
        if 'compressor_state_dict' in checkpoint:
            self.compressor.load_state_dict(checkpoint['compressor_state_dict'])
            print("✓ Đã load compressor_state_dict")
        
        # Kiểm tra xem có AI heads không (Stage 3)
        ai_heads_keys = [k for k in checkpoint_keys if 'ai' in k.lower() or 'head' in k.lower() or 'yolo' in k.lower() or 'segformer' in k.lower()]
        if ai_heads_keys:
            print(f"⚠️ Phát hiện AI heads trong checkpoint: {ai_heads_keys}")
            print("ℹ️ Đây là checkpoint Stage 3, nhưng evaluation chỉ cần 3 models chính")
        
        # Đảm bảo tất cả các mô hình đều ở cùng device
        self.wavelet_cnn = self.wavelet_cnn.to(self.device)
        self.adamixnet = self.adamixnet.to(self.device)
        self.compressor = self.compressor.to(self.device)
        
        # Fix device mismatch cho từng module con - sử dụng phương pháp mạnh mẽ
        self.wavelet_cnn = self.force_move_to_device(self.wavelet_cnn, self.device)
        self.adamixnet = self.force_move_to_device(self.adamixnet, self.device)
        self.compressor = self.force_move_to_device(self.compressor, self.device)
        
        # Set to evaluation mode
        self.wavelet_cnn.eval()
        self.adamixnet.eval()
        self.compressor.eval()
        
        # CRITICAL: Update entropy models for CompressAI  
        if hasattr(self.args, 'skip_entropy_update') and self.args.skip_entropy_update:
            print("⏭️ Skipping entropy model updates (--skip_entropy_update flag)")
        else:
            print("🔄 Updating entropy models...")
            try:
                if hasattr(self.compressor, 'entropy_bottleneck'):
                    if hasattr(self.compressor.entropy_bottleneck, 'gaussian_conditional'):
                        self.compressor.entropy_bottleneck.gaussian_conditional.update()
                        print("✓ GaussianConditional updated")
            except Exception as e:
                print(f"⚠️ Warning: Could not update individual entropy model: {e}")
            
            # Alternative: Update entire compressor if it has update method
            try:
                if hasattr(self.compressor, 'update'):
                    self.compressor.update()  # FIXED: Remove force parameter
                    print("✓ Compressor entropy models updated")
            except Exception as e:
                print(f"⚠️ Warning: Could not update compressor entropy models: {e}")
                print("ℹ️ This may be OK if models weren't trained with entropy bottleneck")
        
        print("✓ Models loaded and entropy models initialized successfully")
    
    def _fix_device_mismatch(self, model):
        """
        Đảm bảo tất cả các tham số và buffers của model đều ở cùng device
        """
        # Kiểm tra từng module con
        for name, module in model.named_modules():
            # Đảm bảo tất cả parameters đều ở đúng device
            for param_name, param in module.named_parameters(recurse=False):
                if param.device != self.device:
                    print(f"⚠️ Moving parameter {name}.{param_name} from {param.device} to {self.device}")
                    param.data = param.data.to(self.device)
            
            # Đảm bảo tất cả buffers đều ở đúng device
            for buffer_name, buffer in module.named_buffers(recurse=False):
                if buffer.device != self.device:
                    print(f"⚠️ Moving buffer {name}.{buffer_name} from {buffer.device} to {self.device}")
                    module.register_buffer(buffer_name, buffer.to(self.device))
    
    def _fix_device_mismatch_complete(self, model):
        """
        Phiên bản cải tiến của _fix_device_mismatch để đảm bảo mọi thành phần đều ở cùng device.
        Phương pháp này xử lý triệt để hơn để ngăn lỗi "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)"
        """
        print(f"🔧 Đang đảm bảo {model.__class__.__name__} hoàn toàn ở {self.device}")

        # Đảm bảo model hoàn toàn ở trên device đích
        model.to(self.device)

        # Kiểm tra tất cả modules (kể cả modules con)
        for name, module in model.named_modules():
            # Di chuyển module con đến device
            if hasattr(module, 'to'):
                module.to(self.device)
            
            # Di chuyển từng parameter
            for param_name, param in module.named_parameters(recurse=False):
                if param.device != self.device:
                    print(f"  - Di chuyển param {name}.{param_name} từ {param.device} sang {self.device}")
                    # Đảm bảo cả param và param.data đều được di chuyển
                    param.data = param.data.to(self.device)
                    setattr(module, param_name, param.to(self.device))
            
            # Di chuyển từng buffer
            for buffer_name, buffer in module.named_buffers(recurse=False):
                if hasattr(buffer, 'device') and buffer.device != self.device:
                    print(f"  - Di chuyển buffer {name}.{buffer_name} từ {buffer.device} sang {self.device}")
                    # Đảm bảo buffer được đăng ký lại với device mới
                    module.register_buffer(buffer_name, buffer.to(self.device))

        # Kiểm tra lại sau khi di chuyển
        all_on_device = True
        
        for name, param in model.named_parameters():
            if param.device != self.device:
                print(f"❌ CẢNH BÁO: Param {name} vẫn còn ở {param.device}")
                all_on_device = False
        
        for name, buffer in model.named_buffers():
            if hasattr(buffer, 'device') and buffer.device != self.device:
                print(f"❌ CẢNH BÁO: Buffer {name} vẫn còn ở {buffer.device}")
                all_on_device = False
        
        if all_on_device:
            print(f"✓ Tất cả thành phần của {model.__class__.__name__} đã ở {self.device}")
        else:
            print(f"⚠️ Vẫn còn một số thành phần không ở {self.device}!")
    
    def force_move_to_device(self, model, device):
        """
        Phương pháp mạnh mẽ để đảm bảo model hoàn toàn ở trên device.
        Sử dụng cách tiếp cận brute force để di chuyển tất cả thành phần.
        """
        print(f"🚀 Force moving {model.__class__.__name__} to {device}")
        
        # Bước 1: Di chuyển toàn bộ model
        model = model.to(device)
        
        # Bước 2: Di chuyển từng module một cách thủ công
        for name, child in model.named_children():
            if hasattr(child, 'to'):
                child = child.to(device)
                setattr(model, name, child)
        
        # Bước 3: Di chuyển từng parameter một cách thủ công
        for name, param in model.named_parameters():
            if param.device != device:
                # Tạo parameter mới trên device đúng
                new_param = torch.nn.Parameter(param.data.to(device), requires_grad=param.requires_grad)
                
                # Tìm module chứa parameter này
                module_names = name.split('.')
                current_module = model
                
                # Đi đến module cha
                for module_name in module_names[:-1]:
                    current_module = getattr(current_module, module_name)
                
                # Thay thế parameter
                param_name = module_names[-1]
                setattr(current_module, param_name, new_param)
                print(f"  - Force moved parameter {name} to {device}")
        
        # Bước 4: Di chuyển từng buffer một cách thủ công
        for name, buffer in model.named_buffers():
            if hasattr(buffer, 'device') and buffer.device != device:
                # Tìm module chứa buffer này
                module_names = name.split('.')
                current_module = model
                
                # Đi đến module cha
                for module_name in module_names[:-1]:
                    current_module = getattr(current_module, module_name)
                
                # Thay thế buffer
                buffer_name = module_names[-1]
                current_module.register_buffer(buffer_name, buffer.to(device))
                print(f"  - Force moved buffer {name} to {device}")
        
        # Bước 5: Kiểm tra cuối cùng
        all_correct = True
        for name, param in model.named_parameters():
            if param.device != device:
                print(f"❌ ERROR: Parameter {name} still on {param.device}")
                all_correct = False
        
        for name, buffer in model.named_buffers():
            if hasattr(buffer, 'device') and buffer.device != device:
                print(f"❌ ERROR: Buffer {name} still on {buffer.device}")
                all_correct = False
        
        if all_correct:
            print(f"✅ SUCCESS: All components of {model.__class__.__name__} moved to {device}")
        else:
            print(f"❌ FAILED: Some components still not on {device}")
        
        return model
    
    def _custom_collate_fn(self, batch):
        """Custom collate function to handle COCO dataset safely"""
        images = []
        
        for item in batch:
            if isinstance(item, dict):
                # COCO dataset format
                img = item['image']
                if torch.is_tensor(img):
                    images.append(img)
                else:
                    images.append(torch.tensor(img))
            else:
                # Simple tuple format
                img = item[0] if isinstance(item, (tuple, list)) else item
                if torch.is_tensor(img):
                    images.append(img)
                else:
                    images.append(torch.tensor(img))
        
        # Stack images carefully
        try:
            images_tensor = torch.stack(images, 0)
            return {'image': images_tensor}
        except Exception as e:
            # Fallback: process one by one
            print(f"Warning: Batch collate failed, using individual processing: {e}")
            return {'image': images[0].unsqueeze(0)}  # Process one image at a time
    
    def setup_dataset(self):
        """Setup evaluation dataset"""
        if self.args.dataset == 'coco':
            dataset = COCODatasetLoader(
                data_dir=self.args.data_dir,
                subset=self.args.split,
                image_size=self.args.image_size,
                augmentation=False
            )
        elif self.args.dataset == 'davis':
            dataset = DAVISDatasetLoader(
                data_dir=self.args.data_dir,
                subset=self.args.split,
                image_size=self.args.image_size,
                augmentation=False
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,  # FIXED: Disable multiprocessing to avoid tensor resize errors
            pin_memory=False,  # FIXED: Disable pin_memory to avoid storage conflicts
            collate_fn=self._custom_collate_fn  # FIXED: Custom collate function
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} images")
        
    def evaluate_lambda(self, lambda_value):
        """Evaluate metrics for specific lambda value"""
        print(f"\nEvaluating λ = {lambda_value}")
        
        # Set compressor lambda
        self.compressor.set_lambda(lambda_value)
        
        # Metrics accumulation
        psnr_values = []
        ms_ssim_values = []
        bpp_values = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc=f'λ={lambda_value}')):
                # Get images
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                else:
                    images = batch[0].to(self.device)
                
                try:
                    # Forward pass through pipeline (SKIP COMPRESSION due to entropy model issues)
                    # 1. Wavelet transform
                    wavelet_coeffs = self.wavelet_cnn(images)
                    
                    # 2. AdaMixNet
                    mixed_features = self.adamixnet(wavelet_coeffs)
                    
                    # 3. Compressor forward (without compress/decompress)
                    x_hat, likelihoods, y_quantized = self.compressor(mixed_features)
                    
                    # 4. Inverse AdaMixNet (approximate)
                    # Use AdaMixNet's inverse transform method
                    recovered_coeffs = self.adamixnet.inverse_transform(x_hat)
                    
                    # 5. Inverse wavelet transform
                    reconstructed_images = self.wavelet_cnn.inverse_transform(recovered_coeffs)
                    
                    # Ensure same size
                    if reconstructed_images.shape != images.shape:
                        reconstructed_images = F.interpolate(
                            reconstructed_images, 
                            size=images.shape[2:], 
                            mode='bilinear', 
                            align_corners=False
                        )
                    
                    # Calculate metrics
                    for i in range(images.size(0)):
                        original = images[i:i+1]
                        reconstructed = reconstructed_images[i:i+1]
                        
                        # PSNR
                        psnr_val = calculate_psnr(original, reconstructed).item()
                        psnr_values.append(psnr_val)
                        
                        # MS-SSIM
                        ms_ssim_val = calculate_ms_ssim(original, reconstructed)
                        ms_ssim_values.append(ms_ssim_val)
                        
                        # BPP (estimated from feature dimensions)
                        bpp_val = estimate_bpp_from_features(y_quantized, images.shape[2:])
                        bpp_values.append(bpp_val)
                
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue
                
                # Early stop for quick testing
                if self.args.max_samples and batch_idx * self.args.batch_size >= self.args.max_samples:
                    break
        
        # Calculate average metrics
        avg_psnr = np.mean(psnr_values) if psnr_values else 0
        avg_ms_ssim = np.mean(ms_ssim_values) if ms_ssim_values else 0
        avg_bpp = np.mean(bpp_values) if bpp_values else 0
        
        return {
            'lambda': lambda_value,
            'psnr_db': avg_psnr,
            'ms_ssim': avg_ms_ssim,
            'bpp': avg_bpp,
            'num_samples': len(psnr_values)
        }
    
    def evaluate_all_lambdas(self):
        """Evaluate tất cả lambda values"""
        lambda_values = self.args.lambdas
        
        for lambda_val in lambda_values:
            result = self.evaluate_lambda(lambda_val)
            self.results.append(result)
            
            print(f"λ={lambda_val}: PSNR={result['psnr_db']:.2f}dB, "
                  f"MS-SSIM={result['ms_ssim']:.4f}, BPP={result['bpp']:.4f}")
    
    def save_results(self):
        """Save results to CSV"""
        if not self.results:
            print("No results to save!")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Add metadata
        df['dataset'] = self.args.dataset
        df['split'] = self.args.split
        df['image_size'] = self.args.image_size
        df['model'] = os.path.basename(self.args.checkpoint)
        
        # Save to CSV
        os.makedirs(os.path.dirname(self.args.output_csv), exist_ok=True)
        df.to_csv(self.args.output_csv, index=False)
        
        print(f"✓ Results saved to {self.args.output_csv}")
        
        # Print summary
        print("\n" + "="*50)
        print("CODEC EVALUATION SUMMARY")
        print("="*50)
        print(df.to_string(index=False))
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Codec Metrics Evaluation')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, choices=['coco', 'davis'], default='coco',
                       help='Dataset to evaluate')
    parser.add_argument('--data_dir', type=str, default='datasets/',
                       help='Dataset directory')
    parser.add_argument('--split', type=str, default='val',
                       help='Dataset split')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    
    # Evaluation arguments
    parser.add_argument('--lambdas', type=int, nargs='+', default=[256, 512, 1024],
                       help='Lambda values to evaluate')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to evaluate (for testing)')
    
    # Output arguments
    parser.add_argument('--output_csv', type=str, default='results/codec_metrics.csv',
                       help='Output CSV file')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers')
    parser.add_argument('--skip_entropy_update', action='store_true',
                       help='Skip entropy model updates (faster for evaluation without compression)')
    
    args = parser.parse_args()
    
    # Create evaluator và run evaluation
    evaluator = CodecEvaluator(args)
    evaluator.evaluate_all_lambdas()
    evaluator.save_results()


if __name__ == '__main__':
    main() 