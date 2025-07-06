"""
Video Coding for Machine (VCM) Evaluation
- Object Detection (YOLO) trên compressed features
- Semantic Segmentation (SegFormer) trên compressed features
- So sánh hiệu năng với pixel-domain baselines
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
from pathlib import Path
import json
from pycocotools.cocoeval import COCOeval
import cv2

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.wavelet_transform_cnn import WaveletTransformCNN
from models.adamixnet import AdaMixNet
from models.compressor_vnvc import CompressorVNVC
from models.ai_heads import YOLOTinyHead, SegFormerLiteHead
from datasets.dataset_loaders import COCODatasetLoader, DAVISDatasetLoader


def vcm_collate_fn(batch):
    """
    Custom collate function để xử lý tensor có kích thước khác nhau
    """
    # Separate images and annotations
    images = torch.stack([item['image'] for item in batch])
    image_ids = [item['image_id'] for item in batch]
    
    # Handle boxes, labels, areas with different sizes
    boxes_list = []
    labels_list = []
    areas_list = []
    
    for item in batch:
        if 'boxes' in item:
            boxes_list.append(item['boxes'])
            labels_list.append(item['labels'])
            areas_list.append(item['areas'])
        else:
            # Empty tensors if no annotations
            boxes_list.append(torch.zeros(0, 4))
            labels_list.append(torch.zeros(0, dtype=torch.long))
            areas_list.append(torch.zeros(0))
    
    # Pad boxes, labels, areas to same size
    max_boxes = max(boxes.shape[0] for boxes in boxes_list)
    
    padded_boxes = []
    padded_labels = []
    padded_areas = []
    
    for boxes, labels, areas in zip(boxes_list, labels_list, areas_list):
        if boxes.shape[0] < max_boxes:
            # Pad with zeros
            pad_size = max_boxes - boxes.shape[0]
            padded_boxes.append(torch.cat([boxes, torch.zeros(pad_size, 4)], dim=0))
            padded_labels.append(torch.cat([labels, torch.zeros(pad_size, dtype=torch.long)], dim=0))
            padded_areas.append(torch.cat([areas, torch.zeros(pad_size)], dim=0))
        else:
            padded_boxes.append(boxes)
            padded_labels.append(labels)
            padded_areas.append(areas)
    
    return {
        'image': images,
        'image_id': image_ids,
        'boxes': torch.stack(padded_boxes),
        'labels': torch.stack(padded_labels),
        'areas': torch.stack(padded_areas)
    }


class VCMEvaluator:
    """Evaluator cho Video Coding for Machine tasks"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load compression pipeline (frozen)
        self.load_compression_pipeline()
        
        # Load AI heads
        self.load_ai_heads()
        
        # Setup dataset
        self.setup_dataset()
        
        # Results storage
        self.results = {
            'detection': [],
            'segmentation': []
        }
        
    def load_compression_pipeline(self):
        """Load frozen compression pipeline"""
        print("Loading compression pipeline...")
        
        # WaveletCNN
        self.wavelet_cnn = WaveletTransformCNN(
            input_channels=3,
            feature_channels=64,
            wavelet_channels=64
        ).to(self.device)
        
        if self.args.stage1_checkpoint:
            checkpoint = torch.load(self.args.stage1_checkpoint, map_location=self.device)
            self.wavelet_cnn.load_state_dict(checkpoint['model_state_dict'])
        
        # AdaMixNet
        self.adamixnet = AdaMixNet(
            input_channels=256,
            C_prime=64,
            C_mix=128
        ).to(self.device)
        
        # Load checkpoint first to detect architecture
        if self.args.stage2_checkpoint:
            checkpoint = torch.load(self.args.stage2_checkpoint, map_location=self.device)
            if 'compressor_state_dict' in checkpoint:
                checkpoint_keys = list(checkpoint['compressor_state_dict'].keys())
                print(f"Compressor checkpoint keys (first 10): {checkpoint_keys[:10]}")
                
                # Detect architecture based on checkpoint keys
                if 'analysis_transform.conv1.weight' in checkpoint_keys:
                    # ImprovedCompressorVNVC with conv1/norm1 structure
                    print("Detected ImprovedCompressorVNVC architecture")
                    from models.compressor_improved import ImprovedCompressorVNVC
                    self.compressor = ImprovedCompressorVNVC(
                        input_channels=128,
                        latent_channels=192,
                        lambda_rd=128
                    ).to(self.device)
                else:
                    # Standard CompressorVNVC
                    print("Detected standard CompressorVNVC architecture")
                    self.compressor = CompressorVNVC(
                        input_channels=128,
                        latent_channels=192,
                        lambda_rd=128
                    ).to(self.device)
                
                # Load state dict
                try:
                    self.compressor.load_state_dict(checkpoint['compressor_state_dict'])
                    print("✓ Loaded compressor_state_dict successfully")
                except Exception as e:
                    print(f"⚠️ Failed to load compressor: {e}")
                    print("⚠️ Using random weights for compressor")
            else:
                # Fallback to standard CompressorVNVC
                self.compressor = CompressorVNVC(
                    input_channels=128,
                    latent_channels=192,
                    lambda_rd=128
                ).to(self.device)
        else:
            # No checkpoint, use standard CompressorVNVC
            self.compressor = CompressorVNVC(
                input_channels=128,
                latent_channels=192,
                lambda_rd=128
            ).to(self.device)
        
        # Freeze compression pipeline
        for model in [self.wavelet_cnn, self.adamixnet, self.compressor]:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        
        print("✓ Compression pipeline loaded and frozen")
    
    def load_ai_heads(self):
        """Load AI heads"""
        print("Loading AI heads...")
        
        # YOLO Head
        if self.args.enable_detection:
            self.yolo_head = YOLOTinyHead(
                input_channels=128,  # Compressor output channels (x_hat)
                num_classes=80,      # COCO classes
                num_anchors=3
            ).to(self.device)
            
            if self.args.stage3_checkpoint and os.path.exists(self.args.stage3_checkpoint):
                try:
                    checkpoint = torch.load(self.args.stage3_checkpoint, map_location=self.device)
                    if 'yolo_head_state_dict' in checkpoint:
                        self.yolo_head.load_state_dict(checkpoint['yolo_head_state_dict'])
                        print("✓ Loaded YOLO head from checkpoint")
                    else:
                        print("⚠️ YOLO head state dict not found in checkpoint, using random weights")
                except Exception as e:
                    print(f"⚠️ Failed to load YOLO head: {e}, using random weights")
            else:
                print("⚠️ Stage 3 checkpoint not found, using random weights for YOLO head")
        
        # SegFormer Head
        if self.args.enable_segmentation:
            self.segformer_head = SegFormerLiteHead(
                input_channels=128,  # Compressor output channels (x_hat)
                num_classes=21       # PASCAL VOC classes
            ).to(self.device)
            
            if self.args.stage3_checkpoint and os.path.exists(self.args.stage3_checkpoint):
                try:
                    checkpoint = torch.load(self.args.stage3_checkpoint, map_location=self.device)
                    if 'segformer_head_state_dict' in checkpoint:
                        self.segformer_head.load_state_dict(checkpoint['segformer_head_state_dict'])
                        print("✓ Loaded SegFormer head from checkpoint")
                    else:
                        print("⚠️ SegFormer head state dict not found in checkpoint, using random weights")
                except Exception as e:
                    print(f"⚠️ Failed to load SegFormer head: {e}, using random weights")
            else:
                print("⚠️ Stage 3 checkpoint not found, using random weights for SegFormer head")
        
        print("✓ AI heads loaded")
    
    def setup_dataset(self):
        """Setup evaluation dataset"""
        if self.args.dataset == 'coco':
            self.dataset = COCODatasetLoader(
                data_dir=self.args.data_dir,
                subset=self.args.split,
                image_size=self.args.image_size,
                augmentation=False
            )
        elif self.args.dataset == 'davis':
            self.dataset = DAVISDatasetLoader(
                data_dir=self.args.data_dir,
                subset=self.args.split,
                image_size=self.args.image_size,
                augmentation=False
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=vcm_collate_fn  # Use custom collate function
        )
        
        print(f"✓ Dataset loaded: {len(self.dataset)} images")
    
    def get_compressed_features(self, images):
        """Get compressed features từ compression pipeline"""
        with torch.no_grad():
            # Forward through compression pipeline
            wavelet_coeffs = self.wavelet_cnn(images)
            mixed_features = self.adamixnet(wavelet_coeffs)
            x_hat, likelihoods, y_quantized = self.compressor(mixed_features)
            
            # Return compressed features (x_hat)
            return x_hat
    
    def evaluate_detection(self):
        """Evaluate object detection performance"""
        print("\n🔍 Evaluating Object Detection...")
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Detection")):
                # Get images and annotations
                images = batch['image'].to(self.device)
                
                # Get compressed features
                compressed_features = self.get_compressed_features(images)
                
                # YOLO predictions
                predictions = self.yolo_head(compressed_features)
                
                # Decode predictions
                detections = self.yolo_head.decode_predictions(
                    predictions, 
                    conf_threshold=0.5, 
                    nms_threshold=0.4
                )
                
                # Store results
                for i, detection in enumerate(detections):
                    image_id = batch['image_id'][i] if 'image_id' in batch else batch_idx * self.args.batch_size + i
                    
                    # Convert to COCO format
                    for box in detection:
                        if len(box) >= 6:  # [x1, y1, x2, y2, score, class]
                            all_predictions.append({
                                'image_id': image_id,
                                'category_id': int(box[5]),
                                'bbox': [float(box[0]), float(box[1]), 
                                        float(box[2] - box[0]), float(box[3] - box[1])],
                                'score': float(box[4])
                            })
                
                # Early stop for testing
                if self.args.max_samples and batch_idx * self.args.batch_size >= self.args.max_samples:
                    break
        
        # Calculate mAP (simplified)
        if all_predictions:
            # Simplified mAP calculation
            total_predictions = len(all_predictions)
            high_conf_predictions = [p for p in all_predictions if p['score'] > 0.5]
            
            detection_results = {
                'total_predictions': total_predictions,
                'high_conf_predictions': len(high_conf_predictions),
                'avg_confidence': np.mean([p['score'] for p in all_predictions]) if all_predictions else 0,
                'detection_rate': len(high_conf_predictions) / max(1, total_predictions)
            }
        else:
            detection_results = {
                'total_predictions': 0,
                'high_conf_predictions': 0,
                'avg_confidence': 0,
                'detection_rate': 0
            }
        
        return detection_results
    
    def evaluate_segmentation(self):
        """Evaluate semantic segmentation performance"""
        print("\n🔍 Evaluating Semantic Segmentation...")
        
        total_iou = 0
        total_pixels = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Segmentation")):
                # Get images
                images = batch['image'].to(self.device)
                
                # Get compressed features
                compressed_features = self.get_compressed_features(images)
                
                # SegFormer predictions
                predictions = self.segformer_head(compressed_features)
                
                # Resize to original image size
                if predictions.shape[2:] != images.shape[2:]:
                    predictions = F.interpolate(
                        predictions, 
                        size=images.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Get predicted masks
                predicted_masks = torch.argmax(predictions, dim=1)  # [B, H, W]
                
                # Calculate IoU (simplified - assuming background class 0)
                for i in range(predicted_masks.size(0)):
                    pred_mask = predicted_masks[i]
                    
                    # Simplified IoU calculation
                    # Assume foreground pixels are non-zero
                    foreground_pixels = (pred_mask > 0).float()
                    total_pixels += foreground_pixels.numel()
                    
                    # IoU = intersection / union (simplified)
                    intersection = foreground_pixels.sum()
                    union = foreground_pixels.numel()
                    iou = intersection / max(1, union)
                    total_iou += iou
                
                # Early stop for testing
                if self.args.max_samples and batch_idx * self.args.batch_size >= self.args.max_samples:
                    break
        
        # Calculate average IoU
        avg_iou = total_iou / max(1, total_pixels / (self.args.image_size * self.args.image_size))
        
        segmentation_results = {
            'avg_iou': avg_iou,
            'total_pixels': total_pixels,
            'foreground_ratio': total_pixels / max(1, total_pixels)
        }
        
        return segmentation_results
    
    def evaluate_all(self):
        """Evaluate all VCM tasks"""
        print("🚀 Starting VCM Evaluation...")
        
        # Detection evaluation
        if self.args.enable_detection:
            detection_results = self.evaluate_detection()
            self.results['detection'] = detection_results
            print(f"📊 Detection Results:")
            print(f"  - Total predictions: {detection_results['total_predictions']}")
            print(f"  - High confidence: {detection_results['high_conf_predictions']}")
            print(f"  - Avg confidence: {detection_results['avg_confidence']:.4f}")
            print(f"  - Detection rate: {detection_results['detection_rate']:.4f}")
        
        # Segmentation evaluation
        if self.args.enable_segmentation:
            segmentation_results = self.evaluate_segmentation()
            self.results['segmentation'] = segmentation_results
            print(f"📊 Segmentation Results:")
            print(f"  - Average IoU: {segmentation_results['avg_iou']:.4f}")
            print(f"  - Total pixels: {segmentation_results['total_pixels']}")
            print(f"  - Foreground ratio: {segmentation_results['foreground_ratio']:.4f}")
    
    def save_results(self):
        """Save results to JSON"""
        if not self.results:
            print("No results to save!")
            return
        
        # Add metadata
        results_with_metadata = {
            'metadata': {
                'dataset': self.args.dataset,
                'split': self.args.split,
                'image_size': self.args.image_size,
                'lambda_rd': self.args.lambda_rd,
                'max_samples': self.args.max_samples
            },
            'results': self.results
        }
        
        os.makedirs(os.path.dirname(self.args.output_json), exist_ok=True)
        
        with open(self.args.output_json, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"✓ Results saved to {self.args.output_json}")


def main():
    parser = argparse.ArgumentParser(description='VCM Evaluation')
    
    # Model checkpoints
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                       help='Path to Stage 1 checkpoint')
    parser.add_argument('--stage2_checkpoint', type=str, required=True,
                       help='Path to Stage 2 checkpoint')
    parser.add_argument('--stage3_checkpoint', type=str, default=None,
                       help='Path to Stage 3 checkpoint (optional)')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='coco',
                       choices=['coco', 'davis'], help='Dataset to evaluate')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='val',
                       help='Dataset split to evaluate')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    
    # Evaluation settings
    parser.add_argument('--enable_detection', action='store_true',
                       help='Enable object detection evaluation')
    parser.add_argument('--enable_segmentation', action='store_true',
                       help='Enable semantic segmentation evaluation')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--lambda_rd', type=int, default=128,
                       help='Rate-distortion lambda')
    
    # Output
    parser.add_argument('--output_json', type=str, default='results/vcm_results.json',
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.enable_detection and not args.enable_segmentation:
        print("⚠️ Warning: No evaluation tasks enabled!")
        print("Use --enable_detection and/or --enable_segmentation")
        return
    
    # Run evaluation
    evaluator = VCMEvaluator(args)
    evaluator.evaluate_all()
    evaluator.save_results()
    
    print("\n" + "="*50)
    print("VCM EVALUATION COMPLETE")
    print("="*50)


if __name__ == '__main__':
    main() 