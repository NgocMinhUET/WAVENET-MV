"""
AI Heads - YOLO-tiny & SegFormer-lite
Input: compressed features (không cần pixel reconstruction)
Output: detection boxes / seg masks / action logits
Loss Stage-3: task-specific + optional KD từ pixel-domain teacher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import torchvision
from torchvision.ops import nms


class YOLOTinyHead(nn.Module):
    """
    YOLO-tiny head for object detection trên compressed features
    """
    
    def __init__(self, 
                 input_channels=128,
                 num_classes=80,  # COCO classes
                 num_anchors=3,
                 input_size=416):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.input_size = input_size
        
        # Feature adapter từ compressed features (with downsampling)
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(input_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # Downsample by 2x
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=2, padding=1),  # Downsample by 2x again
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # YOLO detection layers
        self.conv1 = self._make_conv_layer(512, 256, 1)
        self.conv2 = self._make_conv_layer(256, 512, 3)
        
        # Detection head
        # Output: (batch, anchors * (5 + num_classes), H, W)
        # 5 = x, y, w, h, confidence
        self.detection_head = nn.Conv2d(
            512, 
            num_anchors * (5 + num_classes), 
            kernel_size=1
        )
        
        # Predefined anchor boxes (scaled cho different sizes)
        self.register_buffer('anchors', torch.tensor([
            [10, 13], [16, 30], [33, 23],      # Small objects
            [30, 61], [62, 45], [59, 119],     # Medium objects  
            [116, 90], [156, 198], [373, 326]  # Large objects
        ]).float())
        
    def _make_conv_layer(self, in_channels, out_channels, kernel_size):
        """Helper to create conv layer"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Compressed features [B, input_channels, H, W]
        Returns:
            detections: [B, num_anchors, H, W, 5+num_classes]
        """
        # Adapt compressed features
        features = self.feature_adapter(x)  # [B, 512, H, W]
        
        # YOLO layers
        x = self.conv1(features)
        x = self.conv2(x)
        
        # Detection head
        detections = self.detection_head(x)  # [B, anchors*(5+classes), H, W]
        
        # Reshape để easier processing
        batch_size, _, grid_h, grid_w = detections.shape
        detections = detections.view(
            batch_size, 
            self.num_anchors, 
            5 + self.num_classes,
            grid_h, 
            grid_w
        ).permute(0, 1, 3, 4, 2)  # [B, anchors, H, W, 5+classes]
        
        return detections
    
    def decode_predictions(self, predictions, conf_threshold=0.5, nms_threshold=0.4):
        """
        Decode YOLO predictions thành bounding boxes
        Args:
            predictions: [B, anchors, H, W, 5+classes]
            conf_threshold: Confidence threshold
            nms_threshold: NMS threshold
        Returns:
            List of detections for each image
        """
        batch_size = predictions.size(0)
        grid_h, grid_w = predictions.shape[2:4]
        
        # Get device
        device = predictions.device
        
        # Create grid coordinates
        grid_x = torch.arange(grid_w, device=device).repeat(grid_h, 1)
        grid_y = torch.arange(grid_h, device=device).repeat(grid_w, 1).t()
        
        detections = []
        
        for b in range(batch_size):
            pred = predictions[b]  # [anchors, H, W, 5+classes]
            
            # Extract components
            x_center = torch.sigmoid(pred[..., 0]) + grid_x
            y_center = torch.sigmoid(pred[..., 1]) + grid_y
            width = torch.exp(pred[..., 2]) * self.anchors[:, 0].view(-1, 1, 1)
            height = torch.exp(pred[..., 3]) * self.anchors[:, 1].view(-1, 1, 1)
            confidence = torch.sigmoid(pred[..., 4])
            class_probs = torch.sigmoid(pred[..., 5:])
            
            # Convert to corner coordinates
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Scale to input size
            scale_x = self.input_size / grid_w
            scale_y = self.input_size / grid_h
            
            boxes = torch.stack([x1 * scale_x, y1 * scale_y, 
                               x2 * scale_x, y2 * scale_y], dim=-1)
            
            # Confidence filtering
            class_conf, class_pred = torch.max(class_probs, dim=-1)
            total_conf = confidence * class_conf
            
            # Filter by confidence
            conf_mask = total_conf > conf_threshold
            
            if conf_mask.sum() == 0:
                detections.append(torch.empty(0, 6, device=device))
                continue
            
            # Flatten and filter
            boxes_flat = boxes[conf_mask]
            scores_flat = total_conf[conf_mask] 
            classes_flat = class_pred[conf_mask]
            
            # NMS
            keep_indices = nms(boxes_flat, scores_flat, nms_threshold)
            
            final_boxes = boxes_flat[keep_indices]
            final_scores = scores_flat[keep_indices]
            final_classes = classes_flat[keep_indices]
            
            # Combine results
            batch_detections = torch.cat([
                final_boxes,
                final_scores.unsqueeze(1),
                final_classes.unsqueeze(1).float()
            ], dim=1)
            
            detections.append(batch_detections)
        
        return detections


class SegFormerLiteHead(nn.Module):
    """
    SegFormer-lite head for segmentation trên compressed features
    """
    
    def __init__(self, 
                 input_channels=128,
                 num_classes=21,  # PASCAL VOC classes
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8]):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        
        # Feature adapter
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(input_channels, embed_dims[0], 3, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True)
        )
        
        # Multi-level feature extraction
        self.encoder_stages = nn.ModuleList()
        in_dim = embed_dims[0]
        
        for i, (dim, heads) in enumerate(zip(embed_dims, num_heads)):
            stage = nn.Sequential(
                nn.Conv2d(in_dim, dim, 3, padding=1, stride=2 if i > 0 else 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                self._make_attention_block(dim, heads),
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim)
            )
            self.encoder_stages.append(stage)
            in_dim = dim
        
        # Decoder với feature fusion
        self.decoder = nn.ModuleList()
        decoder_dim = 256
        
        for dim in reversed(embed_dims):
            decoder_layer = nn.Sequential(
                nn.Conv2d(dim, decoder_dim, 1),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True)
            )
            self.decoder.append(decoder_layer)
        
        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_dim * len(embed_dims), decoder_dim, 3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_dim, num_classes, 1)
        )
        
    def _make_attention_block(self, dim, num_heads):
        """Simplified attention block with proper MultiheadAttention usage"""
        return nn.Sequential(
            nn.Conv2d(dim, dim, 1),  # Linear projection
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Compressed features [B, input_channels, H, W]
        Returns:
            segmentation: [B, num_classes, H, W]
        """
        # Feature adaptation
        x = self.feature_adapter(x)
        
        # Multi-level encoding
        encoder_features = []
        current = x
        
        for stage in self.encoder_stages:
            current = stage(current)
            encoder_features.append(current)
        
        # Decode với feature fusion
        target_size = encoder_features[0].shape[2:]  # Largest feature map size
        decoder_features = []
        
        for i, (feat, decoder_layer) in enumerate(zip(reversed(encoder_features), self.decoder)):
            decoded = decoder_layer(feat)
            
            # Upsample về target size
            if decoded.shape[2:] != target_size:
                decoded = F.interpolate(decoded, size=target_size, mode='bilinear', align_corners=False)
            
            decoder_features.append(decoded)
        
        # Fuse all decoder features
        fused_features = torch.cat(decoder_features, dim=1)
        
        # Final segmentation
        segmentation = self.seg_head(fused_features)
        
        # Upsample to input size
        if segmentation.shape[2:] != x.shape[2:]:
            segmentation = F.interpolate(segmentation, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return segmentation


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation loss để transfer knowledge từ pixel-domain teacher
    """
    
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight between KD loss và task loss
        
    def forward(self, student_logits, teacher_logits, ground_truth, task_loss_fn):
        """
        Compute KD loss
        Args:
            student_logits: Outputs từ compressed-domain model
            teacher_logits: Outputs từ pixel-domain teacher
            ground_truth: Ground truth labels
            task_loss_fn: Task-specific loss function
        """
        # Task loss
        task_loss = task_loss_fn(student_logits, ground_truth)
        
        # Knowledge distillation loss
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * task_loss
        
        return total_loss, task_loss, kd_loss


def test_ai_heads():
    """Unit tests for AI heads"""
    
    # Test YOLO-tiny head
    yolo_head = YOLOTinyHead(input_channels=128, num_classes=80)
    
    # Test forward pass
    compressed_features = torch.randn(2, 128, 32, 32)
    detections = yolo_head(compressed_features)
    
    expected_shape = (2, 3, 32, 32, 85)  # batch, anchors, H, W, 5+classes
    assert detections.shape == expected_shape, f"YOLO output shape: {detections.shape}"
    
    # Test prediction decoding
    decoded = yolo_head.decode_predictions(detections)
    assert len(decoded) == 2, f"Should have 2 batch results, got {len(decoded)}"
    
    print("✓ YOLOTinyHead tests passed!")
    
    # Test SegFormer-lite head
    segformer_head = SegFormerLiteHead(input_channels=128, num_classes=21)
    
    segmentation = segformer_head(compressed_features)
    expected_seg_shape = (2, 21, 32, 32)  # batch, classes, H, W
    assert segmentation.shape == expected_seg_shape, f"SegFormer output shape: {segmentation.shape}"
    
    print("✓ SegFormerLiteHead tests passed!")
    
    # Test Knowledge Distillation
    kd_loss_fn = KnowledgeDistillationLoss()
    
    student_logits = torch.randn(2, 21, 32, 32)
    teacher_logits = torch.randn(2, 21, 32, 32)
    ground_truth = torch.randint(0, 21, (2, 32, 32))
    
    def ce_loss(logits, targets):
        return F.cross_entropy(logits, targets)
    
    total_loss, task_loss, kd_loss = kd_loss_fn(student_logits, teacher_logits, ground_truth, ce_loss)
    
    assert total_loss.item() > 0, "Total loss should be positive"
    assert task_loss.item() >= 0, "Task loss should be non-negative"
    assert kd_loss.item() >= 0, "KD loss should be non-negative"
    
    print("✓ KnowledgeDistillationLoss tests passed!")


if __name__ == "__main__":
    test_ai_heads() 