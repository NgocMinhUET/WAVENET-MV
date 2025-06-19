"""
Dataset Loaders cho COCO 2017 và DAVIS 2017
Hỗ trợ cả training và evaluation
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import albumentations as A
from albumentations.pytorch import ToTensorV2


class COCODatasetLoader(Dataset):
    """
    COCO 2017 Dataset Loader
    Hỗ trợ cả detection và segmentation tasks
    """
    
    def __init__(self, 
                 data_dir='datasets/COCO',
                 subset='val',  # 'train' or 'val'
                 image_size=256,
                 task='detection',  # 'detection' or 'segmentation'
                 augmentation=True):
        
        self.data_dir = data_dir
        self.subset = subset
        self.image_size = image_size
        self.task = task
        self.augmentation = augmentation and (subset == 'train')
        
        # COCO annotation file paths
        if subset == 'val':
            self.image_dir = os.path.join(data_dir, 'val2017')
            self.ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
        else:
            # For training, we use val2017 as proxy (since we don't have train2017 in setup)
            self.image_dir = os.path.join(data_dir, 'val2017')
            self.ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
        
        # Initialize COCO API
        if os.path.exists(self.ann_file):
            self.coco = COCO(self.ann_file)
            self.image_ids = list(self.coco.imgs.keys())
        else:
            # Fallback: load images directly from directory
            print(f"Warning: Annotation file not found at {self.ann_file}")
            print("Loading images directly from directory...")
            self.coco = None
            self.image_ids = []
            if os.path.exists(self.image_dir):
                for file in os.listdir(self.image_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_ids.append(file)
        
        # Setup transforms
        self.setup_transforms()
        
        print(f"COCO Dataset loaded: {len(self.image_ids)} images from {subset}")
        
    def setup_transforms(self):
        """Setup augmentation transforms"""
        if self.augmentation:
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        if self.coco is not None:
            # Use COCO API
            image_id = self.image_ids[idx]
            image_info = self.coco.imgs[image_id]
            image_path = os.path.join(self.image_dir, image_info['file_name'])
        else:
            # Direct file loading
            image_path = os.path.join(self.image_dir, self.image_ids[idx])
            image_id = idx
        
        # Load image
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            # Fallback to PIL
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Prepare return data
        data = {
            'image': image,
            'image_id': image_id,
        }
        
        # Add annotations if available
        if self.coco is not None:
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            
            if self.task == 'detection':
                # Detection annotations
                boxes = []
                labels = []
                areas = []
                
                for ann in anns:
                    if 'bbox' in ann and ann['area'] > 0:
                        x, y, w, h = ann['bbox']
                        boxes.append([x, y, x + w, y + h])
                        labels.append(ann['category_id'])
                        areas.append(ann['area'])
                
                data.update({
                    'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
                    'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
                    'areas': torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros(0)
                })
            
            elif self.task == 'segmentation':
                # Segmentation masks
                masks = []
                labels = []
                
                for ann in anns:
                    if 'segmentation' in ann:
                        mask = coco_mask.decode(coco_mask.frPyObjects(
                            ann['segmentation'], 
                            image_info['height'], 
                            image_info['width']
                        ))
                        if mask.ndim == 3:
                            mask = mask.max(axis=2)  # Multiple polygons
                        masks.append(mask)
                        labels.append(ann['category_id'])
                
                if masks:
                    masks = np.stack(masks, axis=0)
                    data.update({
                        'masks': torch.tensor(masks, dtype=torch.float32),
                        'labels': torch.tensor(labels, dtype=torch.long)
                    })
                else:
                    data.update({
                        'masks': torch.zeros(0, self.image_size, self.image_size),
                        'labels': torch.zeros(0, dtype=torch.long)
                    })
        
        return data


class DAVISDatasetLoader(Dataset):
    """
    DAVIS 2017 Dataset Loader
    Primarily for video segmentation tasks
    """
    
    def __init__(self, 
                 data_dir='datasets/DAVIS',
                 subset='val',  # 'train' or 'val'
                 image_size=256,
                 augmentation=True):
        
        self.data_dir = data_dir
        self.subset = subset
        self.image_size = image_size
        self.augmentation = augmentation and (subset == 'train')
        
        # DAVIS directory structure
        self.image_dir = os.path.join(data_dir, 'DAVIS/JPEGImages/480p')
        self.mask_dir = os.path.join(data_dir, 'DAVIS/Annotations/480p')
        
        # Find all video sequences
        self.sequences = []
        if os.path.exists(self.image_dir):
            for seq_name in os.listdir(self.image_dir):
                seq_path = os.path.join(self.image_dir, seq_name)
                if os.path.isdir(seq_path):
                    self.sequences.append(seq_name)
        
        # Build frame list
        self.frame_list = []
        for seq_name in self.sequences:
            seq_image_dir = os.path.join(self.image_dir, seq_name)
            seq_mask_dir = os.path.join(self.mask_dir, seq_name)
            
            if os.path.exists(seq_image_dir):
                for frame_file in os.listdir(seq_image_dir):
                    if frame_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        frame_path = os.path.join(seq_image_dir, frame_file)
                        mask_path = os.path.join(seq_mask_dir, frame_file.replace('.jpg', '.png'))
                        
                        self.frame_list.append({
                            'sequence': seq_name,
                            'frame': frame_file,
                            'image_path': frame_path,
                            'mask_path': mask_path if os.path.exists(mask_path) else None
                        })
        
        # Split data (80% train, 20% val)
        if subset == 'train':
            self.frame_list = self.frame_list[:int(0.8 * len(self.frame_list))]
        else:
            self.frame_list = self.frame_list[int(0.8 * len(self.frame_list)):]
        
        # Setup transforms
        self.setup_transforms()
        
        print(f"DAVIS Dataset loaded: {len(self.frame_list)} frames from {len(self.sequences)} sequences ({subset})")
    
    def setup_transforms(self):
        """Setup augmentation transforms"""
        if self.augmentation:
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        else:
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
    
    def __len__(self):
        return len(self.frame_list)
    
    def __getitem__(self, idx):
        frame_info = self.frame_list[idx]
        
        # Load image
        try:
            image = cv2.imread(frame_info['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            image = Image.open(frame_info['image_path']).convert('RGB')
            image = np.array(image)
        
        # Load mask if available
        mask = None
        if frame_info['mask_path'] and os.path.exists(frame_info['mask_path']):
            try:
                mask = cv2.imread(frame_info['mask_path'], cv2.IMREAD_GRAYSCALE)
            except:
                mask = Image.open(frame_info['mask_path']).convert('L')
                mask = np.array(mask)
        
        # Apply transforms
        if mask is not None and self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        elif self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            if mask is not None:
                mask = torch.from_numpy(mask).float() / 255.0
        
        # Prepare return data
        data = {
            'image': image,
            'sequence': frame_info['sequence'],
            'frame': frame_info['frame'],
        }
        
        if mask is not None:
            data['mask'] = mask
        
        return data


def test_dataset_loaders():
    """Test dataset loaders"""
    
    # Test COCO loader
    try:
        coco_dataset = COCODatasetLoader(
            data_dir='datasets/COCO',
            subset='val',
            image_size=256,
            task='detection'
        )
        
        if len(coco_dataset) > 0:
            sample = coco_dataset[0]
            print(f"✓ COCO Dataset test passed! Sample keys: {sample.keys()}")
            print(f"  Image shape: {sample['image'].shape}")
        else:
            print("⚠ COCO Dataset empty - check data directory")
            
    except Exception as e:
        print(f"⚠ COCO Dataset test failed: {e}")
    
    # Test DAVIS loader
    try:
        davis_dataset = DAVISDatasetLoader(
            data_dir='datasets/DAVIS',
            subset='val',
            image_size=256
        )
        
        if len(davis_dataset) > 0:
            sample = davis_dataset[0]
            print(f"✓ DAVIS Dataset test passed! Sample keys: {sample.keys()}")
            print(f"  Image shape: {sample['image'].shape}")
        else:
            print("⚠ DAVIS Dataset empty - check data directory")
            
    except Exception as e:
        print(f"⚠ DAVIS Dataset test failed: {e}")


if __name__ == "__main__":
    test_dataset_loaders() 