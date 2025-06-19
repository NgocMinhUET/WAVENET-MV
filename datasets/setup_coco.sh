#!/bin/bash

# COCO 2017 Dataset Setup Script
echo "Setting up COCO 2017 Dataset..."

# Create COCO directory
mkdir -p datasets/COCO
cd datasets/COCO

# Download COCO 2017 validation images
echo "Downloading COCO 2017 validation images..."
if [ ! -f "val2017.zip" ]; then
    wget http://images.cocodataset.org/zips/val2017.zip -O val2017.zip
fi

# Extract validation images
if [ ! -d "val2017" ]; then
    echo "Extracting validation images..."
    unzip val2017.zip
    rm val2017.zip
fi

# Download annotations
echo "Downloading COCO 2017 annotations..."
if [ ! -f "annotations_trainval2017.zip" ]; then
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O annotations_trainval2017.zip
fi

# Extract annotations
if [ ! -d "annotations" ]; then
    echo "Extracting annotations..."
    unzip annotations_trainval2017.zip
    rm annotations_trainval2017.zip
fi

# Optional: Download train images (large, ~18GB)
read -p "Do you want to download COCO train2017 images? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading COCO 2017 training images (this will take a while)..."
    if [ ! -f "train2017.zip" ]; then
        wget http://images.cocodataset.org/zips/train2017.zip -O train2017.zip
    fi
    
    if [ ! -d "train2017" ]; then
        echo "Extracting training images..."
        unzip train2017.zip
        rm train2017.zip
    fi
fi

# Verify dataset structure
echo "Verifying COCO dataset structure..."
if [ -d "val2017" ] && [ -d "annotations" ]; then
    echo "✓ COCO dataset setup completed successfully!"
    echo "✓ Val images: $(find val2017 -name "*.jpg" | wc -l) files"
    echo "✓ Annotations: $(find annotations -name "*.json" | wc -l) files"
    
    # Check if train images exist
    if [ -d "train2017" ]; then
        echo "✓ Train images: $(find train2017 -name "*.jpg" | wc -l) files"
    fi
else
    echo "✗ COCO dataset setup failed!"
    exit 1
fi

cd ../..
echo "COCO setup complete. Dataset located at: datasets/COCO/" 