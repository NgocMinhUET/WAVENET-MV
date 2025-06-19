#!/bin/bash

# DAVIS 2017 Dataset Setup Script
echo "Setting up DAVIS 2017 Dataset..."

# Create DAVIS directory
mkdir -p datasets/DAVIS
cd datasets/DAVIS

# Download DAVIS 2017 dataset
echo "Downloading DAVIS 2017 dataset..."
if [ ! -f "DAVIS-2017-trainval-480p.zip" ]; then
    # DAVIS official download link
    wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip -O DAVIS-2017-trainval-480p.zip
    
    # If official link fails, try alternative
    if [ ! -f "DAVIS-2017-trainval-480p.zip" ]; then
        echo "Official link failed, trying alternative..."
        # Alternative download (you may need to provide your own link)
        echo "Please manually download DAVIS-2017-trainval-480p.zip from https://davischallenge.org/davis2017/code.html"
        echo "and place it in datasets/DAVIS/ directory"
        exit 1
    fi
fi

# Extract DAVIS dataset
if [ ! -d "DAVIS" ]; then
    echo "Extracting DAVIS dataset..."
    unzip DAVIS-2017-trainval-480p.zip
    rm DAVIS-2017-trainval-480p.zip
fi

# Optional: Download test set
read -p "Do you want to download DAVIS 2017 test set? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading DAVIS 2017 test set..."
    if [ ! -f "DAVIS-2017-test-dev-480p.zip" ]; then
        wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip -O DAVIS-2017-test-dev-480p.zip
    fi
    
    if [ -f "DAVIS-2017-test-dev-480p.zip" ]; then
        echo "Extracting test set..."
        unzip DAVIS-2017-test-dev-480p.zip
        rm DAVIS-2017-test-dev-480p.zip
    fi
fi

# Verify dataset structure
echo "Verifying DAVIS dataset structure..."
if [ -d "DAVIS/JPEGImages/480p" ] && [ -d "DAVIS/Annotations/480p" ]; then
    echo "✓ DAVIS dataset setup completed successfully!"
    
    # Count sequences and frames
    SEQUENCES=$(find DAVIS/JPEGImages/480p -type d -mindepth 1 | wc -l)
    FRAMES=$(find DAVIS/JPEGImages/480p -name "*.jpg" | wc -l)
    MASKS=$(find DAVIS/Annotations/480p -name "*.png" | wc -l)
    
    echo "✓ Sequences: $SEQUENCES"
    echo "✓ Frames: $FRAMES"
    echo "✓ Masks: $MASKS"
    
    # List some sequences
    echo "✓ Sample sequences:"
    find DAVIS/JPEGImages/480p -type d -mindepth 1 | head -5 | while read seq; do
        seq_name=$(basename "$seq")
        frame_count=$(find "$seq" -name "*.jpg" | wc -l)
        echo "  - $seq_name: $frame_count frames"
    done
    
else
    echo "✗ DAVIS dataset setup failed!"
    echo "Expected directories:"
    echo "  - DAVIS/JPEGImages/480p"
    echo "  - DAVIS/Annotations/480p"
    exit 1
fi

cd ../..
echo "DAVIS setup complete. Dataset located at: datasets/DAVIS/"

# Create ImageSets directory for easier access
mkdir -p datasets/DAVIS/DAVIS/ImageSets/2017
echo "Creating train/val splits..."

# Create train.txt và val.txt files
find datasets/DAVIS/DAVIS/JPEGImages/480p -type d -mindepth 1 | while read seq_dir; do
    seq_name=$(basename "$seq_dir")
    echo "$seq_name" >> datasets/DAVIS/DAVIS/ImageSets/2017/all_sequences.txt
done

# Split into train (80%) and val (20%)
total_sequences=$(wc -l < datasets/DAVIS/DAVIS/ImageSets/2017/all_sequences.txt)
train_count=$((total_sequences * 80 / 100))

head -n $train_count datasets/DAVIS/DAVIS/ImageSets/2017/all_sequences.txt > datasets/DAVIS/DAVIS/ImageSets/2017/train.txt
tail -n +$((train_count + 1)) datasets/DAVIS/DAVIS/ImageSets/2017/all_sequences.txt > datasets/DAVIS/DAVIS/ImageSets/2017/val.txt

echo "✓ Created train/val splits:"
echo "  - Train sequences: $(wc -l < datasets/DAVIS/DAVIS/ImageSets/2017/train.txt)"
echo "  - Val sequences: $(wc -l < datasets/DAVIS/DAVIS/ImageSets/2017/val.txt)" 