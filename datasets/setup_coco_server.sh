#!/bin/bash

# COCO 2017 Dataset Setup Script - Server Version with Resume Support
# Usage: ./setup_coco_server.sh

LOG_FILE="coco_setup.log"
DATASET_DIR="datasets/COCO"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a $LOG_FILE
}

# Function to download with resume support
download_file() {
    local url=$1
    local output=$2
    local description=$3
    
    log "Starting download: $description"
    log "URL: $url"
    log "Output: $output"
    
    # Check if file already exists and is complete
    if [ -f "$output" ]; then
        log "File $output already exists, checking if complete..."
        # Try to get file size from server
        remote_size=$(curl -sI "$url" | grep -i content-length | awk '{print $2}' | tr -d '\r')
        local_size=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output" 2>/dev/null)
        
        if [ "$remote_size" = "$local_size" ]; then
            log "✓ File $output is already complete, skipping download"
            return 0
        else
            log "File $output is incomplete (local: $local_size, remote: $remote_size), resuming..."
        fi
    fi
    
    # Download with resume support and progress tracking
    wget -c -T 30 -t 5 --progress=bar:force:noscroll "$url" -O "$output" 2>&1 | tee -a $LOG_FILE
    
    if [ $? -eq 0 ]; then
        log "✓ Successfully downloaded: $description"
        return 0
    else
        log "✗ Failed to download: $description"
        return 1
    fi
}

# Function to extract with verification
extract_file() {
    local zip_file=$1
    local extract_dir=$2
    local description=$3
    
    if [ ! -f "$zip_file" ]; then
        log "✗ Zip file $zip_file not found"
        return 1
    fi
    
    if [ -d "$extract_dir" ]; then
        log "✓ Directory $extract_dir already exists, skipping extraction"
        return 0
    fi
    
    log "Extracting $description..."
    unzip -q "$zip_file"
    
    if [ $? -eq 0 ]; then
        log "✓ Successfully extracted: $description"
        # Remove zip file after successful extraction
        rm "$zip_file"
        log "✓ Removed zip file: $zip_file"
        return 0
    else
        log "✗ Failed to extract: $description"
        return 1
    fi
}

# Main setup function
main() {
    log "=== COCO 2017 Dataset Setup Started ==="
    log "PID: $$"
    log "Working directory: $(pwd)"
    
    # Create COCO directory
    mkdir -p $DATASET_DIR
    cd $DATASET_DIR
    
    log "Changed to directory: $(pwd)"
    
    # Download validation images
    if ! download_file "http://images.cocodataset.org/zips/val2017.zip" "val2017.zip" "COCO 2017 validation images"; then
        log "✗ Failed to download validation images"
        exit 1
    fi
    
    # Extract validation images
    if ! extract_file "val2017.zip" "val2017" "validation images"; then
        log "✗ Failed to extract validation images"
        exit 1
    fi
    
    # Download annotations
    if ! download_file "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "annotations_trainval2017.zip" "COCO 2017 annotations"; then
        log "✗ Failed to download annotations"
        exit 1
    fi
    
    # Extract annotations
    if ! extract_file "annotations_trainval2017.zip" "annotations" "annotations"; then
        log "✗ Failed to extract annotations"
        exit 1
    fi
    
    # Optional: Download train images
    read -t 10 -p "Download COCO train2017 images? (y/N) [timeout 10s]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "User requested to download training images"
        if ! download_file "http://images.cocodataset.org/zips/train2017.zip" "train2017.zip" "COCO 2017 training images"; then
            log "✗ Failed to download training images"
            exit 1
        fi
        
        if ! extract_file "train2017.zip" "train2017" "training images"; then
            log "✗ Failed to extract training images"
            exit 1
        fi
    else
        log "Skipping training images download"
    fi
    
    # Verify dataset structure
    log "Verifying COCO dataset structure..."
    if [ -d "val2017" ] && [ -d "annotations" ]; then
        val_count=$(find val2017 -name "*.jpg" | wc -l)
        ann_count=$(find annotations -name "*.json" | wc -l)
        
        log "✓ COCO dataset setup completed successfully!"
        log "✓ Val images: $val_count files"
        log "✓ Annotations: $ann_count files"
        
        if [ -d "train2017" ]; then
            train_count=$(find train2017 -name "*.jpg" | wc -l)
            log "✓ Train images: $train_count files"
        fi
        
        # Calculate total size
        total_size=$(du -sh . | cut -f1)
        log "✓ Total dataset size: $total_size"
        
    else
        log "✗ COCO dataset setup failed!"
        exit 1
    fi
    
    cd ../..
    log "=== COCO Dataset Setup Completed Successfully ==="
    log "Dataset location: $DATASET_DIR/"
    log "Log file: $DATASET_DIR/$LOG_FILE"
}

# Handle script interruption
cleanup() {
    log "Script interrupted by user"
    exit 1
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@" 