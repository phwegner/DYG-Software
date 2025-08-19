#!/bin/bash

# Set variables
MODEL_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt"
TARGET_DIR="../model_files"
TARGET_FILE="$TARGET_DIR/yolo11x-pose.pt"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Ensure target directory exists
log "Checking if target directory exists: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

# Download the file if not already present
if [ -f "$TARGET_FILE" ]; then
    log "Model already exists at $TARGET_FILE. Skipping download."
else
    log "Downloading model from $MODEL_URL ..."
    if curl -L -o "$TARGET_FILE" "$MODEL_URL"; then
        log "Download completed successfully: $TARGET_FILE"
    else
        log "Download failed!"
        exit 1
    fi
fi
