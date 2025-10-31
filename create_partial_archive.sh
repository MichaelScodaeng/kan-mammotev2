#!/bin/bash

# Script to create a partial archive with only kan_mammote_dual_kmote related files
# Keeps only 2 checkpoints per folder for faster download

set -e

ARCHIVE_NAME="/home/s2516027/kan-mammotev2/test_kmmv2_partial.tar.zst"
BASE_DIR="/home/s2516027/kan-mammotev2"
TEMP_DIR="/tmp/kmmv2_partial_$(date +%s)"

echo "🗜️  Creating partial KAN-MAMMOTE archive..."
echo "📁 Base directory: $BASE_DIR"
echo "📦 Archive: $ARCHIVE_NAME"
echo "🔧 Temp directory: $TEMP_DIR"

# Create temporary directory structure
mkdir -p "$TEMP_DIR"
cd "$BASE_DIR"

echo ""
echo "🔍 Searching for kan_mammote_dual_kmote related files..."

# Function to copy files with limited checkpoints
copy_with_limit() {
    local src_pattern="$1"
    local dest_dir="$2"
    local max_files="$3"
    
    # Find all matching directories/files
    find . -path "$src_pattern" -type d 2>/dev/null | while read -r dir; do
        if [ -d "$dir" ]; then
            # Create destination directory
            mkdir -p "$TEMP_DIR/$dir"
            
            # Copy non-checkpoint files first
            find "$dir" -maxdepth 1 -type f ! -name "*.pt" ! -name "*.pth" ! -name "*checkpoint*" 2>/dev/null | while read -r file; do
                if [ -f "$file" ]; then
                    cp "$file" "$TEMP_DIR/$file" 2>/dev/null || true
                fi
            done
            
            # Copy limited number of checkpoint files (keep the most recent ones)
            find "$dir" -maxdepth 1 -type f \( -name "*.pt" -o -name "*.pth" -o -name "*checkpoint*" \) 2>/dev/null | \
                sort -t_ -k3 -nr | head -n "$max_files" | while read -r checkpoint; do
                if [ -f "$checkpoint" ]; then
                    echo "  📄 Including: $checkpoint"
                    cp "$checkpoint" "$TEMP_DIR/$checkpoint" 2>/dev/null || true
                fi
            done
        fi
    done
}

# Function to copy individual files
copy_files() {
    local pattern="$1"
    find . -path "$pattern" -type f 2>/dev/null | while read -r file; do
        if [ -f "$file" ]; then
            # Create destination directory
            mkdir -p "$TEMP_DIR/$(dirname "$file")"
            echo "  📄 Including: $file"
            cp "$file" "$TEMP_DIR/$file" 2>/dev/null || true
        fi
    done
}

echo ""
echo "📂 Processing saved_models..."
# Copy saved_models with kan_mammote_dual_kmote pattern (limit to 2 checkpoints per folder)
copy_with_limit "./saved_models/*/*kan_mammote_dual_kmote*" "$TEMP_DIR" 2

echo ""
echo "📊 Processing saved_results..."
# Copy saved_results with kan_mammote_dual_kmote pattern
copy_files "./saved_results/*/*kan_mammote_dual_kmote*"

echo ""
echo "📋 Summary of included files:"
if [ -d "$TEMP_DIR" ]; then
    echo "📁 Directory structure:"
    find "$TEMP_DIR" -type f | sort
    
    echo ""
    echo "📊 File counts:"
    echo "  🎯 Total files: $(find "$TEMP_DIR" -type f | wc -l)"
    echo "  💾 Checkpoint files: $(find "$TEMP_DIR" -type f \( -name "*.pt" -o -name "*.pth" -o -name "*checkpoint*" \) | wc -l)"
    echo "  📄 Other files: $(find "$TEMP_DIR" -type f ! \( -name "*.pt" -o -name "*.pth" -o -name "*checkpoint*" \) | wc -l)"
    
    echo ""
    echo "💽 Estimated size:"
    du -sh "$TEMP_DIR" | cut -f1
    
    echo ""
    echo "🗜️  Creating compressed archive..."
    cd "$TEMP_DIR"
    
    # Create the compressed archive
    tar -I "zstd -T0 -9" -cvf "$ARCHIVE_NAME" .
    
    echo ""
    echo "✅ Archive created successfully!"
    echo "📦 Archive: $ARCHIVE_NAME"
    echo "📏 Archive size: $(du -sh "$ARCHIVE_NAME" | cut -f1)"
    
    # Cleanup
    echo ""
    echo "🧹 Cleaning up temporary directory..."
    rm -rf "$TEMP_DIR"
    
    echo ""
    echo "🎉 Done! Archive ready for download:"
    echo "   $ARCHIVE_NAME"
    
else
    echo "❌ No matching files found!"
    rm -rf "$TEMP_DIR"
    exit 1
fi