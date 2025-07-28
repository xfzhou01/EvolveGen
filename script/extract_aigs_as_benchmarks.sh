#!/bin/bash

# Script to extract all AIG files from output directories and rename to avoid duplicates
# Usage: ./extract_aigs_as_benchmarks.sh <output_directory>
# Example: ./extract_aigs_as_benchmarks.sh /home/x/xiaofeng-zhou/hls_model_checking_benchmark_generator/output_20250727_144414

set -e  # Exit on any error

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <output_directory>"
    echo "Example: $0 /home/x/xiaofeng-zhou/hls_model_checking_benchmark_generator/output_20250727_144414"
    exit 1
fi

# Source directory from command line argument
SOURCE_DIR="$1"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Directory '$SOURCE_DIR' does not exist"
    exit 1
fi

# Base directory for creating benchmarks folder
BASE_DIR=$(dirname "$SOURCE_DIR")

# Create benchmarks directory if it doesn't exist
BENCHMARKS_DIR="$BASE_DIR/benchmarks"
mkdir -p "$BENCHMARKS_DIR"

# Counter for naming files to avoid duplicates
counter=1

echo "Starting AIG extraction from: $SOURCE_DIR"
echo "Target directory: $BENCHMARKS_DIR"

# Get the output directory name for naming convention
output_name=$(basename "$SOURCE_DIR")
output_timestamp=$(echo "$output_name" | sed 's/output_//')

# Find all example_* subdirectories in the specified source directory
for example_dir in "$SOURCE_DIR"/example_*; do
    if [ -d "$example_dir" ]; then
        example_name=$(basename "$example_dir")
        miter_dir="$example_dir/miter"
        
        # Check if miter directory exists and contains miter.aig
        if [ -d "$miter_dir" ] && [ -f "$miter_dir/miter.aig" ]; then
            # Create unique filename: benchmark_timestamp_example_number.aig
            example_number=$(echo "$example_name" | sed 's/example_//')
            
            new_filename="benchmark_${output_timestamp}_${example_number}.aig"
            target_path="$BENCHMARKS_DIR/$new_filename"
            
            # Copy the AIG file with new name
            cp "$miter_dir/miter.aig" "$target_path"
            
            echo "  Extracted: $example_name/miter/miter.aig -> $new_filename"
            ((counter++))
        else
            echo "  Warning: No miter.aig found in $example_name/miter/"
        fi
    fi
done

echo ""
echo "Extraction completed!"
echo "Total AIG files extracted: $((counter - 1))"
echo "Files saved to: $BENCHMARKS_DIR"

# Display summary
if [ $((counter - 1)) -gt 0 ]; then
    echo ""
    echo "Summary of extracted files:"
    ls -la "$BENCHMARKS_DIR"/*.aig | wc -l | xargs echo "Total files:"
    echo "First few files:"
    ls "$BENCHMARKS_DIR"/*.aig | head -5
    echo ""
    echo "File sizes:"
    du -h "$BENCHMARKS_DIR"/*.aig | head -5
fi

echo "Done!"
