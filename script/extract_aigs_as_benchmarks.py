#!/usr/bin/env python3

"""
Script to extract all AIG files from output directories and rename to avoid duplicates
Usage: python extract_aigs_as_benchmarks.py <output_directory>
Example: python extract_aigs_as_benchmarks.py /home/x/xiaofeng-zhou/hls_model_checking_benchmark_generator/output_20250727_144414
"""

import os
import sys
import shutil
import glob
import re
from pathlib import Path


def main():
    # Check if argument is provided
    if len(sys.argv) != 2:
        print("Usage: python extract_aigs_as_benchmarks.py <output_directory>")
        print("Example: python extract_aigs_as_benchmarks.py /home/x/xiaofeng-zhou/hls_model_checking_benchmark_generator/output_20250727_144414")
        sys.exit(1)
    
    # Source directory from command line argument
    source_dir = Path(sys.argv[1])
    
    # Check if source directory exists
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Error: Directory '{source_dir}' does not exist")
        sys.exit(1)
    
    # Base directory for creating benchmarks folder
    base_dir = source_dir.parent
    
    # Create benchmarks directory if it doesn't exist
    benchmarks_dir = base_dir / "benchmarks"
    benchmarks_dir.mkdir(exist_ok=True)
    
    # Counter for naming files to avoid duplicates
    counter = 1
    
    print(f"Starting AIG extraction from: {source_dir}")
    print(f"Target directory: {benchmarks_dir}")
    
    # Get the output directory name for naming convention
    output_name = source_dir.name
    output_timestamp = re.sub(r'^output_', '', output_name)
    
    # Find all example_* subdirectories in the specified source directory
    example_dirs = glob.glob(str(source_dir / "example_*"))
    example_dirs.sort()  # Sort for consistent ordering
    
    for example_dir_path in example_dirs:
        example_dir = Path(example_dir_path)
        
        if example_dir.is_dir():
            example_name = example_dir.name
            miter_dir = example_dir / "miter"
            miter_file = miter_dir / "miter.aig"
            
            # Check if miter directory exists and contains miter.aig
            if miter_dir.is_dir() and miter_file.exists():
                # Create unique filename: benchmark_timestamp_example_number.aig
                example_number = re.sub(r'^example_', '', example_name)
                
                new_filename = f"benchmark_{output_timestamp}_{example_number}.aig"
                target_path = benchmarks_dir / new_filename
                
                # Copy the AIG file with new name
                shutil.copy2(miter_file, target_path)
                
                print(f"  Extracted: {example_name}/miter/miter.aig -> {new_filename}")
                counter += 1
            else:
                print(f"  Warning: No miter.aig found in {example_name}/miter/")
    
    print()
    print("Extraction completed!")
    print(f"Total AIG files extracted: {counter - 1}")
    print(f"Files saved to: {benchmarks_dir}")
    
    # Display summary
    if counter - 1 > 0:
        print()
        print("Summary of extracted files:")
        
        # Count total files
        aig_files = list(benchmarks_dir.glob("*.aig"))
        print(f"Total files: {len(aig_files)}")
        
        # Show first few files
        print("First few files:")
        for file_path in sorted(aig_files)[:5]:
            print(f"  {file_path.name}")
        
        print()
        print("File sizes:")
        for file_path in sorted(aig_files)[:5]:
            file_size = file_path.stat().st_size
            # Convert to human readable format
            if file_size < 1024:
                size_str = f"{file_size}B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f}K"
            elif file_size < 1024 * 1024 * 1024:
                size_str = f"{file_size / (1024 * 1024):.1f}M"
            else:
                size_str = f"{file_size / (1024 * 1024 * 1024):.1f}G"
            
            print(f"  {size_str}\t{file_path.name}")
    
    print("Done!")


if __name__ == "__main__":
    main()
