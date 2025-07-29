#!/usr/bin/env python3

"""
Script to convert miter.v files to BTOR2 format using Yosys
Usage: python miter_to_btor.py <input_folder> [output_folder]

This script:
- Reads all miter.v files in the specified folder
- Uses Yosys to convert Verilog to BTOR2 format
- Stores output in output_btor2 by default
"""

import os
import sys
import subprocess
import tempfile
import argparse
from pathlib import Path
import re


def display_usage():
    """Display usage information"""
    print("Usage: python miter_to_btor.py <input_folder> [output_folder]")
    print("  input_folder:  Directory containing miter.v files")
    print("  output_folder: Output directory (default: output_btor2)")
    print("")
    print("Example: python miter_to_btor.py ./output output_btor2")


def find_top_module(verilog_file):
    """Find the top module name in a Verilog file"""
    try:
        with open(verilog_file, 'r') as f:
            content = f.read()
            # Find module declarations
            module_pattern = r'^module\s+(\w+)'
            matches = re.findall(module_pattern, content, re.MULTILINE)
            if matches:
                return matches[-1]  # Return the last module found
    except Exception as e:
        print(f"  Warning: Could not read file {verilog_file}: {e}")
    return None


def create_yosys_script(verilog_file, output_file, top_module="top_A_times_top_B"):
    """Create a temporary Yosys script for conversion"""
    script_content = f"""# Yosys script to convert {verilog_file} to BTOR2
read -sv {verilog_file}
prep -top {top_module}
flatten
memory -nomap
hierarchy -check
setundef -undriven -init -expose
write_btor -s {output_file}
"""
    return script_content


def run_yosys(script_content):
    """Run Yosys with the given script content"""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ys', delete=False) as temp_script:
            temp_script.write(script_content)
            temp_script_path = temp_script.name
        
        # Run yosys with the script
        result = subprocess.run(
            ['yosys', '-s', temp_script_path],
            capture_output=True,
            text=True
        )
        
        # Clean up temporary script
        os.unlink(temp_script_path)
        
        return result.returncode == 0, result.stderr
        
    except FileNotFoundError:
        print("Error: Yosys not found. Please make sure Yosys is installed and in your PATH.")
        return False, "Yosys not found"
    except Exception as e:
        return False, str(e)


def convert_verilog_to_btor2(verilog_file, output_file):
    """Convert a single Verilog file to BTOR2 format"""
    print(f"Processing: {verilog_file}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # First attempt with default top module name
    script_content = create_yosys_script(verilog_file, output_file)
    success, error_msg = run_yosys(script_content)
    
    if success:
        print(f"  ✓ Successfully converted to: {output_file}")
        return True
    else:
        print(f"  ✗ Failed to convert: {verilog_file}")
        print("    Trying alternative approach...")
        
        # Try alternative approach - detect top module name
        top_module = find_top_module(verilog_file)
        
        if top_module:
            script_content = create_yosys_script(verilog_file, output_file, top_module)
            success, error_msg = run_yosys(script_content)
            
            if success:
                print(f"  ✓ Successfully converted with top module '{top_module}': {output_file}")
                return True
            else:
                print(f"  ✗ Failed to convert even with detected top module '{top_module}'")
                return False
        else:
            print("  ✗ Could not detect top module name")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Convert miter.v files to BTOR2 format using Yosys",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python miter_to_btor.py ./output output_btor2"
    )
    
    parser.add_argument("input_folder", help="Directory containing miter.v files")
    parser.add_argument("output_folder", nargs='?', default="output_btor2",
                       help="Output directory (default: output_btor2)")
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_folder)
    output_dir = Path(args.output_folder)
    
    # Check if input directory exists
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting miter.v files from '{input_dir}' to BTOR2 format...")
    print(f"Output directory: '{output_dir}'")
    print("")
    
    # Counter for processed files
    processed_count = 0
    error_count = 0
    
    # Find all miter.v files recursively
    verilog_files = list(input_dir.rglob("miter.v"))
    
    if not verilog_files:
        print(f"Warning: No miter.v files found in '{input_dir}'")
        print("Make sure the input directory contains files named 'miter.v'")
        sys.exit(0)
    
    for verilog_file in verilog_files:
        # Get the relative path from input directory
        rel_path = verilog_file.relative_to(input_dir)
        # Create corresponding output path
        output_subdir = output_dir / rel_path.parent
        # Generate output filename (replace .v with .btor2)
        base_name = verilog_file.stem
        output_file = output_subdir / f"{base_name}.btor2"
        
        # Convert the file
        if convert_verilog_to_btor2(str(verilog_file), str(output_file)):
            processed_count += 1
        else:
            error_count += 1
        
        print("")  # Empty line for readability
    
    # Summary
    print("Conversion completed!")
    print(f"Processed files: {processed_count}")
    if error_count > 0:
        print(f"Failed files: {error_count}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
