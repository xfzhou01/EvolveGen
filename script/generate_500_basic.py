#!/usr/bin/env python3

"""
HLS Model Checking Benchmark Generator - 500 Examples Script
This script generates 500 examples using different seeds
Script should be launched from the project root folder
"""

import os
import sys
import subprocess
import time
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def check_project_root():
    """Check if we're in the correct directory (should have src/main.py)"""
    if not Path("src/main.py").exists():
        print("[ERROR] This script must be run from the project root directory (where src/main.py exists)")
        sys.exit(1)


def generate_example(seed, output_base_dir, total_examples):
    """Generate a single example with the given seed"""
    example_dir = Path(output_base_dir) / f"example_{seed}"
    
    print(f"[INFO] Generating example {seed}/{total_examples} with seed {seed}...")
    
    try:
        # Run main.py with specific seed and output directory
        cmd = [
            "python3", "src/main.py",
            "--seed", str(seed),
            "--output-dir", str(example_dir),
            "--cpp-file", "benchmark.cpp",
            "--project-name", "hls_benchmark",
            "--top-function", "top",
            "--action-count", "30",
            "--verbose"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[SUCCESS] Example {seed} generated successfully")
        return seed, True, None
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed with return code {e.returncode}"
        if e.stderr:
            error_msg += f": {e.stderr.strip()}"
        print(f"[ERROR] Example {seed} failed to generate: {error_msg}")
        return seed, False, error_msg
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Example {seed} failed to generate: {error_msg}")
        return seed, False, error_msg


def check_example_completeness(seed, output_base_dir):
    """Check if an example was generated successfully"""
    example_dir = Path(output_base_dir) / f"example_{seed}"
    
    # Check if directory exists and contains expected files
    if not example_dir.exists():
        return False
    
    benchmark1 = example_dir / "benchmark_1.cpp"
    benchmark2 = example_dir / "benchmark_2.cpp"
    
    return benchmark1.exists() and benchmark2.exists()


def create_summary_file(output_base_dir, total_examples, start_seed, success_count, failed_count, duration):
    """Create a summary file with generation statistics"""
    summary_file = Path(output_base_dir) / "generation_summary.txt"
    
    hours = duration // 3600
    minutes = (duration % 3600) // 60
    seconds = duration % 60
    
    with open(summary_file, 'w') as f:
        f.write("HLS Model Checking Benchmark Generation Summary\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total examples: {total_examples}\n")
        f.write(f"Seeds used: {start_seed} to {start_seed + total_examples - 1}\n")
        f.write(f"Success count: {success_count}\n")
        f.write(f"Failed count: {failed_count}\n")
        f.write(f"Total time: {hours:02d}h {minutes:02d}m {seconds:02d}s\n")
        f.write("\n")
        f.write("Directory structure:\n")
        
        # List first 20 entries in the output directory
        try:
            entries = sorted(os.listdir(output_base_dir))
            for i, entry in enumerate(entries[:20]):
                entry_path = Path(output_base_dir) / entry
                if entry_path.is_dir():
                    f.write(f"drwxr-xr-x  {entry}/\n")
                else:
                    f.write(f"-rw-r--r--  {entry}\n")
            
            if len(entries) > 20:
                f.write(f"... and {len(entries) - 20} more entries\n")
                
        except OSError as e:
            f.write(f"Error listing directory: {e}\n")
    
    return summary_file


def main():
    parser = argparse.ArgumentParser(description="Generate 500 HLS benchmark examples")
    parser.add_argument("--total-examples", type=int, default=500,
                        help="Total number of examples to generate (default: 500)")
    parser.add_argument("--start-seed", type=int, default=1,
                        help="Starting seed value (default: 1)")
    parser.add_argument("--max-parallel-jobs", type=int, default=10,
                        help="Maximum number of parallel jobs (default: 10)")
    
    args = parser.parse_args()
    
    # Check if we're in the correct directory
    check_project_root()
    
    # Configuration
    total_examples = args.total_examples
    start_seed = args.start_seed
    max_parallel_jobs = args.max_parallel_jobs
    
    # Get current timestamp for output folder
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = f"output_{current_time}"
    
    # Create the main output directory
    print(f"[INFO] Creating output directory: {output_base_dir}")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Start time tracking
    start_time = time.time()
    print(f"[INFO] Starting generation of {total_examples} examples at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Output directory: {output_base_dir}")
    print(f"[INFO] Using maximum {max_parallel_jobs} parallel jobs")
    
    # Generate all examples in parallel
    seeds = list(range(start_seed, start_seed + total_examples))
    completed_tasks = []
    
    with ProcessPoolExecutor(max_workers=max_parallel_jobs) as executor:
        # Submit all tasks
        future_to_seed = {
            executor.submit(generate_example, seed, output_base_dir, total_examples): seed 
            for seed in seeds
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_seed):
            seed, success, error = future.result()
            completed_tasks.append((seed, success, error))
    
    # Check generation results
    print("\n[INFO] Checking generation results...")
    
    failed_count = 0
    success_count = 0
    failed_examples = []
    
    for seed in seeds:
        if check_example_completeness(seed, output_base_dir):
            success_count += 1
        else:
            failed_count += 1
            failed_examples.append(seed)
            print(f"[WARNING] Example {seed} appears to be incomplete or failed")
    
    # End time tracking
    end_time = time.time()
    duration = int(end_time - start_time)
    hours = duration // 3600
    minutes = (duration % 3600) // 60
    seconds = duration % 60
    
    # Final summary
    print("\n" + "=" * 42)
    print("           GENERATION SUMMARY")
    print("=" * 42)
    print(f"Total examples requested: {total_examples}")
    print(f"Successfully generated: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Output directory: {output_base_dir}")
    print(f"Total time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
    print()
    
    if failed_count == 0:
        print(f"[SUCCESS] All {total_examples} examples generated successfully!")
        
        # Create a summary file
        summary_file = create_summary_file(output_base_dir, total_examples, start_seed, 
                                         success_count, failed_count, duration)
        print(f"Summary written to: {summary_file}")
        
    else:
        print(f"[WARNING] {failed_count} examples failed to generate properly")
        if failed_examples:
            print(f"Failed examples (seeds): {', '.join(map(str, failed_examples[:10]))}")
            if len(failed_examples) > 10:
                print(f"... and {len(failed_examples) - 10} more")
        print(f"Check the output directory for details: {output_base_dir}")
        sys.exit(1)
    
    print(f"\nGeneration completed. Results are in: {output_base_dir}")


if __name__ == "__main__":
    main()
