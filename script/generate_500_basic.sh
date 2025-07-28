#!/bin/bash

# HLS Model Checking Benchmark Generator - 500 Examples Script
# This script generates 500 examples using different seeds
# Script should be launched from the project root folder

set -e  # Exit on any error

# Get current timestamp for output folder
current_time=$(date +"%Y%m%d_%H%M%S")
output_base_dir="output_${current_time}"

# Create the main output directory
echo "[INFO] Creating output directory: ${output_base_dir}"
mkdir -p "${output_base_dir}"

# Check if we're in the correct directory (should have src/main.py)
if [ ! -f "src/main.py" ]; then
    echo "[ERROR] This script must be run from the project root directory (where src/main.py exists)"
    exit 1
fi

# Configuration
total_examples=500
start_seed=1
max_parallel_jobs=10  # Adjust based on your system capabilities

# Function to generate a single example
generate_example() {
    local seed=$1
    local example_dir="${output_base_dir}/example_${seed}"
    
    echo "[INFO] Generating example ${seed}/${total_examples} with seed ${seed}..."
    
    # Run main.py with specific seed and output directory
    python3 src/main.py \
        --seed "${seed}" \
        --output-dir "${example_dir}" \
        --cpp-file "benchmark.cpp" \
        --project-name "hls_benchmark" \
        --top-function "top" \
        --clock-period 10 \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Example ${seed} generated successfully"
    else
        echo "[ERROR] Example ${seed} failed to generate"
        return 1
    fi
}

# Export the function so it can be used by parallel processes
export -f generate_example
export output_base_dir
export total_examples

# Start time tracking
start_time=$(date +%s)
echo "[INFO] Starting generation of ${total_examples} examples at $(date)"
echo "[INFO] Output directory: ${output_base_dir}"
echo "[INFO] Using maximum ${max_parallel_jobs} parallel jobs"

# Create a list of seeds and use GNU parallel or xargs to run them
seq ${start_seed} $((start_seed + total_examples - 1)) | \
    xargs -n 1 -P ${max_parallel_jobs} -I {} bash -c 'generate_example {}'

# Check if all examples were generated successfully
failed_count=0
success_count=0

echo ""
echo "[INFO] Checking generation results..."

for seed in $(seq ${start_seed} $((start_seed + total_examples - 1))); do
    example_dir="${output_base_dir}/example_${seed}"
    
    if [ -d "${example_dir}" ] && [ -f "${example_dir}/benchmark_1.cpp" ] && [ -f "${example_dir}/benchmark_2.cpp" ]; then
        success_count=$((success_count + 1))
    else
        failed_count=$((failed_count + 1))
        echo "[WARNING] Example ${seed} appears to be incomplete or failed"
    fi
done

# End time tracking
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

# Final summary
echo ""
echo "=========================================="
echo "           GENERATION SUMMARY"
echo "=========================================="
echo "Total examples requested: ${total_examples}"
echo "Successfully generated: ${success_count}"
echo "Failed: ${failed_count}"
echo "Output directory: ${output_base_dir}"
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo ""

if [ ${failed_count} -eq 0 ]; then
    echo "[SUCCESS] All ${total_examples} examples generated successfully!"
    
    # Create a summary file
    summary_file="${output_base_dir}/generation_summary.txt"
    {
        echo "HLS Model Checking Benchmark Generation Summary"
        echo "Generated on: $(date)"
        echo "Total examples: ${total_examples}"
        echo "Seeds used: ${start_seed} to $((start_seed + total_examples - 1))"
        echo "Success count: ${success_count}"
        echo "Failed count: ${failed_count}"
        echo "Total time: ${hours}h ${minutes}m ${seconds}s"
        echo ""
        echo "Directory structure:"
        ls -la "${output_base_dir}" | head -20
        if [ $(ls -1 "${output_base_dir}" | grep "example_" | wc -l) -gt 15 ]; then
            echo "... and $((success_count - 15)) more examples"
        fi
    } > "${summary_file}"
    
    echo "Summary written to: ${summary_file}"
    
else
    echo "[WARNING] ${failed_count} examples failed to generate properly"
    echo "Check the output directory for details: ${output_base_dir}"
    exit 1
fi

echo ""
echo "Generation completed. Results are in: ${output_base_dir}"

