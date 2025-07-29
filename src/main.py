from random_graph_manager import RandomGraphManager
from vitis_hls_compiler import VitisHLSCompiler
from miter_generator import MiterGenerator
from yosys_compiler import YosysCompiler
from kairos_pre_processor import KairosPreprocessor
import os
import sys
import argparse
import random


def main():
    """
    Main function to generate random HLS benchmark graphs, dump C++ files, and compile with Vitis HLS.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HLS Model Checking Benchmark Generator')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for graph generation (default: 42)')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory for generated files (default: ./output)')
    parser.add_argument('--cpp-file', type=str, default='benchmark.cpp', help='Name of the C++ output file (default: benchmark.cpp)')
    parser.add_argument('--project-name', type=str, default='hls_benchmark', help='HLS project name (default: hls_benchmark)')
    parser.add_argument('--top-function', type=str, default='top', help='Top-level function name (default: top)')
    parser.add_argument('--skip-compilation', action='store_true', help='Skip Vitis HLS compilation step')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--action-count', '-a', type=int, default=20, help='Max number of actions')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        if args.verbose:
            print(f"[INFO] Created output directory: {args.output_dir}")
    
    # Set up file paths for comparison files
    cpp_file_1_path = os.path.join(args.output_dir, f"benchmark_1.cpp")
    cpp_file_2_path = os.path.join(args.output_dir, f"benchmark_2.cpp")
    
    try:
        # Step 1: Generate random graph
        print(f"[INFO] Generating random graph with seed {args.seed}...")
        action_count = args.action_count
        graph_manager = RandomGraphManager(seed=args.seed)
        
        
        success = graph_manager.generate_random_graph(action_number_total=action_count)
        if not success:
            print("[ERROR] Failed to generate random graph")
            return 1
        
        if args.verbose:
            # Print graph statistics
            graph = graph_manager.program_graph
            op_nodes = graph_manager._get_op_node_list()
            array_nodes = graph_manager._get_array_node_list()
            
            print(f"[INFO] Graph generation completed:")
            print(f"  Total nodes: {graph.number_of_nodes()}")
            print(f"  Total edges: {graph.number_of_edges()}")
            print(f"  Operation nodes: {len(op_nodes)}")
            print(f"  Array nodes: {len(array_nodes)}")
        
        # Step 2: Dump C++ comparison files
        print(f"[INFO] Dumping C++ comparison files...")
        graph_manager.dump_cpp_comparsion(cpp_file_1_path, cpp_file_2_path)
        
        # Validate both C++ files were created
        files_to_check = [cpp_file_1_path, cpp_file_2_path]
        cpp_files_created = []
        
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                print(f"[ERROR] Failed to create C++ file: {file_path}")
                return 1
            
            # Validate file content
            with open(file_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    print(f"[ERROR] Generated C++ file is empty: {file_path}")
                    return 1
            
            cpp_files_created.append(file_path)
            print(f"[INFO] C++ file generated successfully: {file_path} ({len(content)} characters)")
        
        print(f"[INFO] Generated {len(cpp_files_created)} comparison C++ files")
        
        if args.skip_compilation:
            print("[INFO] Skipping Vitis HLS compilation as requested")
            print(f"[INFO] Generated files:")
            for cpp_file in cpp_files_created:
                print(f"  C++ source: {cpp_file}")
            return 0
        
        # Step 3: Compile with Vitis HLS (compile each file separately)
        print(f"[INFO] Starting Vitis HLS compilation for {len(cpp_files_created)} files...")
        
        all_compile_results = []
        compilation_success = True
        
        for i, cpp_file in enumerate(cpp_files_created, 1):

            if i > 2 or i <= 0:
                raise ValueError(f"[ERROR] unexpected i = {i}, current cpp file = {cpp_file}")

            print(f"[INFO] Compiling file {i}/{len(cpp_files_created)}: {os.path.basename(cpp_file)}")
            
            # Create separate working directory for each compilation
            compile_dir = os.path.join(args.output_dir, f"compile_{i}")
            if not os.path.exists(compile_dir):
                os.makedirs(compile_dir)
            
            # Create separate HLS compiler instance for this file
            hls_compiler = VitisHLSCompiler(
                working_dir=compile_dir,
                hls_script_path=os.path.join(compile_dir, f"hls_script_{i}.tcl"),
                log_file_path=os.path.join(compile_dir, f"hls_compile_{i}.log")
            )
            
            try:

                if i == 1:
                    cp = graph_manager.cp_1
                elif i == 2:
                    cp = graph_manager.cp_2
                else:
                    raise ValueError()

                # Compile this specific C++ file
                compile_result = hls_compiler.compile(
                    project_name=f"{args.project_name}_{i}",
                    top_name=args.top_function,
                    clock_period=cp,
                    cpp_file_list=[cpp_file]
                )
                
                compile_result["cpp_file"] = cpp_file
                compile_result["compile_index"] = i
                all_compile_results.append(compile_result)
                
                if compile_result["success"]:
                    print(f"[INFO] File {i} compilation completed successfully!")
                    if args.verbose:
                        print(f"  Project: {compile_result['project_path']}")
                        print(f"  Log: {compile_result['log_file']}")
                        if compile_result["verilog_files"]:
                            print(f"  HDL files: {len(compile_result['verilog_files'])}")
                else:
                    print(f"[ERROR] File {i} compilation failed")
                    compilation_success = False
                    if os.path.exists(compile_result["log_file"]):
                        print(f"[ERROR] Check log file: {compile_result['log_file']}")
                        
            except Exception as e:
                print(f"[ERROR] Exception during compilation of file {i}: {str(e)}")
                compilation_success = False
                continue
        
        # Print summary of all compilations
        if compilation_success:
            print("[INFO] All Vitis HLS compilations completed successfully!")
            print(f"[INFO] Summary of generated files:")
            for result in all_compile_results:
                print(f"  File {result['compile_index']}: {os.path.basename(result['cpp_file'])}")
                print(f"    Project: {result['project_path']}")
                print(f"    Log: {result['log_file']}")
                if result["verilog_files"]:
                    print(f"    HDL files ({len(result['verilog_files'])}):")
                    for verilog_file in result["verilog_files"]:
                        print(f"      {verilog_file}")
                else:
                    print(f"    No HDL files found")
                print()
        else:
            print("[ERROR] One or more Vitis HLS compilations failed")
            return 1
        
        # Step 4: Generate miter using the compiled Verilog files
        print("[INFO] Starting miter generation...")
        try:
            # Ensure we have exactly 2 compilation results for miter generation
            if len(all_compile_results) != 2:
                print(f"[WARNING] Miter generation requires exactly 2 compiled files, but got {len(all_compile_results)}. Skipping miter generation.")
            else:
                result_1 = all_compile_results[0]
                result_2 = all_compile_results[1]
                
                # Check if both compilations have verilog files
                if not result_1["verilog_files"] or not result_2["verilog_files"]:
                    print("[WARNING] One or both compilations have no Verilog files. Skipping miter generation.")
                else:
                    # Create miter output directory
                    miter_output_dir = os.path.join(args.output_dir, "miter")
                    if not os.path.exists(miter_output_dir):
                        os.makedirs(miter_output_dir)
                        if args.verbose:
                            print(f"[INFO] Created miter output directory: {miter_output_dir}")
                    
                    # Validate that all Verilog files exist
                    verilog_files_1 = result_1["verilog_files"]
                    verilog_files_2 = result_2["verilog_files"]
                    
                    print(f"[INFO] Creating miter from {len(verilog_files_1)} and {len(verilog_files_2)} Verilog files...")
                    
                    if args.verbose:
                        print(f"[INFO] Verilog files from compilation 1:")
                        for vf in verilog_files_1:
                            print(f"  {vf}")
                        print(f"[INFO] Verilog files from compilation 2:")
                        for vf in verilog_files_2:
                            print(f"  {vf}")
                    
                    # Initialize MiterGenerator
                    miter_generator = MiterGenerator(
                        verilog_file_path_list_1=verilog_files_1,
                        verilog_file_path_list_2=verilog_files_2,
                        merged_verilog_folder_path=miter_output_dir,
                        top_name=args.top_function
                    )
                    
                    # Generate the miter
                    kairos_top = miter_generator.generate_miter(insert_assertions=False)
                    
                    print(f"[INFO] Miter generation completed successfully!")
                    print(f"  Merged file 1: {miter_generator.merged_verilog_file_path_1}")
                    print(f"  Merged file 2: {miter_generator.merged_verilog_file_path_2}")
                    print(f"  Miter file: {miter_generator.miter_verilog_file_path}")
                    if kairos_top:
                        print(f"  Kairos top module: {kairos_top}")
                    
                    # Step 5: Compile miter to AIGER using YosysCompiler
                    print("[INFO] Starting Yosys compilation to AIGER...")
                    try:
                        yosys_compiler = YosysCompiler()
                        
                        # Set up AIGER output file path
                        aiger_output_path = os.path.join(miter_output_dir, "miter.aig")
                        
                        # Get the miter file path and top module name
                        miter_file_path = miter_generator.miter_verilog_file_path
                        
                        # Use the kairos_top if available, otherwise default to "top"
                        top_module_name = kairos_top if kairos_top else "top"
                        
                        if args.verbose:
                            print(f"[INFO] Compiling miter file: {miter_file_path}")
                            print(f"[INFO] Top module: {top_module_name}")
                            print(f"[INFO] Output AIGER file: {aiger_output_path}")
                        
                        # Execute Yosys compilation
                        yosys_compiler.execute(
                            verilog_file_path=miter_file_path,
                            working_dir=miter_output_dir,
                            aiger_file_path=aiger_output_path,
                            top_name=top_module_name
                        )
                        
                        # Verify AIGER file was created
                        if os.path.exists(aiger_output_path):
                            file_size = os.path.getsize(aiger_output_path)
                            print(f"[INFO] AIGER file generated successfully: {aiger_output_path} ({file_size} bytes)")
                        else:
                            print(f"[ERROR] AIGER file was not created: {aiger_output_path}")
                            
                    except Exception as yosys_e:
                        print(f"[ERROR] Yosys compilation to AIGER failed: {str(yosys_e)}")
                        raise yosys_e
                        if args.verbose:
                            import traceback
                            traceback.print_exc()
                        print("[WARNING] Continuing despite Yosys compilation failure...")
                    
        except Exception as e:

            print(f"[ERROR] Miter generation failed: {str(e)}")
            raise e
            if args.verbose:
                import traceback
                traceback.print_exc()
            # Don't return error here, as the main compilation was successful
            print("[WARNING] Continuing despite miter generation failure...")
            
    except Exception as e:
        print(f"[ERROR] An error occurred: {str(e)}")
        raise e
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    print("[INFO] Benchmark generation and compilation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())


