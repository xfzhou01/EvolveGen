import os
import sys
import shutil
import datetime
import pickle
from contextlib import contextmanager
from io import StringIO


class BanditFuzzUtils:
    """Utility functions for HLS BanditFuzz operations"""
    
    def __init__(self, verbose=False, output_dir="./output"):
        self.verbose = verbose
        self.output_dir = output_dir
        self.error_dump_dir = os.path.join(output_dir, "error_dumps")
        self.timeout_cases_dir = os.path.join(output_dir, "timeout_cases")
        self.btor2_output_dir = os.path.join(output_dir, "btor2")
        
        # Create necessary directories
        for dir_path in [self.error_dump_dir, self.timeout_cases_dir]:
            os.makedirs(dir_path, exist_ok=True)

    @contextmanager
    def suppress_output(self):
        """Context manager to suppress stdout when not in verbose mode"""
        if self.verbose:
            yield
        else:
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                yield
            finally:
                sys.stdout = old_stdout

    def log_debug(self, message):
        """Log debug message only in verbose mode"""
        if self.verbose:
            print(f"[DEBUG] {message}")

    def dump_error_state(self, step_name, error_msg, graph=None):
        """Dump error state files for debugging"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            error_folder = os.path.join(self.error_dump_dir, f"{step_name}_{timestamp}")
            os.makedirs(error_folder, exist_ok=True)
            
            # Save error information
            error_info_file = os.path.join(error_folder, "error_info.txt")
            with open(error_info_file, 'w') as f:
                f.write(f"Error Step: {step_name}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Error Message: {error_msg}\n")
                f.write(f"Output Directory: {self.output_dir}\n")
                if graph:
                    f.write(f"Graph Nodes: {graph.number_of_nodes()}\n")
                    f.write(f"Graph Edges: {graph.number_of_edges()}\n")
            
            # Copy current output directory contents
            if os.path.exists(self.output_dir):
                for item in os.listdir(self.output_dir):
                    if item != "error_dumps":  # Don't copy error dumps recursively
                        src = os.path.join(self.output_dir, item)
                        dst = os.path.join(error_folder, item)
                        if os.path.isdir(src):
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src, dst, follow_symlinks=False)
            
            # Save graph if provided
            if graph:
                graph_file = os.path.join(error_folder, "error_graph.pkl")
                with open(graph_file, 'wb') as f:
                    pickle.dump(graph, f)
            
            self.log_debug(f"Error state dumped to: {error_folder}")
            if not self.verbose:
                print(f"[ERROR] {step_name} failed. Debug files saved to: {error_folder}")
            
        except Exception as e:
            self.log_debug(f"Failed to dump error state: {e}")

    def dump_timeout_case(self, graph=None, mutation_history=None):
        """Dump timeout case files for good benchmark generation"""
        try:
            # First verify that AIG file exists
            aig_source = os.path.join(self.output_dir, "miter", "miter.aig")
            if not os.path.exists(aig_source):
                self.log_debug("Cannot dump timeout case: AIG file not found")
                return None
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            timeout_folder = os.path.join(self.timeout_cases_dir, f"timeout_case_{timestamp}")
            os.makedirs(timeout_folder, exist_ok=True)
            
            # Save timeout case information
            timeout_info_file = os.path.join(timeout_folder, "timeout_info.txt")
            with open(timeout_info_file, 'w') as f:
                f.write(f"Timeout Case: rIC3 solver timeout (>10s)\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Status: Good benchmark case (hard to solve)\n")
                f.write(f"Output Directory: {self.output_dir}\n")
                if graph:
                    f.write(f"Graph Nodes: {graph.number_of_nodes()}\n")
                    f.write(f"Graph Edges: {graph.number_of_edges()}\n")
                if mutation_history:
                    f.write(f"Total Mutations: {len(mutation_history)}\n")
                
                # Check if AIG file exists in source
                if os.path.exists(aig_source):
                    f.write(f"AIG File Status: Available (miter/miter.aig)\n")
                    aig_size = os.path.getsize(aig_source)
                    f.write(f"AIG File Size: {aig_size} bytes\n")
                else:
                    f.write(f"AIG File Status: Missing from source\n")
            
            # Copy all relevant files from output directory
            if os.path.exists(self.output_dir):
                files_to_copy = [
                    "benchmark_1.cpp", "benchmark_2.cpp",  # C++ source files
                    "compile_1", "compile_2",               # HLS compilation results
                    "merged_verilog",                       # Merged Verilog files
                    "miter"                                 # Miter circuit and AIG files
                ]
                
                for item in files_to_copy:
                    src = os.path.join(self.output_dir, item)
                    if os.path.exists(src):
                        dst = os.path.join(timeout_folder, item)
                        try:
                            if os.path.isdir(src):
                                shutil.copytree(src, dst, dirs_exist_ok=True)
                                # Verify AIG file was copied
                                if item == "miter":
                                    aig_file = os.path.join(dst, "miter.aig")
                                    if os.path.exists(aig_file):
                                        self.log_debug(f"Successfully copied miter.aig to timeout case")
                                    else:
                                        self.log_debug(f"Warning: miter.aig not found in copied miter directory")
                            else:
                                shutil.copy2(src, dst, follow_symlinks=False)
                        except Exception as copy_error:
                            self.log_debug(f"Failed to copy {item}: {copy_error}")
                    else:
                        self.log_debug(f"Source file/directory not found: {src}")
                
                # Also copy BTOR2 files if they exist
                btor2_src = self.btor2_output_dir
                if os.path.exists(btor2_src) and os.listdir(btor2_src):
                    btor2_dst = os.path.join(timeout_folder, "btor2")
                    shutil.copytree(btor2_src, btor2_dst, dirs_exist_ok=True)
            
            # Save graph if provided
            if graph:
                graph_file = os.path.join(timeout_folder, "timeout_graph.pkl")
                with open(graph_file, 'wb') as f:
                    pickle.dump(graph, f)
            
            # Save mutation history for this timeout case
            if mutation_history:
                history_file = os.path.join(timeout_folder, "mutation_history.txt")
                with open(history_file, 'w') as f:
                    f.write(f"Timeout Case Mutation History (Total: {len(mutation_history)}):\n")
                    f.write("=" * 50 + "\n")
                    for i, mutation in enumerate(mutation_history, 1):
                        f.write(f"{i:3d}. Action {mutation['action_idx']:2d}: {mutation['action_name']}\n")
            
            self.log_debug(f"Timeout case dumped to: {timeout_folder}")
            
            # Final verification that AIG file was successfully copied
            copied_aig = os.path.join(timeout_folder, "miter", "miter.aig")
            if os.path.exists(copied_aig):
                if not self.verbose:
                    print(f"[GOOD CASE] rIC3 timeout detected. Benchmark with AIG saved to: {timeout_folder}")
                return timeout_folder
            else:
                if not self.verbose:
                    print(f"[WARNING] Timeout case saved but AIG file missing: {timeout_folder}")
                return timeout_folder
            
        except Exception as e:
            self.log_debug(f"Failed to dump timeout case: {e}")
            return None

    def save_best_graph_info(self, best_performance_margin, best_graph, mutation_history, stagnation_counter, best_graph_action_count=None):
        """Save information about the best performing graph"""
        try:
            best_info = {
                "performance_margin": best_performance_margin,
                "graph_nodes": best_graph.number_of_nodes(),
                "graph_edges": best_graph.number_of_edges(),
                "mutation_history_length": len(mutation_history),
                "stagnation_counter": stagnation_counter
            }

            # Add action count if provided
            if best_graph_action_count is not None:
                best_info["best_graph_action_count"] = best_graph_action_count

            info_file = os.path.join(self.output_dir, "best_graph_info.txt")
            with open(info_file, 'w') as f:
                for key, value in best_info.items():
                    f.write(f"{key}: {value}\n")

            self.save_mutation_history(best_performance_margin, mutation_history, stagnation_counter)
        except Exception as e:
            self.log_debug(f"Failed to save best graph info: {e}")

    def save_mutation_history(self, best_performance_margin, mutation_history, stagnation_counter, 
                            total_attempts=None, successful_iterations=None):
        """Save detailed mutation history for analysis"""
        try:
            history_file = os.path.join(self.output_dir, "mutation_history.txt")
            with open(history_file, 'w') as f:
                f.write(f"Best rIC3 Solving Time: {best_performance_margin:.3f}s\n")
                f.write(f"Total Mutations Applied: {len(mutation_history)}\n")
                f.write(f"Current Stagnation Counter: {stagnation_counter}\n")
                
                if total_attempts is not None and successful_iterations is not None:
                    f.write(f"Total Attempts: {total_attempts}\n")
                    f.write(f"Successful Iterations: {successful_iterations}\n")
                    f.write(f"Success Rate: {(successful_iterations/total_attempts*100):.1f}%\n")
                
                f.write("=" * 50 + "\n")
                f.write("Mutation History:\n")

                for i, mutation in enumerate(mutation_history, 1):
                    f.write(f"{i:3d}. Action {mutation['action_idx']:2d}: {mutation['action_name']}\n")

                if not mutation_history:
                    f.write("No mutations applied yet.\n")

        except Exception as e:
            self.log_debug(f"Failed to save mutation history: {e}")

    def print_final_summary(self, best_performance_margin, mutation_history, stagnation_counter,
                          successful_iterations, total_attempts):
        """Print final summary of fuzzing session"""
        success_rate = (successful_iterations/total_attempts*100) if total_attempts > 0 else 0
        
        # Handle infinity display properly
        if best_performance_margin == float('inf'):
            best_time_str = "timeout"
        elif best_performance_margin == float('-inf'):
            best_time_str = "failed"
        else:
            best_time_str = f"{best_performance_margin:.3f}s"
            
        print(f"\nBanditFuzz completed. Best rIC3 time: {best_time_str}")
        print(f"Efficiency: {successful_iterations}/{total_attempts} iterations ({success_rate:.1f}% success rate)")
        
        if self.verbose:
            print(f"Final mutation summary:")
            print(f"  Total mutations: {len(mutation_history)}")
            print(f"  Final stagnation: {stagnation_counter}")
        
        self.save_mutation_history(best_performance_margin, mutation_history, stagnation_counter, 
                                 total_attempts, successful_iterations)
