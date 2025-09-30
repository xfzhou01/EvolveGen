"""
Utility functions for HLS BanditFuzz operations.

This module provides essential utilities for:
- Output management (error dumps, timeout cases, etc.)
- File I/O operations
- Performance tracking and logging
- Context managers for output suppression
"""

import os
import sys
import shutil
import datetime
import pickle
from contextlib import contextmanager
from io import StringIO
from typing import Optional, Dict, List, Any


class BanditFuzzUtils:
    """
    Comprehensive utility class for HLS BanditFuzz operations.
    
    Responsibilities:
    1. Manage output directories and file organization
    2. Dump error states and timeout cases for debugging
    3. Save performance tracking information
    4. Provide output suppression for clean CLI experience
    """
    
    def __init__(self, verbose: bool = False, output_dir: str = "./output"):
        """
        Initialize utilities with output configuration.
        
        Args:
            verbose: If True, enables detailed logging output
            output_dir: Root directory for all output files
        """
        self.verbose = verbose
        self.output_dir = output_dir
        
        # Specialized output directories
        self.error_dump_dir = os.path.join(output_dir, "error_dumps")
        self.timeout_cases_dir = os.path.join(output_dir, "timeout_cases")
        self.btor2_output_dir = os.path.join(output_dir, "btor2")
        
        # Ensure all directories exist
        self._create_directories()

    def _create_directories(self) -> None:
        """Create all necessary output directories if they don't exist."""
        for dir_path in [self.error_dump_dir, self.timeout_cases_dir, self.btor2_output_dir]:
            os.makedirs(dir_path, exist_ok=True)

    @contextmanager
    def suppress_output(self):
        """
        Context manager to suppress stdout when not in verbose mode.
        
        Usage:
            with utils.suppress_output():
                # Code that prints to stdout
                noisy_function()
        """
        if self.verbose:
            # In verbose mode, don't suppress anything
            yield
        else:
            # Redirect stdout to a string buffer
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                yield
            finally:
                sys.stdout = old_stdout

    def log_debug(self, message: str) -> None:
        """
        Log debug message only in verbose mode.
        
        Args:
            message: Debug message to log
        """
        if self.verbose:
            print(f"[DEBUG] {message}")

    def dump_error_state(self, step_name: str, error_msg: str, graph=None) -> Optional[str]:
        """
        Dump complete error state for debugging.
        
        Creates a timestamped directory containing:
        - Error information file
        - Copy of current output directory
        - Serialized graph (if provided)
        
        Args:
            step_name: Name of the step where error occurred
            error_msg: Error message/description
            graph: Optional NetworkX graph to serialize
            
        Returns:
            Path to error dump directory, or None if dump failed
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            error_folder = os.path.join(self.error_dump_dir, f"{step_name}_{timestamp}")
            os.makedirs(error_folder, exist_ok=True)
            
            # Write error information
            error_info_file = os.path.join(error_folder, "error_info.txt")
            with open(error_info_file, 'w') as f:
                f.write(f"Error Step: {step_name}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Error Message: {error_msg}\n")
                f.write(f"Output Directory: {self.output_dir}\n")
                if graph:
                    f.write(f"Graph Nodes: {graph.number_of_nodes()}\n")
                    f.write(f"Graph Edges: {graph.number_of_edges()}\n")
            
            # Copy current output directory (excluding error_dumps to avoid recursion)
            if os.path.exists(self.output_dir):
                for item in os.listdir(self.output_dir):
                    if item == "error_dumps":
                        continue
                    src = os.path.join(self.output_dir, item)
                    dst = os.path.join(error_folder, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst, follow_symlinks=False)
            
            # Serialize graph if provided
            if graph:
                graph_file = os.path.join(error_folder, "error_graph.pkl")
                with open(graph_file, 'wb') as f:
                    pickle.dump(graph, f)
            
            self.log_debug(f"Error state dumped to: {error_folder}")
            if not self.verbose:
                print(f"[ERROR] {step_name} failed. Debug files saved to: {error_folder}")
            
            return error_folder
            
        except Exception as e:
            self.log_debug(f"Failed to dump error state: {e}")
            return None

    def dump_timeout_case(self, graph=None, mutation_history: Optional[List[Dict]] = None) -> Optional[str]:
        """
        Dump timeout case as a good benchmark (hard to solve).
        
        Timeout cases are valuable because they represent challenging benchmarks
        that take >10s to solve with rIC3.
        
        Args:
            graph: NetworkX graph that produced the timeout
            mutation_history: List of mutations applied to reach this state
            
        Returns:
            Path to timeout case directory, or None if dump failed
        """
        try:
            # Verify AIG file exists before dumping
            aig_source = os.path.join(self.output_dir, "miter", "miter.aig")
            if not os.path.exists(aig_source):
                self.log_debug("Cannot dump timeout case: AIG file not found")
                return None
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            timeout_folder = os.path.join(self.timeout_cases_dir, f"timeout_case_{timestamp}")
            os.makedirs(timeout_folder, exist_ok=True)
            
            # Write timeout case information
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
                
                # AIG file status
                aig_size = os.path.getsize(aig_source)
                f.write(f"AIG File Status: Available (miter/miter.aig)\n")
                f.write(f"AIG File Size: {aig_size} bytes\n")
            
            # Copy relevant files from output directory
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
                        else:
                            shutil.copy2(src, dst, follow_symlinks=False)
                    except Exception as copy_error:
                        self.log_debug(f"Failed to copy {item}: {copy_error}")
            
            # Copy BTOR2 files if they exist
            if os.path.exists(self.btor2_output_dir) and os.listdir(self.btor2_output_dir):
                btor2_dst = os.path.join(timeout_folder, "btor2")
                shutil.copytree(self.btor2_output_dir, btor2_dst, dirs_exist_ok=True)
            
            # Serialize graph
            if graph:
                graph_file = os.path.join(timeout_folder, "timeout_graph.pkl")
                with open(graph_file, 'wb') as f:
                    pickle.dump(graph, f)
            
            # Save mutation history
            if mutation_history:
                history_file = os.path.join(timeout_folder, "mutation_history.txt")
                with open(history_file, 'w') as f:
                    f.write(f"Timeout Case Mutation History (Total: {len(mutation_history)}):\n")
                    f.write("=" * 50 + "\n")
                    for i, mutation in enumerate(mutation_history, 1):
                        f.write(f"{i:3d}. Action {mutation['action_idx']:2d}: {mutation['action_name']}\n")
            
            # Verify AIG was successfully copied
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

    def save_candidate_pool_info(self, candidate_pool: List[tuple], iteration: int) -> None:
        """
        Save current state of the candidate pool for analysis.
        
        Args:
            candidate_pool: List of (graph, performance) tuples
            iteration: Current iteration number
        """
        try:
            info_file = os.path.join(self.output_dir, "candidate_pool_info.txt")
            with open(info_file, 'w') as f:
                f.write(f"Candidate Pool Status (Iteration {iteration})\n")
                f.write("=" * 60 + "\n")
                f.write(f"Pool Size: {len(candidate_pool)}\n\n")
                
                # Calculate statistics
                performances = [perf for _, perf in candidate_pool if perf != float('inf')]
                if performances:
                    avg_perf = sum(performances) / len(performances)
                    min_perf = min(performances)
                    max_perf = max(performances)
                    f.write(f"Performance Statistics:\n")
                    f.write(f"  Average: {avg_perf:.3f}s\n")
                    f.write(f"  Min: {min_perf:.3f}s\n")
                    f.write(f"  Max: {max_perf:.3f}s\n\n")
                
                # List all candidates
                f.write("All Candidates:\n")
                for i, (graph, perf) in enumerate(candidate_pool, 1):
                    nodes = graph.number_of_nodes()
                    edges = graph.number_of_edges()
                    perf_str = f"{perf:.3f}s" if perf != float('inf') else "timeout"
                    f.write(f"  {i:2d}. Nodes: {nodes:3d}, Edges: {edges:3d}, Performance: {perf_str}\n")
                    
        except Exception as e:
            self.log_debug(f"Failed to save candidate pool info: {e}")

    def save_mutation_history(self, mutation_history: List[Dict]) -> None:
        """
        Save detailed mutation history for the current lineage.
        
        Args:
            mutation_history: List of mutation records
        """
        try:
            history_file = os.path.join(self.output_dir, "mutation_history.txt")
            with open(history_file, 'w') as f:
                f.write(f"Mutation History (Total: {len(mutation_history)})\n")
                f.write("=" * 60 + "\n")
                
                for i, mutation in enumerate(mutation_history, 1):
                    f.write(f"{i:3d}. Action {mutation['action_idx']:2d}: {mutation['action_name']}\n")
                
                if not mutation_history:
                    f.write("No mutations applied yet.\n")
                    
        except Exception as e:
            self.log_debug(f"Failed to save mutation history: {e}")

    def print_final_summary(self, candidate_pool: List[tuple], total_iterations: int, 
                          successful_iterations: int, total_attempts: int) -> None:
        """
        Print final summary of the fuzzing session.
        
        Args:
            candidate_pool: Final candidate pool
            total_iterations: Target number of iterations
            successful_iterations: Number of successful iterations completed
            total_attempts: Total number of attempts (including failures)
        """
        success_rate = (successful_iterations / total_attempts * 100) if total_attempts > 0 else 0
        
        print("\n" + "=" * 70)
        print("FUZZING SESSION SUMMARY")
        print("=" * 70)
        print(f"Total Iterations Completed: {successful_iterations}/{total_iterations}")
        print(f"Success Rate: {successful_iterations}/{total_attempts} ({success_rate:.1f}%)")
        print(f"Final Candidate Pool Size: {len(candidate_pool)}")
        
        # Performance statistics
        performances = [perf for _, perf in candidate_pool if perf != float('inf')]
        timeout_count = sum(1 for _, perf in candidate_pool if perf == float('inf'))
        
        if performances:
            avg_perf = sum(performances) / len(performances)
            best_perf = max(performances)
            print(f"\nPerformance Statistics:")
            print(f"  Average rIC3 time: {avg_perf:.3f}s")
            print(f"  Best rIC3 time: {best_perf:.3f}s")
            print(f"  Timeout cases: {timeout_count}")
        
        # Graph size statistics
        sizes = [graph.number_of_nodes() for graph, _ in candidate_pool]
        if sizes:
            avg_size = sum(sizes) / len(sizes)
            min_size = min(sizes)
            max_size = max(sizes)
            print(f"\nGraph Size Statistics:")
            print(f"  Average nodes: {avg_size:.1f}")
            print(f"  Min nodes: {min_size}")
            print(f"  Max nodes: {max_size}")
        
        print(f"\nOutput Directory: {self.output_dir}")
        print("=" * 70 + "\n")