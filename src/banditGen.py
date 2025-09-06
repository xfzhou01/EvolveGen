import os, time, subprocess, copy, sys, shutil, datetime, random
from contextlib import contextmanager
from io import StringIO
from agents import ThompsonSampling
from random_graph_manager import RandomGraphManager
from vitis_hls_compiler import VitisHLSCompiler
from miter_generator import MiterGenerator
from yosys_compiler import YosysCompiler

class HLSBanditFuzz:
    def __init__(self, output_dir="./output", seed=42, verbose=False):
        # Core components initialization
        self.graph_manager = RandomGraphManager(seed=seed)
        self.hls_compiler = VitisHLSCompiler(working_dir=output_dir)
        self.yosys_compiler = YosysCompiler()

        # Seed management for reproducible but evolving generations
        self.seed = seed  # base seed
        self.generation_count = 0  # increments before each new graph

        # BanditFuzz agents with conservative parameters for balanced exploration
        self.actions = self.graph_manager.bandit_action_list
        self.action_agent = ThompsonSampling(n_actions=len(self.actions), decay=0.99, initial_alpha=10, initial_beta=5)
        self.strategy_agent = ThompsonSampling(n_actions=2, decay=0.99, initial_alpha=10, initial_beta=5)  # Generate vs Mutate

        # Performance tracking
        self.best_graph = None
        self.best_performance_margin = float('-inf')

        # Elite pool for population-based evolution
        self.elite_pool = []  # List of (performance, graph, action_count) tuples
        self.pool_size = 5  # Maximum elite pool capacity
        self.initial_pool_size = 5  # Initial population size to fill
        self.pool_refresh_ratio = 0.2  # Ratio of individuals to replace in refresh strategy

        # Learning parameters
        self.max_iter = 100
        self.max_stagnation = 5
        self.mutation_history = []

        # Configuration
        self.verbose = verbose
        self.output_dir = output_dir
        self.btor2_output_dir = os.path.join(output_dir, "btor2")
        self.generate_dir = "./generate"
        self.error_dump_dir = os.path.join(output_dir, "error_dumps")
        self.timeout_cases_dir = os.path.join(output_dir, "timeout_cases")

        # Create necessary directories
        for dir_path in [self.output_dir, self.btor2_output_dir, self.generate_dir, self.error_dump_dir, self.timeout_cases_dir]:
            os.makedirs(dir_path, exist_ok=True)

    @contextmanager
    def _suppress_output(self):
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

    def run_hls_pipeline_and_evaluate(self, graph):
        """
        Execute complete HLS pipeline and evaluate performance using rIC3 solving time.
        Returns: (performance_margin, success_status)
        """
        try:
            # Step 1: Generate C++ code
            cpp_files = self._generate_cpp_from_graph(graph)
            if not cpp_files:
                self._log_debug("C++ generation failed")
                self._dump_error_state("cpp_generation", "C++ generation failed", graph)
                return float('-inf'), False

            # Step 2: HLS compilation
            verilog_files = self._compile_with_hls(cpp_files)
            if not verilog_files:
                self._log_debug("HLS compilation failed")
                self._dump_error_state("hls_compilation", "HLS compilation failed", graph)
                return float('-inf'), False

            # Step 3: Generate miter circuit
            miter_result = self._generate_miter_circuit(verilog_files)
            if not miter_result:
                self._log_debug("Miter generation failed")
                self._dump_error_state("miter_generation", "Miter generation failed", graph)
                return float('-inf'), False
            elif miter_result == "COMBINATIONAL_LOGIC":
                self._log_debug("Pure combinational logic detected - will regenerate")
                return float('-inf'), "RETRY"

            # Step 4: Convert to AIG format
            aig_file = self._convert_miter_to_aig(miter_result)
            if not aig_file:
                self._log_debug("AIG conversion failed")
                self._dump_error_state("aig_conversion", "AIG conversion failed", graph)
                return float('-inf'), False

            # Step 5: Run rIC3 solver and measure performance
            ric3_result = self._run_ric3(aig_file)
            
            # Handle timeout case (good benchmark)
            if ric3_result == "TIMEOUT":
                self._log_debug("rIC3 timeout detected - saving as good benchmark case")
                self._dump_timeout_case(graph)
                return float('inf'), True  # Timeout is considered a successful case (good benchmark)
            
            self._log_debug(f"rIC3 solving time: {ric3_result:.3f}s")
            
            return ric3_result, True

        except Exception as e:
            self._log_debug(f"Pipeline failed: {e}")
            self._dump_error_state("pipeline_exception", str(e), graph)
            if self.verbose:
                import traceback
                traceback.print_exc()
            return float('-inf'), False

    def _generate_cpp_from_graph(self, graph):
        """Generate C++ code from graph using comparison mode"""
        try:
            with self._suppress_output():
                self.graph_manager.program_graph = graph
                cpp_file_1 = os.path.join(self.output_dir, "benchmark_1.cpp")
                cpp_file_2 = os.path.join(self.output_dir, "benchmark_2.cpp")
                
                self.graph_manager.dump_cpp_comparsion(cpp_file_1, cpp_file_2)
            
            if os.path.exists(cpp_file_1) and os.path.exists(cpp_file_2):
                return [cpp_file_1, cpp_file_2]
            return None
        except Exception as e:
            self._log_debug(f"C++ generation failed: {e}")
            return None

    def _compile_with_hls(self, cpp_files):
        """Compile C++ files using HLS with different clock periods"""
        try:
            verilog_files_groups = []
            clock_periods = [self.graph_manager.cp_1, self.graph_manager.cp_2]
            
            for i, cpp_file in enumerate(cpp_files):
                project_name = f"hls_project_{i+1}"
                clock_period = clock_periods[i] if i < len(clock_periods) else 10
                
                with self._suppress_output():
                    result = self.hls_compiler.compile(
                        project_name=project_name,
                        top_name="top",
                        clock_period=clock_period,
                        cpp_file_list=[cpp_file]
                    )
                
                if result["success"]:
                    verilog_files_groups.append(result["verilog_files"])
                else:
                    return None
            
            return verilog_files_groups
        except Exception as e:
            self._log_debug(f"HLS compilation failed: {e}")
            return None

    def _generate_miter_circuit(self, verilog_files_groups):
        """Generate miter circuit from Verilog files"""
        try:
            if len(verilog_files_groups) < 2:
                self._log_debug("Need at least 2 groups of Verilog files for miter generation")
                return None

            verilog_files_1, verilog_files_2 = verilog_files_groups[0], verilog_files_groups[1]
            merged_verilog_folder = os.path.join(self.output_dir, "merged_verilog")
            os.makedirs(merged_verilog_folder, exist_ok=True)

            with self._suppress_output():
                miter_generator = MiterGenerator(
                    verilog_file_path_list_1=verilog_files_1,
                    verilog_file_path_list_2=verilog_files_2,
                    merged_verilog_folder_path=merged_verilog_folder,
                    top_name="top"
                )

                try:
                    miter_generator.generate_miter(insert_assertions=False)
                    return merged_verilog_folder
                except ValueError as ve:
                    if "expected to have `ap_rst` port" in str(ve) or "expected to have `ap_clk` port" in str(ve):
                        return "COMBINATIONAL_LOGIC"
                    else:
                        raise ve

        except Exception as e:
            self._log_debug(f"Miter generation failed: {e}")
            return None

    def _convert_miter_to_aig(self, miter_directory):
        """Convert miter Verilog to AIG format using Yosys"""
        try:
            miter_file = os.path.join(miter_directory, "miter.v")
            if not os.path.exists(miter_file):
                self._log_debug(f"Miter file not found: {miter_file}")
                return None

            aig_output_dir = os.path.join(self.output_dir, "miter")
            os.makedirs(aig_output_dir, exist_ok=True)
            aig_file = os.path.join(aig_output_dir, "miter.aig")

            with self._suppress_output():
                self.yosys_compiler.execute(
                    verilog_file_path=miter_file,
                    working_dir=aig_output_dir,
                    aiger_file_path=aig_file,
                    top_name="top_A_times_top_B"
                )

            return aig_file if os.path.exists(aig_file) else None
        except Exception as e:
            self._log_debug(f"AIG conversion failed: {e}")
            return None

    def _run_ric3(self, aig_file):
        """Run rIC3 solver and return solving time"""
        try:
            cmd = ["./rIC3", aig_file, "--engine", "ic3"]
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            solve_time = time.time() - start_time

            if "SAT" in result.stdout or "UNSAT" in result.stdout:
                return solve_time
            else:
                self._log_debug(f"rIC3 failed with return code: {result.returncode}")
                return float('inf')

        except subprocess.TimeoutExpired:
            self._log_debug("rIC3 timeout (10s)")
            return "TIMEOUT"  # Special identifier for timeout
        except Exception as e:
            self._log_debug(f"rIC3 failed: {e}")
            return float('inf')

    def _generate_robust_initial_graph(self):
        """Generate initial graph with guaranteed sequential logic structure"""
        return self._generate_robust_initial_graph_with_complexity(20)

    def _generate_robust_initial_graph_with_complexity(self, action_count):
        """Generate initial graph with specified complexity"""
        try:
            with self._suppress_output():
                # Reseed RNG to ensure a different graph each generation while keeping determinism
                self.generation_count += 1
                derived_seed = self.seed + self.generation_count  #  derive a new seed per generation
                random.seed(derived_seed)  #  reseed global RNG used by RandomGraphManager
                self.graph_manager.seed = derived_seed  #  keep pragma derivations consistent

                self.graph_manager._reset_all()
                success = self.graph_manager.generate_random_graph(action_number_total=action_count)

            if success:
                op_nodes = self.graph_manager._get_op_node_list()
                if len(op_nodes) >= 3:
                    if not self.verbose:
                        print(f"Generated initial graph: {len(op_nodes)} nodes (complexity: {action_count})")
                    return True

            self._log_debug("Failed to generate sufficient graph complexity")
            return False
        except Exception as e:
            self._log_debug(f"Initial graph generation failed: {e}")
            return False

    def fuzz(self):
        """Main BanditFuzz fuzzing loop with simplified output"""
        print("Starting HLS BanditFuzz...")

        # Generate and validate initial graph
        if not self._initialize_with_valid_graph():
            print("Failed to generate valid initial graph")
            return

        # Main fuzzing loop
        successful_iterations = 0
        total_attempts = 0

        while successful_iterations < self.max_iter:
            total_attempts += 1

            # Strategy selection and execution
            strategy = self.strategy_agent.select_action()

            new_graph, baseline_performance, generation_success = self._execute_strategy(strategy)
            if not generation_success:
                # Strategy execution failed after max attempts - this is rare but possible
                self._log_debug(f"Strategy {strategy} failed after maximum attempts")
                continue

            # Evaluate new graph with unlimited retry mechanism
            eval_attempt = 0
            while True:
                performance_margin, success = self.run_hls_pipeline_and_evaluate(new_graph)

                if success == "RETRY":
                    # Regenerate graph and try again
                    eval_attempt += 1
                    if not self.verbose and eval_attempt % 5 == 0:
                        print(f"  Evaluation retry {eval_attempt}...")

                    # Generate new graph with same strategy
                    new_graph, baseline_performance, generation_success = self._execute_strategy(strategy)
                    # generation_success is always True now, so continue
                    continue
                elif success:
                    # Evaluation succeeded
                    break
                else:
                    # Evaluation failed, try again with same graph
                    eval_attempt += 1
                    if not self.verbose and eval_attempt % 5 == 0:
                        print(f"  Evaluation retry {eval_attempt}...")
                    continue

            # Process successful evaluation
            successful_iterations += 1
            self._process_successful_iteration(successful_iterations, strategy, new_graph,
                                            performance_margin, baseline_performance)

        # Final summary
        self._print_final_summary(successful_iterations, total_attempts)

    def _initialize_with_valid_graph(self):
        """Initialize elite pool with valid non-combinational graphs"""
        print(f"Initializing elite pool with {self.initial_pool_size} individuals...")

        successful_individuals = 0
        max_attempts = self.initial_pool_size * 10  # Allow more attempts to fill pool

        for attempt in range(max_attempts):
            if successful_individuals >= self.initial_pool_size:
                break

            # Generate a new initial graph with unlimited retry
            gen_attempt = 0
            while True:
                if self._generate_robust_initial_graph():
                    candidate_graph = copy.deepcopy(self.graph_manager.program_graph)
                    break

                gen_attempt += 1
                if gen_attempt % 10 == 0:
                    print(f"  Initial graph generation attempt {gen_attempt}...")

            # Evaluate the candidate graph with unlimited retry mechanism
            eval_retry = 0
            while True:
                performance_margin, success = self.run_hls_pipeline_and_evaluate(candidate_graph)

                if success == "RETRY" or not success:
                    # Regenerate graph for retry or after failed evaluation
                    if success == "RETRY":
                        eval_retry += 1
                        if eval_retry % 5 == 0:
                            print(f"  Evaluation retry {eval_retry} (combinational)...")
                    else:
                        # Only print a simple notice to avoid clutter
                        print("  Evaluation failed, regenerating graph...")

                    retry_gen = 0
                    while True:
                        if self._generate_robust_initial_graph():
                            candidate_graph = copy.deepcopy(self.graph_manager.program_graph)
                            break

                        retry_gen += 1
                        if retry_gen % 5 == 0:
                            print(f"    Retry graph generation attempt {retry_gen}...")
                    continue
                elif success:
                    # Successfully evaluated - add to elite pool
                    action_count = 10  # Fixed action count for initial graphs
                    self.elite_pool.append((performance_margin, candidate_graph, action_count))
                    successful_individuals += 1

                    time_str = f"{performance_margin:.3f}s" if performance_margin != float('inf') else "timeout"
                    print(f"Added individual {successful_individuals}/{self.initial_pool_size} to elite pool: rIC3 time {time_str}")
                    break

        if successful_individuals == 0:
            print("Failed to generate any valid initial graphs for elite pool")
            return False

        # Sort elite pool by performance (descending order)
        self.elite_pool.sort(key=lambda x: x[0], reverse=True)

        # Set global best from the top of elite pool
        best_performance, best_graph, _ = self.elite_pool[0]
        self.best_graph = copy.deepcopy(best_graph)
        self.best_performance_margin = best_performance

        print(f"Elite pool initialized with {len(self.elite_pool)} individuals")
        best_time_str = f"{self.best_performance_margin:.3f}s" if self.best_performance_margin != float('inf') else "timeout"
        print(f"Best initial rIC3 time: {best_time_str}")

        return True

    def _execute_strategy(self, strategy):
        """Execute selected strategy: refresh population or mutate from population"""

        if strategy == 0:  # Refresh population strategy
            if not self.verbose:
                print("Refreshing population...")

            # Calculate average action count from elite pool for complexity guidance
            if self.elite_pool:
                avg_action_count = sum(action_count for _, _, action_count in self.elite_pool) / len(self.elite_pool)
                avg_action_count = int(round(avg_action_count))
            else:
                avg_action_count = 10  # Default fallback

            # Keep trying until generation succeeds
            attempt = 0
            while True:
                if self._generate_robust_initial_graph_with_complexity(avg_action_count):
                    new_graph = copy.deepcopy(self.graph_manager.program_graph)
                    # Set initial action count attribute for tracking
                    new_graph.initial_action_count = avg_action_count
                    baseline_performance = self.best_performance_margin
                    return new_graph, baseline_performance, True

                attempt += 1
                if not self.verbose and attempt % 10 == 0:
                    print(f"  Graph generation attempt {attempt}...")

        else:  # Mutate from population strategy
            if not self.verbose:
                print("Mutating from population...")

            if not self.elite_pool:
                # Fallback if pool is empty
                return None, float('-inf'), False

            # Keep trying until mutation succeeds
            attempt = 0
            while True:
                # Randomly select a parent from elite pool
                parent_performance, parent_graph, parent_action_count = random.choice(self.elite_pool)

                # Mutate the parent graph
                mutated_graph = self._mutate_graph_incremental(parent_graph)
                if mutated_graph is not None:
                    # Attach parent info for tracking in _process_successful_iteration
                    mutated_graph.parent_info = (parent_graph, parent_action_count)
                    baseline_performance = parent_performance
                    return mutated_graph, baseline_performance, True

                attempt += 1
                if not self.verbose and attempt % 10 == 0:
                    print(f"  Mutation attempt {attempt}...")

    def _handle_evaluation_failure(self, strategy, _):
        """Handle evaluation failures and provide negative feedback"""
        self.strategy_agent.reward(False)
        if strategy == 1:
            self.action_agent.reward(False)

    def _process_successful_iteration(self, iteration, strategy, new_graph, performance_margin, baseline_performance):
        """Process successful evaluation and maintain elite pool"""
        # Calculate action count for the new graph
        if strategy == 0:  # Refresh strategy - use initial action count
            action_count = getattr(new_graph, 'initial_action_count', 10)
        else:  # Mutation strategy - increment parent's action count
            if hasattr(new_graph, 'parent_info'):
                _, parent_action_count = new_graph.parent_info
                action_count = parent_action_count + 1
            else:
                action_count = 1  # Fallback

        # Basic output for non-verbose mode
        graph_size = new_graph.number_of_nodes()
        time_str = f"{performance_margin:.3f}s" if performance_margin != float('inf') else "timeout"
        if not self.verbose:
            print(f"Iteration {iteration}/{self.max_iter}: Graph size: {graph_size}, rIC3 time: {time_str}")

        # Determine if new graph should be added to elite pool
        should_add_to_pool = False

        if len(self.elite_pool) < self.pool_size:
            # Pool not full - add directly
            should_add_to_pool = True
        else:
            # Pool full - check if better than worst individual
            worst_performance = min(perf for perf, _, _ in self.elite_pool)
            if performance_margin > worst_performance:
                should_add_to_pool = True

        # Add to elite pool if qualified
        if should_add_to_pool:
            self.elite_pool.append((performance_margin, copy.deepcopy(new_graph), action_count))

            # Remove worst individual if pool exceeds capacity
            if len(self.elite_pool) > self.pool_size:
                self.elite_pool.sort(key=lambda x: x[0], reverse=True)
                self.elite_pool = self.elite_pool[:self.pool_size]
            else:
                # Keep pool sorted
                self.elite_pool.sort(key=lambda x: x[0], reverse=True)

            if not self.verbose:
                print(f"Added to elite pool (size: {len(self.elite_pool)}/{self.pool_size})")

        # Update global best if improved
        pool_best_performance = self.elite_pool[0][0] if self.elite_pool else float('-inf')
        if pool_best_performance > self.best_performance_margin:
            self.best_performance_margin = pool_best_performance
            self.best_graph = copy.deepcopy(self.elite_pool[0][1])

            if not self.verbose:
                best_time_str = f"{self.best_performance_margin:.3f}s" if self.best_performance_margin != float('inf') else "timeout"
                print(f"New global best rIC3 time: {best_time_str}")

            self._save_best_graph()

        # Calculate improvement for agent rewards
        local_improvement = performance_margin > baseline_performance

        # Reward agents based on local improvement
        self.strategy_agent.reward(local_improvement)
        if strategy == 1:
            self.action_agent.reward(local_improvement)

    def _mutate_graph_incremental(self, input_graph):
        """Incrementally mutate the given input graph"""
        try:
            # Set the input graph as the working graph
            self.graph_manager.program_graph = copy.deepcopy(input_graph)

            # Select a random action for mutation
            action_idx = self.action_agent.select_action()
            action = self.actions[action_idx]
            action_name = getattr(action, '__name__', f"action_{action_idx}")

            success = action()
            if success:
                mutated_graph = copy.deepcopy(self.graph_manager.program_graph)

                # Update mutation history for tracking
                self.mutation_history.append({
                    'action_idx': action_idx,
                    'action_name': action_name,
                    'iteration': len(self.mutation_history) + 1
                })

                return mutated_graph
            else:
                # Mutation failed, return None to indicate failure
                return None

        except Exception as e:
            self._log_debug(f"Incremental mutation failed: {e}")
            return None



    def _save_best_graph(self):
        """Save information about the best performing graph"""
        try:
            best_info = {
                "performance_margin": self.best_performance_margin,
                "graph_nodes": self.best_graph.number_of_nodes(),
                "graph_edges": self.best_graph.number_of_edges(),
                "mutation_history_length": len(self.mutation_history),
                "elite_pool_size": len(self.elite_pool)
            }

            info_file = os.path.join(self.output_dir, "best_graph_info.txt")
            with open(info_file, 'w') as f:
                for key, value in best_info.items():
                    f.write(f"{key}: {value}\n")

            self._save_mutation_history()
        except Exception as e:
            self._log_debug(f"Failed to save best graph info: {e}")

    def _save_mutation_history(self, total_attempts=None, successful_iterations=None):
        """Save detailed mutation history for analysis"""
        try:
            history_file = os.path.join(self.output_dir, "mutation_history.txt")
            with open(history_file, 'w') as f:
                best_time_str = f"{self.best_performance_margin:.3f}s" if self.best_performance_margin != float('inf') else "timeout"
                f.write(f"Best rIC3 Solving Time: {best_time_str}\n")
                f.write(f"Total Mutations Applied: {len(self.mutation_history)}\n")
                f.write(f"Elite Pool Size: {len(self.elite_pool)}/{self.pool_size}\n")

                if total_attempts is not None and successful_iterations is not None:
                    f.write(f"Total Attempts: {total_attempts}\n")
                    f.write(f"Successful Iterations: {successful_iterations}\n")
                    f.write(f"Success Rate: {(successful_iterations/total_attempts*100):.1f}%\n")

                f.write("=" * 50 + "\n")
                f.write("Elite Pool Summary:\n")
                for i, (perf, _, action_count) in enumerate(self.elite_pool[:5], 1):  # Show top 5
                    perf_str = f"{perf:.3f}s" if perf != float('inf') else "timeout"
                    f.write(f"{i:2d}. Performance: {perf_str}, Actions: {action_count}\n")

                f.write("=" * 50 + "\n")
                f.write("Mutation History:\n")

                for i, mutation in enumerate(self.mutation_history, 1):
                    f.write(f"{i:3d}. Action {mutation['action_idx']:2d}: {mutation['action_name']}\n")

                if not self.mutation_history:
                    f.write("No mutations applied yet.\n")

        except Exception as e:
            self._log_debug(f"Failed to save mutation history: {e}")

    def _print_final_summary(self, successful_iterations, total_attempts):
        """Print final summary of fuzzing session"""
        success_rate = (successful_iterations/total_attempts*100) if total_attempts > 0 else 0
        
        # Handle infinity display properly
        if self.best_performance_margin == float('inf'):
            best_time_str = "timeout"
        elif self.best_performance_margin == float('-inf'):
            best_time_str = "failed"
        else:
            best_time_str = f"{self.best_performance_margin:.3f}s"
            
        print(f"\nBanditFuzz completed. Best rIC3 time: {best_time_str}")
        print(f"Efficiency: {successful_iterations}/{total_attempts} iterations ({success_rate:.1f}% success rate)")
        
        if self.verbose:
            print(f"Final summary:")
            print(f"  Total mutations: {len(self.mutation_history)}")
            print(f"  Elite pool size: {len(self.elite_pool)}/{self.pool_size}")
            if self.elite_pool:
                avg_actions = sum(ac for _, _, ac in self.elite_pool) / len(self.elite_pool)
                print(f"  Average complexity: {avg_actions:.1f} actions")
        
        self._save_mutation_history(total_attempts, successful_iterations)

    def _log_debug(self, message):
        """Log debug message only in verbose mode"""
        if self.verbose:
            print(f"[DEBUG] {message}")

    def _dump_error_state(self, step_name, error_msg, graph=None):
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
                import pickle
                graph_file = os.path.join(error_folder, "error_graph.pkl")
                with open(graph_file, 'wb') as f:
                    pickle.dump(graph, f)
            
            self._log_debug(f"Error state dumped to: {error_folder}")
            if not self.verbose:
                print(f"[ERROR] {step_name} failed. Debug files saved to: {error_folder}")
            
        except Exception as e:
            self._log_debug(f"Failed to dump error state: {e}")

    def _dump_timeout_case(self, graph=None):
        """Dump timeout case files for good benchmark generation"""
        try:
            # First verify that AIG file exists
            aig_source = os.path.join(self.output_dir, "miter", "miter.aig")
            if not os.path.exists(aig_source):
                self._log_debug("Cannot dump timeout case: AIG file not found")
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
                if hasattr(self, 'mutation_history'):
                    f.write(f"Total Mutations: {len(self.mutation_history)}\n")
                
                # Check if AIG file exists in source
                aig_source = os.path.join(self.output_dir, "miter", "miter.aig")
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
                                        self._log_debug(f"Successfully copied miter.aig to timeout case")
                                    else:
                                        self._log_debug(f"Warning: miter.aig not found in copied miter directory")
                            else:
                                shutil.copy2(src, dst, follow_symlinks=False)
                        except Exception as copy_error:
                            self._log_debug(f"Failed to copy {item}: {copy_error}")
                    else:
                        self._log_debug(f"Source file/directory not found: {src}")
                
                # Also copy BTOR2 files if they exist
                btor2_src = self.btor2_output_dir
                if os.path.exists(btor2_src) and os.listdir(btor2_src):
                    btor2_dst = os.path.join(timeout_folder, "btor2")
                    shutil.copytree(btor2_src, btor2_dst, dirs_exist_ok=True)
            
            # Save graph if provided
            if graph:
                import pickle
                graph_file = os.path.join(timeout_folder, "timeout_graph.pkl")
                with open(graph_file, 'wb') as f:
                    pickle.dump(graph, f)
            
            # Save mutation history for this timeout case
            if hasattr(self, 'mutation_history') and self.mutation_history:
                history_file = os.path.join(timeout_folder, "mutation_history.txt")
                with open(history_file, 'w') as f:
                    f.write(f"Timeout Case Mutation History (Total: {len(self.mutation_history)}):\n")
                    f.write("=" * 50 + "\n")
                    for i, mutation in enumerate(self.mutation_history, 1):
                        f.write(f"{i:3d}. Action {mutation['action_idx']:2d}: {mutation['action_name']}\n")
            
            self._log_debug(f"Timeout case dumped to: {timeout_folder}")
            
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
            self._log_debug(f"Failed to dump timeout case: {e}")
            return None