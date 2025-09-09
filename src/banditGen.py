import os, time, subprocess, copy, random
from agents import ThompsonSampling
from random_graph_manager import RandomGraphManager
from vitis_hls_compiler import VitisHLSCompiler
from miter_generator import MiterGenerator
from yosys_compiler import YosysCompiler
from utils import BanditFuzzUtils

class HLSBanditFuzz:
    def __init__(self, output_dir="./output", seed=114514, verbose=False):
        # Core components initialization
        self.graph_manager = RandomGraphManager(seed=seed)
        self.hls_compiler = VitisHLSCompiler(working_dir=output_dir)
        self.yosys_compiler = YosysCompiler()
        self.utils = BanditFuzzUtils(verbose=verbose, output_dir=output_dir)

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
        self.current_working_graph = None
        self.current_working_performance = float('-inf')

        # Learning parameters
        self.max_iter = 100
        self.max_stagnation = 5
        self.stagnation_counter = 0
        self.mutation_history = []

        # Configuration
        self.verbose = verbose
        self.output_dir = output_dir
        self.btor2_output_dir = os.path.join(output_dir, "btor2")
        self.generate_dir = "./generate"

        # Create necessary directories
        for dir_path in [self.output_dir, self.btor2_output_dir, self.generate_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def run_hls_pipeline_and_evaluate(self, graph):
        """
        Execute complete HLS pipeline and evaluate performance using rIC3 solving time.
        Returns: (performance_margin, success_status)
        """
        try:
            # Step 1: Generate C++ code
            cpp_files = self._generate_cpp_from_graph(graph)
            if not cpp_files:
                self.utils.log_debug("C++ generation failed")
                self.utils.dump_error_state("cpp_generation", "C++ generation failed", graph)
                return float('-inf'), False

            # Step 2: HLS compilation
            verilog_files = self._compile_with_hls(cpp_files)
            if not verilog_files:
                self.utils.log_debug("HLS compilation failed")
                self.utils.dump_error_state("hls_compilation", "HLS compilation failed", graph)
                return float('-inf'), False

            # Step 3: Generate miter circuit
            miter_result = self._generate_miter_circuit(verilog_files)
            if not miter_result:
                self.utils.log_debug("Miter generation failed")
                self.utils.dump_error_state("miter_generation", "Miter generation failed", graph)
                return float('-inf'), False
            elif miter_result == "COMBINATIONAL_LOGIC":
                self.utils.log_debug("Pure combinational logic detected - will regenerate")
                return float('-inf'), "RETRY"

            # Step 4: Convert to AIG format
            aig_file = self._convert_miter_to_aig(miter_result)
            if not aig_file:
                self.utils.log_debug("AIG conversion failed")
                self.utils.dump_error_state("aig_conversion", "AIG conversion failed", graph)
                return float('-inf'), False

            # Step 5: Run rIC3 solver and measure performance
            ric3_result = self._run_ric3(aig_file)

            # Handle timeout case (good benchmark)
            if ric3_result == "TIMEOUT":
                self.utils.log_debug("rIC3 timeout detected - saving as good benchmark case")
                self.utils.dump_timeout_case(graph, self.mutation_history)
                return float('inf'), True  # Timeout is considered a successful case (good benchmark)

            self.utils.log_debug(f"rIC3 solving time: {ric3_result:.3f}s")
            
            return ric3_result, True

        except Exception as e:
            self.utils.log_debug(f"Pipeline failed: {e}")
            self.utils.dump_error_state("pipeline_exception", str(e), graph)
            if self.verbose:
                import traceback
                traceback.print_exc()
            return float('-inf'), False

    def _generate_cpp_from_graph(self, graph):
        """Generate C++ code from graph using comparison mode"""
        try:
            with self.utils.suppress_output():
                self.graph_manager.program_graph = graph
                cpp_file_1 = os.path.join(self.output_dir, "benchmark_1.cpp")
                cpp_file_2 = os.path.join(self.output_dir, "benchmark_2.cpp")

                self.graph_manager.dump_cpp_comparsion(cpp_file_1, cpp_file_2)

            if os.path.exists(cpp_file_1) and os.path.exists(cpp_file_2):
                return [cpp_file_1, cpp_file_2]
            return None
        except Exception as e:
            self.utils.log_debug(f"C++ generation failed: {e}")
            return None

    def _compile_with_hls(self, cpp_files):
        """Compile C++ files using HLS with different clock periods"""
        try:
            verilog_files_groups = []
            clock_periods = [self.graph_manager.cp_1, self.graph_manager.cp_2]

            for i, cpp_file in enumerate(cpp_files):
                project_name = f"hls_project_{i+1}"
                clock_period = clock_periods[i] if i < len(clock_periods) else 10

                with self.utils.suppress_output():
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
            self.utils.log_debug(f"HLS compilation failed: {e}")
            return None

    def _generate_miter_circuit(self, verilog_files_groups):
        """Generate miter circuit from Verilog files"""
        try:
            if len(verilog_files_groups) < 2:
                self.utils.log_debug("Need at least 2 groups of Verilog files for miter generation")
                return None

            verilog_files_1, verilog_files_2 = verilog_files_groups[0], verilog_files_groups[1]
            merged_verilog_folder = os.path.join(self.output_dir, "merged_verilog")
            os.makedirs(merged_verilog_folder, exist_ok=True)

            with self.utils.suppress_output():
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
            import traceback
            traceback.print_exc()
            self._log_debug(f"Miter generation failed: {e}")
            return None

    def _convert_miter_to_aig(self, miter_directory):
        """Convert miter Verilog to AIG format using Yosys"""
        try:
            miter_file = os.path.join(miter_directory, "miter.v")
            if not os.path.exists(miter_file):
                self.utils.log_debug(f"Miter file not found: {miter_file}")
                return None

            aig_output_dir = os.path.join(self.output_dir, "miter")
            os.makedirs(aig_output_dir, exist_ok=True)
            aig_file = os.path.join(aig_output_dir, "miter.aig")

            with self.utils.suppress_output():
                self.yosys_compiler.execute(
                    verilog_file_path=miter_file,
                    working_dir=aig_output_dir,
                    aiger_file_path=aig_file,
                    top_name="top_A_times_top_B"
                )

            return aig_file if os.path.exists(aig_file) else None
        except Exception as e:
            self.utils.log_debug(f"AIG conversion failed: {e}")
            return None

    def _run_ric3(self, aig_file):
        """Run rIC3 solver and return solving time"""
        try:
            cmd = ["./rIC3-code/target/release/rIC3", aig_file]
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            solve_time = time.time() - start_time

            if "SAT" in result.stdout or "UNSAT" in result.stdout:
                return solve_time
            else:
                self.utils.log_debug(f"rIC3 failed with return code: {result.returncode}")
                return float('inf')

        except subprocess.TimeoutExpired:
            self.utils.log_debug("rIC3 timeout (10s)")
            return "TIMEOUT"
        except Exception as e:
            self.utils.log_debug(f"rIC3 failed: {e}")
            return float('inf')

    def _generate_robust_initial_graph(self):
        """Generate initial graph with guaranteed sequential logic structure"""
        try:
            with self.utils.suppress_output():
                # Reseed RNG to ensure a different graph each generation while keeping determinism
                self.generation_count += 1
                derived_seed = self.seed + self.generation_count  #  derive a new seed per generation
                random.seed(derived_seed)  #  reseed global RNG used by RandomGraphManager
                self.graph_manager.seed = derived_seed  #  keep pragma derivations consistent

                self.graph_manager._reset_all()
                success = self.graph_manager.generate_random_graph(action_number_total=100)

            if success:
                op_nodes = self.graph_manager._get_op_node_list()
                if len(op_nodes) >= 3:
                    if not self.verbose:
                        print(f"Generated initial graph: {len(op_nodes)} nodes")
                    return True

            self.utils.log_debug("Failed to generate sufficient graph complexity")
            return False
        except Exception as e:
            self.utils.log_debug(f"Initial graph generation failed: {e}")
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
            baseline_performance = self.current_working_performance if strategy == 1 else self.best_performance_margin
            
            new_graph, generation_success = self._execute_strategy(strategy)
            if not generation_success:
                continue

            # Evaluate new graph
            performance_margin, success = self.run_hls_pipeline_and_evaluate(new_graph)
            
            if success == "RETRY" or not success:
                self._handle_evaluation_failure(strategy, success)
                continue

            # Process successful evaluation
            successful_iterations += 1
            self._process_successful_iteration(successful_iterations, strategy, new_graph, 
                                            performance_margin, baseline_performance)

        # Final summary
        self.utils.print_final_summary(self.best_performance_margin, self.mutation_history,
                                      self.stagnation_counter, successful_iterations, total_attempts)

    def _initialize_with_valid_graph(self):
        """Initialize with a valid non-combinational graph"""
        if not self._generate_robust_initial_graph():
            return False

        self.best_graph = copy.deepcopy(self.graph_manager.program_graph)
        
        # Retry until we get a valid sequential circuit
        max_retries = 5
        for _ in range(max_retries):
            self.best_performance_margin, success = self.run_hls_pipeline_and_evaluate(self.best_graph)
            
            if success == "RETRY":
                if self._generate_robust_initial_graph():
                    self.best_graph = copy.deepcopy(self.graph_manager.program_graph)
                    continue
                else:
                    return False
            elif success:
                break
            else:
                return False
        else:
            return False

        print(f"Initial rIC3 time: {self.best_performance_margin:.3f}s" if self.best_performance_margin != float('inf') else "Initial rIC3 time: timeout")
        self.current_working_graph = copy.deepcopy(self.best_graph)
        self.current_working_performance = self.best_performance_margin
        return True

    def _execute_strategy(self, strategy):
        """Execute selected strategy: generate new graph or mutate existing"""
        if strategy == 0:  # Generate new graph
            if not self.verbose:
                print("Generating new graph...")
            if self._generate_robust_initial_graph():
                self._reset_working_graph()
                return copy.deepcopy(self.graph_manager.program_graph), True
            return None, False
        else:  # Mutate existing graph
            if not self.verbose:
                print("Mutating working graph...")
            if self.stagnation_counter >= self.max_stagnation:
                if not self.verbose:
                    print(f"Resetting due to stagnation ({self.stagnation_counter} iterations)")
                self._reset_working_graph()
            
            action_idx = self.action_agent.select_action()
            return self._mutate_graph_incremental(action_idx), True

    def _handle_evaluation_failure(self, strategy, _):
        """Handle evaluation failures and provide negative feedback"""
        self.strategy_agent.reward(False)
        if strategy == 1:
            self.action_agent.reward(False)

    def _process_successful_iteration(self, iteration, strategy, new_graph, performance_margin, baseline_performance):
        """Process successful evaluation and update state"""
        # Calculate improvements
        global_improvement = performance_margin > self.best_performance_margin
        local_improvement = performance_margin > baseline_performance
        
        # Basic output for non-verbose mode
        graph_size = new_graph.number_of_nodes()
        time_str = f"{performance_margin:.3f}s" if performance_margin != float('inf') else "timeout"
        if not self.verbose:
            print(f"Iteration {iteration}/{self.max_iter}: Graph size: {graph_size}, rIC3 time: {time_str}")
        
        # Update best graph if globally improved
        if global_improvement:
            if not self.verbose:
                print(f"New best rIC3 time: {performance_margin:.3f}s (was {self.best_performance_margin:.3f}s)")
            self.best_graph = new_graph
            self.best_performance_margin = performance_margin
            self.stagnation_counter = 0
            
            if strategy == 1:
                self.current_working_graph = copy.deepcopy(new_graph)
                self.current_working_performance = performance_margin
            
            self.utils.save_best_graph_info(self.best_performance_margin, self.best_graph,
                                           self.mutation_history, self.stagnation_counter)
        elif local_improvement and strategy == 1:
            if not self.verbose:
                print(f"Local improvement: {performance_margin:.3f}s (was {self.current_working_performance:.3f}s)")
            self.current_working_graph = copy.deepcopy(new_graph)
            self.current_working_performance = performance_margin
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        # Reward agents
        self.strategy_agent.reward(local_improvement)
        if strategy == 1:
            self.action_agent.reward(local_improvement)

    def _mutate_graph_incremental(self, action_idx):
        """Incrementally mutate current working graph"""
        try:
            if self.current_working_graph is None:
                self.current_working_graph = copy.deepcopy(self.best_graph)
                self.mutation_history = []

            self.graph_manager.program_graph = copy.deepcopy(self.current_working_graph)
            action = self.actions[action_idx]
            action_name = getattr(action, '__name__', f"action_{action_idx}")
            
            success = action()
            if success:
                self.graph_manager._make_single_output()
                self.current_working_graph = copy.deepcopy(self.graph_manager.program_graph)
                self.mutation_history.append({
                    'action_idx': action_idx,
                    'action_name': action_name,
                    'iteration': len(self.mutation_history) + 1
                })

            return self.current_working_graph
        except Exception as e:
            self.utils.log_debug(f"Incremental mutation failed: {e}")
            return self.current_working_graph or self.best_graph

    def _reset_working_graph(self):
        """Reset working graph to best graph state"""
        self.current_working_graph = copy.deepcopy(self.best_graph)
        self.current_working_performance = self.best_performance_margin
        self.mutation_history = []
        self.stagnation_counter = 0