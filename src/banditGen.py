"""
Hybrid Evolutionary Bandit Fuzzing for HLS Model Checking Benchmark Generation.

This module implements a novel algorithm that combines:
1. Population-based evolutionary search (inspired by AIGROW)
2. Multi-armed bandit learning (Thompson Sampling)

Key Innovation:
Instead of hill-climbing around a single "best" solution, we maintain a diverse
candidate pool (gene pool) and evolve it through intelligent mutations and fresh
injections. This prevents unbounded graph growth while maintaining diversity.

Algorithm Components:
- Candidate Pool: A population of (graph, performance) tuples representing the gene pool
- Action Agent: Learns which mutation operations are most effective
- Strategy Agent: Decides between "Evolve" (mutate existing) or "Inject" (generate fresh)
"""

import os
import time
import subprocess
import copy
import random
from typing import List, Tuple, Optional, Dict, Any

from agents import ThompsonSampling
from random_graph_manager import RandomGraphManager
from vitis_hls_compiler import VitisHLSCompiler
from miter_generator import MiterGenerator
from yosys_compiler import YosysCompiler
from utils import BanditFuzzUtils


class HLSBanditFuzz:
    """
    Main fuzzing engine implementing Hybrid Evolutionary Bandit Fuzzing.
    
    The algorithm maintains a candidate pool (gene pool) of diverse graphs
    and uses two intelligent agents to guide the evolutionary process:
    
    1. Strategy Agent: Decides macro-strategy (Evolve vs Inject)
    2. Action Agent: Selects specific mutation operations
    
    This design naturally maintains diversity and prevents unbounded growth.
    """
    
    def __init__(self, output_dir: str = "./output", seed: int = 114514, 
                 verbose: bool = False, initial_action_count: Optional[int] = None):
        """
        Initialize the fuzzing engine.
        
        Args:
            output_dir: Directory for all output files
            seed: Base random seed for reproducibility
            verbose: Enable detailed logging
            initial_action_count: Initial number of actions for graph generation
        """
        # ===== Core Components =====
        self.graph_manager = RandomGraphManager(seed=seed)
        self.hls_compiler = VitisHLSCompiler(working_dir=output_dir)
        self.yosys_compiler = YosysCompiler()
        self.utils = BanditFuzzUtils(verbose=verbose, output_dir=output_dir)
        
        # ===== Seed Management =====
        # We use a base seed and increment it for each generation to ensure
        # reproducibility while still getting different graphs each time
        self.seed = seed
        self.generation_count = 0
        
        # ===== The Candidate Pool (Gene Pool) =====
        # This is the heart of our evolutionary algorithm
        # Each entry is (graph, performance_score) where performance_score is rIC3 solving time
        self.candidate_pool: List[Tuple[Any, float]] = []
        
        # ===== Intelligent Agents =====
        # Action Agent: Learns which mutations work best
        self.actions = self.graph_manager.bandit_action_list
        self.action_agent = ThompsonSampling(
            n_actions=len(self.actions),
            decay=0.99,           # High decay to preserve historical information
            initial_alpha=10,     # Conservative initialization
            initial_beta=5        # Encourages exploration
        )
        
        # Strategy Agent: Decides between Evolve and Inject
        # Action 0: Evolve (mutate from pool)
        # Action 1: Inject (generate fresh graph)
        self.strategy_agent = ThompsonSampling(
            n_actions=2,
            decay=0.99,
            initial_alpha=10,
            initial_beta=5
        )
        
        # ===== Evolution Tracking =====
        self.current_lineage_history: List[Dict] = []  # Mutations in current evolution line
        
        # ===== Configuration =====
        self.verbose = verbose
        self.output_dir = output_dir
        self.btor2_output_dir = os.path.join(output_dir, "btor2")
        
        # Create necessary directories
        for dir_path in [self.output_dir, self.btor2_output_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # ===== Action Count Management =====
        # This controls the size of newly generated graphs
        if initial_action_count is not None:
            if not isinstance(initial_action_count, int) or initial_action_count <= 0:
                raise ValueError("initial_action_count must be a positive integer")
            self.default_action_count = int(initial_action_count)
        else:
            self.default_action_count = 100  # Default
        
        # ===== Iteration Control =====
        self.max_iter = 100  # Target number of successful evaluations
        
        # ===== Timing Trackers =====
        self._fuzz_start_time = None
        self._first_timeout_elapsed = None

    # ========================================================================
    # CORE ALGORITHM: Main Fuzzing Loop
    # ========================================================================

    def fuzz(self) -> None:
        """
        Main fuzzing loop implementing the Hybrid Evolutionary Bandit algorithm.
        
        Algorithm Overview:
        1. Initialize candidate pool with a valid graph
        2. For each iteration:
           a. Strategy agent decides: Evolve or Inject?
           b. Execute chosen strategy
           c. Evaluate resulting graph
           d. Update candidate pool based on results
           e. Provide feedback to agents
        """
        print("=" * 70)
        print("Starting Hybrid Evolutionary Bandit Fuzzing")
        print("=" * 70)
        
        self._fuzz_start_time = time.time()
        
        # ===== Phase 1: Initialization =====
        if not self._initialize_candidate_pool():
            print("[FATAL] Failed to initialize candidate pool")
            return
        
        print(f"Initial pool size: {len(self.candidate_pool)}")
        self._print_pool_stats()
        
        # ===== Phase 2: Main Evolution Loop =====
        successful_iterations = 0
        total_attempts = 0
        
        while successful_iterations < self.max_iter:
            total_attempts += 1
            iteration_start = time.time()
            
            # Step 1: Strategy selection
            strategy = self.strategy_agent.select_action()
            strategy_name = "EVOLVE" if strategy == 0 else "INJECT"
            
            if not self.verbose:
                print(f"\n[Iteration {successful_iterations + 1}/{self.max_iter}] Strategy: {strategy_name}")
            
            # Step 2: Execute strategy
            new_graph, parent_performance = self._execute_strategy(strategy)
            if new_graph is None:
                self.utils.log_debug(f"{strategy_name} failed to produce a graph")
                continue
            
            # Step 3: Evaluate new graph
            performance, success = self.run_hls_pipeline_and_evaluate(new_graph)
            
            if success == "RETRY":
                # Pure combinational logic detected, skip this iteration
                self._handle_evaluation_failure(strategy, is_retry=True)
                continue
            elif not success:
                # Evaluation failed
                self._handle_evaluation_failure(strategy, is_retry=False)
                continue
            
            # Step 4: Process successful evaluation
            successful_iterations += 1
            self._process_successful_evaluation(
                strategy=strategy,
                new_graph=new_graph,
                performance=performance,
                parent_performance=parent_performance,
                iteration=successful_iterations
            )
            
            iteration_time = time.time() - iteration_start
            if not self.verbose:
                print(f"Iteration completed in {iteration_time:.2f}s")
                print("-" * 70)
        
        # ===== Phase 3: Final Summary =====
        self.utils.print_final_summary(
            self.candidate_pool,
            self.max_iter,
            successful_iterations,
            total_attempts
        )

    # ========================================================================
    # STRATEGY EXECUTION
    # ========================================================================

    def _execute_strategy(self, strategy: int) -> Tuple[Optional[Any], float]:
        """
        Execute the chosen strategy: Evolve or Inject.
        
        Args:
            strategy: 0 for Evolve, 1 for Inject
            
        Returns:
            (new_graph, parent_performance) tuple
            - new_graph: The generated/mutated graph, or None on failure
            - parent_performance: Performance of parent (for Evolve) or pool average (for Inject)
        """
        if strategy == 0:
            return self._strategy_evolve()
        else:
            return self._strategy_inject()

    def _strategy_evolve(self) -> Tuple[Optional[Any], float]:
        """
        EVOLVE strategy: Select a parent from the pool and mutate it.
        
        This is analogous to sexual reproduction in biology - we select
        a parent and apply mutations to create offspring.
        
        Returns:
            (child_graph, parent_performance) tuple
        """
        if not self.candidate_pool:
            self.utils.log_debug("Cannot evolve: pool is empty")
            return None, float('-inf')
        
        # Select random parent from pool (uniform selection promotes diversity)
        parent_graph, parent_performance = random.choice(self.candidate_pool)
        
        if not self.verbose:
            parent_size = parent_graph.number_of_nodes()
            perf_str = f"{parent_performance:.3f}s" if parent_performance != float('inf') else "timeout"
            print(f"  Selected parent: {parent_size} nodes, performance: {perf_str}")
        
        # Apply mutation
        action_idx = self.action_agent.select_action()
        child_graph = self._mutate_graph(parent_graph, action_idx)
        
        return child_graph, parent_performance

    def _strategy_inject(self) -> Tuple[Optional[Any], float]:
        """
        INJECT strategy: Generate a completely fresh graph.
        
        This introduces new genetic material into the pool, preventing
        premature convergence and maintaining diversity.
        
        The size of the new graph is based on the current pool's average size,
        which naturally adapts to the evolution state.
        
        Returns:
            (fresh_graph, pool_average_performance) tuple
        """
        # Calculate average size from current pool
        avg_size = self._calculate_average_pool_size()
        target_action_count = max(50, int(avg_size))  # At least 50 actions
        
        if not self.verbose:
            print(f"  Generating fresh graph with ~{target_action_count} actions")
        
        # Generate fresh graph
        success = self._generate_fresh_graph(action_count=target_action_count)
        if not success:
            return None, float('-inf')
        
        fresh_graph = copy.deepcopy(self.graph_manager.program_graph)
        
        # Return pool average as the baseline for comparison
        pool_avg = self._calculate_average_pool_performance()
        return fresh_graph, pool_avg

    # ========================================================================
    # EVALUATION AND FEEDBACK
    # ========================================================================

    def _process_successful_evaluation(self, strategy: int, new_graph: Any,
                                       performance: float, parent_performance: float,
                                       iteration: int) -> None:
        """
        Process a successful evaluation and update the candidate pool.
        
        This implements the core evolutionary logic:
        - For EVOLVE: Add to pool if child > parent (successful evolution)
        - For INJECT: Add to pool if new graph > pool average (meets admission threshold)
        
        Args:
            strategy: 0 for Evolve, 1 for Inject
            new_graph: The evaluated graph
            performance: Performance (rIC3 time) of the new graph
            parent_performance: Baseline performance for comparison
            iteration: Current iteration number
        """
        graph_size = new_graph.number_of_nodes()
        perf_str = f"{performance:.3f}s" if performance != float('inf') else "timeout"
        
        if strategy == 0:  # EVOLVE
            self._process_evolve_result(new_graph, performance, parent_performance, graph_size, perf_str)
        else:  # INJECT
            self._process_inject_result(new_graph, performance, parent_performance, graph_size, perf_str)
        
        # Save pool state periodically
        if iteration % 10 == 0:
            self.utils.save_candidate_pool_info(self.candidate_pool, iteration)

    def _process_evolve_result(self, child_graph: Any, child_performance: float,
                               parent_performance: float, graph_size: int, perf_str: str) -> None:
        """
        Process the result of an EVOLVE strategy.
        
        Success criterion: child_performance > parent_performance
        (Higher is better - means harder to solve)
        """
        is_improvement = child_performance > parent_performance
        
        if is_improvement:
            # Successful evolution!
            self.candidate_pool.append((child_graph, child_performance))
            self.utils.log_debug(f"Evolution SUCCESS: {graph_size} nodes, {perf_str}")
            
            if not self.verbose:
                parent_str = f"{parent_performance:.3f}s" if parent_performance != float('inf') else "timeout"
                print(f"  ✓ Evolution successful! {perf_str} > {parent_str}")
                print(f"  Pool size: {len(self.candidate_pool)}")
            
            # Save mutation history for this successful lineage
            if self.current_lineage_history:
                self.utils.save_mutation_history(self.current_lineage_history)
            
            # Positive reward to both agents
            self.strategy_agent.reward(True)
            self.action_agent.reward(True)
        else:
            # Evolution failed
            self.utils.log_debug(f"Evolution FAILED: {perf_str} <= {parent_performance:.3f}s")
            
            if not self.verbose:
                parent_str = f"{parent_performance:.3f}s" if parent_performance != float('inf') else "timeout"
                print(f"  ✗ Evolution failed: {perf_str} <= {parent_str}")
            
            # Negative reward
            self.strategy_agent.reward(False)
            self.action_agent.reward(False)

    def _process_inject_result(self, fresh_graph: Any, fresh_performance: float,
                               pool_avg_performance: float, graph_size: int, perf_str: str) -> None:
        """
        Process the result of an INJECT strategy.
        
        Success criterion: fresh_performance > pool_avg_performance
        (New graph must be better than average to enter the pool)
        """
        meets_threshold = fresh_performance > pool_avg_performance
        
        if meets_threshold:
            # Successful injection!
            self.candidate_pool.append((fresh_graph, fresh_performance))
            self.utils.log_debug(f"Injection SUCCESS: {graph_size} nodes, {perf_str}")
            
            if not self.verbose:
                avg_str = f"{pool_avg_performance:.3f}s" if pool_avg_performance != float('inf') else "timeout"
                print(f"  ✓ Injection successful! {perf_str} > pool_avg {avg_str}")
                print(f"  Pool size: {len(self.candidate_pool)}")
            
            # Positive reward
            self.strategy_agent.reward(True)
        else:
            # Injection failed to meet threshold
            self.utils.log_debug(f"Injection FAILED: {perf_str} <= pool_avg {pool_avg_performance:.3f}s")
            
            if not self.verbose:
                avg_str = f"{pool_avg_performance:.3f}s" if pool_avg_performance != float('inf') else "timeout"
                print(f"  ✗ Injection failed: {perf_str} <= pool_avg {avg_str}")
            
            # Negative reward
            self.strategy_agent.reward(False)

    def _handle_evaluation_failure(self, strategy: int, is_retry: bool = False) -> None:
        """
        Handle evaluation failures and provide negative feedback to agents.
        
        Args:
            strategy: Which strategy was being executed
            is_retry: True if this was a RETRY (combinational logic), False if hard failure
        """
        if is_retry:
            self.utils.log_debug("Combinational logic detected - will regenerate")
        else:
            self.utils.log_debug("Evaluation failed")
        
        # Negative feedback
        self.strategy_agent.reward(False)
        if strategy == 0:  # Only action agent is involved in EVOLVE
            self.action_agent.reward(False)

    # ========================================================================
    # GRAPH MANIPULATION
    # ========================================================================

    def _mutate_graph(self, parent_graph: Any, action_idx: int) -> Any:
        """
        Apply a mutation to a parent graph.
        
        Args:
            parent_graph: The graph to mutate
            action_idx: Index of the mutation action to apply
            
        Returns:
            Mutated graph (a copy of the parent with mutation applied)
        """
        try:
            # Work on a copy of the parent
            self.graph_manager.program_graph = copy.deepcopy(parent_graph)
            
            # Apply mutation
            action = self.actions[action_idx]
            action_name = getattr(action, '__name__', f"action_{action_idx}")
            
            success = action()
            if success:
                self.graph_manager._make_single_output()
                
                # Record mutation in current lineage
                self.current_lineage_history.append({
                    'action_idx': action_idx,
                    'action_name': action_name,
                    'step': len(self.current_lineage_history) + 1
                })
                
                if not self.verbose:
                    print(f"  Applied mutation: {action_name}")
            
            return copy.deepcopy(self.graph_manager.program_graph)
            
        except Exception as e:
            self.utils.log_debug(f"Mutation failed: {e}")
            return parent_graph  # Return parent unchanged on failure

    def _generate_fresh_graph(self, action_count: int) -> bool:
        """
        Generate a completely fresh random graph.
        
        Args:
            action_count: Number of actions to use for graph generation
            
        Returns:
            True if generation succeeded, False otherwise
        """
        try:
            with self.utils.suppress_output():
                # Increment generation counter and derive new seed
                self.generation_count += 1
                derived_seed = self.seed + self.generation_count
                random.seed(derived_seed)
                self.graph_manager.seed = derived_seed
                
                # Generate graph
                self.graph_manager._reset_all()
                success = self.graph_manager.generate_random_graph(action_number_total=action_count)
            
            if success:
                op_nodes = self.graph_manager._get_op_node_list()
                if len(op_nodes) >= 3:
                    if not self.verbose:
                        print(f"  Generated: {len(op_nodes)} nodes")
                    return True
            
            return False
            
        except Exception as e:
            self.utils.log_debug(f"Fresh graph generation failed: {e}")
            return False

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def _initialize_candidate_pool(self) -> bool:
        """
        Initialize the candidate pool with a valid starting graph.
        
        This ensures we always start with at least one working graph
        that produces sequential logic (not pure combinational).
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        print("Initializing candidate pool...")
        
        # Generate initial graph
        if not self._generate_fresh_graph(action_count=self.default_action_count):
            return False
        
        initial_graph = copy.deepcopy(self.graph_manager.program_graph)
        
        # Retry until we get valid sequential logic
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            performance, success = self.run_hls_pipeline_and_evaluate(initial_graph)
            
            if success == "RETRY":
                # Combinational logic, try again
                print(f"  Attempt {attempt}: Combinational logic detected, regenerating...")
                if not self._generate_fresh_graph(action_count=self.default_action_count):
                    return False
                initial_graph = copy.deepcopy(self.graph_manager.program_graph)
                continue
            elif success:
                # Success!
                self.candidate_pool.append((initial_graph, performance))
                perf_str = f"{performance:.3f}s" if performance != float('inf') else "timeout"
                print(f"  Initial graph performance: {perf_str}")
                return True
            else:
                # Hard failure
                return False
        
        print(f"  Failed after {max_retries} attempts")
        return False

    # ========================================================================
    # POOL STATISTICS
    # ========================================================================

    def _calculate_average_pool_size(self) -> float:
        """
        Calculate the average graph size (number of nodes) in the candidate pool.
        
        Returns:
            Average number of nodes across all graphs in pool
        """
        if not self.candidate_pool:
            return self.default_action_count
        
        sizes = [graph.number_of_nodes() for graph, _ in self.candidate_pool]
        return sum(sizes) / len(sizes)

    def _calculate_average_pool_performance(self) -> float:
        """
        Calculate the average performance across the candidate pool.
        
        Timeout cases (inf) are excluded from the average.
        
        Returns:
            Average rIC3 solving time (excluding timeouts)
        """
        if not self.candidate_pool:
            return float('-inf')
        
        performances = [perf for _, perf in self.candidate_pool if perf != float('inf')]
        if not performances:
            return float('-inf')
        
        return sum(performances) / len(performances)

    def _print_pool_stats(self) -> None:
        """Print current statistics about the candidate pool."""
        if not self.candidate_pool:
            return
        
        sizes = [graph.number_of_nodes() for graph, _ in self.candidate_pool]
        performances = [perf for _, perf in self.candidate_pool if perf != float('inf')]
        timeouts = sum(1 for _, perf in self.candidate_pool if perf == float('inf'))
        
        print(f"Pool statistics:")
        print(f"  Size: {len(self.candidate_pool)} graphs")
        if sizes:
            print(f"  Avg nodes: {sum(sizes)/len(sizes):.1f} (min: {min(sizes)}, max: {max(sizes)})")
        if performances:
            print(f"  Avg performance: {sum(performances)/len(performances):.3f}s")
            print(f"  Best performance: {max(performances):.3f}s")
        if timeouts:
            print(f"  Timeout cases: {timeouts}")

    # ========================================================================
    # HLS PIPELINE EXECUTION
    # ========================================================================

    def run_hls_pipeline_and_evaluate(self, graph: Any) -> Tuple[float, Any]:
        """
        Execute complete HLS pipeline and evaluate performance using rIC3 solving time.
        
        Pipeline steps:
        1. Generate C++ code from graph
        2. Compile with Vitis HLS
        3. Generate miter circuit
        4. Convert to AIG format
        5. Run rIC3 solver and measure time
        
        Args:
            graph: NetworkX graph to evaluate
            
        Returns:
            (performance_margin, success_status) tuple where:
            - performance_margin: rIC3 solving time (float) or inf (timeout)
            - success_status: True (success), False (failure), or "RETRY" (combinational logic)
        """
        try:
            # Step 1: Generate C++ code
            cpp_files = self._generate_cpp_from_graph(graph)
            if not cpp_files:
                self.utils.dump_error_state("cpp_generation", "C++ generation failed", graph)
                return float('-inf'), False

            # Step 2: HLS compilation
            verilog_files = self._compile_with_hls(cpp_files)
            if not verilog_files:
                self.utils.dump_error_state("hls_compilation", "HLS compilation failed", graph)
                return float('-inf'), False

            # Step 3: Generate miter circuit
            miter_result = self._generate_miter_circuit(verilog_files)
            if not miter_result:
                self.utils.dump_error_state("miter_generation", "Miter generation failed", graph)
                return float('-inf'), False
            elif miter_result == "COMBINATIONAL_LOGIC":
                # Special case: pure combinational logic detected
                return float('-inf'), "RETRY"

            # Step 4: Convert to AIG format
            aig_file = self._convert_miter_to_aig(miter_result)
            if not aig_file:
                self.utils.dump_error_state("aig_conversion", "AIG conversion failed", graph)
                return float('-inf'), False

            # Step 5: Run rIC3 solver
            ric3_result = self._run_ric3(aig_file)

            # Handle timeout (good benchmark case)
            if ric3_result == "TIMEOUT":
                self.utils.log_debug("rIC3 timeout detected - saving as good benchmark")
                
                # Track time to first timeout
                if self._first_timeout_elapsed is None and self._fuzz_start_time is not None:
                    self._first_timeout_elapsed = time.time() - self._fuzz_start_time
                    if not self.verbose:
                        print(f"  Time to first timeout: {self._first_timeout_elapsed:.3f}s")
                
                self.utils.dump_timeout_case(graph, self.current_lineage_history)
                return float('inf'), True  # Timeout is success (hardest case)

            return ric3_result, True

        except Exception as e:
            self.utils.log_debug(f"Pipeline failed: {e}")
            self.utils.dump_error_state("pipeline_exception", str(e), graph)
            return float('-inf'), False

    def _generate_cpp_from_graph(self, graph: Any) -> Optional[List[str]]:
        """Generate C++ code from graph using comparison mode."""
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

    def _compile_with_hls(self, cpp_files: List[str]) -> Optional[List[List[str]]]:
        """Compile C++ files using HLS with different clock periods."""
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

    def _generate_miter_circuit(self, verilog_files_groups: List[List[str]]) -> Optional[str]:
        """Generate miter circuit from Verilog files."""
        try:
            if len(verilog_files_groups) < 2:
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
                    if "expected to have `ap_rst` port" in str(ve) or \
                       "expected to have `ap_clk` port" in str(ve):
                        return "COMBINATIONAL_LOGIC"
                    else:
                        raise ve

        except Exception as e:
            self.utils.log_debug(f"Miter generation failed: {e}")
            return None

    def _convert_miter_to_aig(self, miter_directory: str) -> Optional[str]:
        """Convert miter Verilog to AIG format using Yosys."""
        try:
            miter_file = os.path.join(miter_directory, "miter.v")
            if not os.path.exists(miter_file):
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

    def _run_ric3(self, aig_file: str) -> Any:
        """Run rIC3 solver and return solving time."""
        try:
            cmd = ["../rIC3-code/target/release/rIC3", aig_file]
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            solve_time = time.time() - start_time

            if "SAT" in result.stdout or "UNSAT" in result.stdout:
                return solve_time
            else:
                return float('inf')

        except subprocess.TimeoutExpired:
            return "TIMEOUT"
        except Exception as e:
            self.utils.log_debug(f"rIC3 failed: {e}")
            return float('inf')