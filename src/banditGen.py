import os, glob, time, subprocess, copy
from agents import ThompsonSampling
from random_graph_manager import RandomGraphManager
from vitis_hls_compiler import VitisHLSCompiler
from miter_generator import MiterGenerator
from yosys_compiler import YosysCompiler

class HLSBanditFuzz:
	def __init__(self, output_dir="./output", seed=42, verbose=False):
		# HLS toolchain components
		self.graph_manager = RandomGraphManager(seed=seed)
		self.hls_compiler = VitisHLSCompiler(working_dir=output_dir)
		self.yosys_compiler = YosysCompiler()
		# MiterGenerator will be created dynamically when needed

		# BanditFuzz components
		self.actions = self.graph_manager.bandit_action_list
		# Use more conservative parameters to encourage balanced exploration
		self.action_agent = ThompsonSampling(n_actions=len(self.actions), decay=0.99, initial_alpha=5, initial_beta=5)
		self.strategy_agent = ThompsonSampling(n_actions=2, decay=0.99, initial_alpha=5, initial_beta=5)  # Generate new graph vs. Mutate existing graph

		# State management
		self.best_graph = None
		self.best_performance_margin = float('-inf')  # Goal is to maximize performance difference
		self.max_iter = 1000
		self.verbose = verbose

		# Incremental mutation state
		self.current_working_graph = None  # Current graph being mutated
		self.current_working_performance = float('-inf')  # Performance of current working graph
		self.mutation_history = []  # Track applied actions for debugging
		self.stagnation_counter = 0  # Count iterations without improvement
		self.max_stagnation = 10  # Reset working graph after this many stagnant iterations

		# Path configuration
		self.output_dir = output_dir
		self.btor2_output_dir = os.path.join(output_dir, "btor2")
		self.generate_dir = "./generate"
		os.makedirs(self.output_dir, exist_ok=True)
		os.makedirs(self.btor2_output_dir, exist_ok=True)
		os.makedirs(self.generate_dir, exist_ok=True)

	def run_hls_pipeline_and_evaluate(self, graph):
		"""
		Runs the complete HLS flow and evaluates performance difference.
		Returns: (performance_margin, success)
		"""
		try:
			# 1. Generate C++ code
			if self.verbose:
				print("[DEBUG] Step 1: Generating C++ code...")
			cpp_files = self._generate_cpp_from_graph(graph)
			if not cpp_files:
				if self.verbose:
					print("[ERROR] Step 1 failed: C++ generation")
				return float('-inf'), False

			# 2. HLS compilation to generate Verilog
			if self.verbose:
				print("[DEBUG] Step 2: HLS compilation...")
			verilog_files = self._compile_with_hls(cpp_files)
			if not verilog_files:
				if self.verbose:
					print("[ERROR] Step 2 failed: HLS compilation")
				return float('-inf'), False

			# 3. Generate Miter circuit
			if self.verbose:
				print("[DEBUG] Step 3: Miter generation...")
			miter_file = self._generate_miter_circuit(verilog_files)
			if not miter_file:
				if self.verbose:
					print("[ERROR] Step 3 failed: Miter generation")
				return float('-inf'), False

			# 4. Convert to BTOR2 format
			if self.verbose:
				print("[DEBUG] Step 4: BTOR2 conversion...")
			btor2_file = self._convert_to_btor2(miter_file)
			if not btor2_file:
				if self.verbose:
					print("[ERROR] Step 4 failed: BTOR2 conversion")
				return float('-inf'), False

			# 4.5. Fix BTOR2 file: convert output to bad for property checking
			self._fix_btor2_output_to_bad(btor2_file)

			# 5. Run dual solver test
			if self.verbose:
				print("[DEBUG] Step 5: Running solvers...")
			smt_sweeper_time = self._run_smt_sweeper(btor2_file)
			bitwuzla_time = self._run_bitwuzla(btor2_file)

			# 6. Calculate performance difference (Goal: bitwuzla slow, smt-sweeper fast)
			performance_margin = bitwuzla_time - smt_sweeper_time

			if self.verbose:
				print(f"SMT-Sweeper time: {smt_sweeper_time:.3f}s")
				print(f"Bitwuzla time: {bitwuzla_time:.3f}s")
				print(f"Performance margin: {performance_margin:.3f}s")

			return performance_margin, True

		except Exception as e:
			if self.verbose:
				print(f"Pipeline failed: {e}")
				import traceback
				traceback.print_exc()
			return float('-inf'), False

	def _generate_cpp_from_graph(self, graph):
		"""Generates C++ code from the graph"""
		try:
			# Set the graph for the graph manager
			self.graph_manager.program_graph = graph

			# Generate two versions of C++ code for comparison
			cpp_file_1 = os.path.join(self.output_dir, "benchmark_1.cpp")
			cpp_file_2 = os.path.join(self.output_dir, "benchmark_2.cpp")

			# Use dump_cpp_comparsion method to generate two versions
			self.graph_manager.dump_cpp_comparsion(cpp_file_1, cpp_file_2)

			# Check if files were actually generated
			if os.path.exists(cpp_file_1) and os.path.exists(cpp_file_2):
				if self.verbose:
					print(f"[DEBUG] C++ files generated: {cpp_file_1}, {cpp_file_2}")
				return [cpp_file_1, cpp_file_2]
			else:
				if self.verbose:
					print(f"[DEBUG] C++ generation failed. File1 exists: {os.path.exists(cpp_file_1)}, File2 exists: {os.path.exists(cpp_file_2)}")
				return None
		except Exception as e:
			if self.verbose:
				print(f"C++ generation failed: {e}")
				import traceback
				traceback.print_exc()
			return None

	def _compile_with_hls(self, cpp_files):
		"""Compiles C++ code using the HLS compiler"""
		try:
			# Return grouped Verilog files, not a mixed list
			verilog_files_groups = []
			for i, cpp_file in enumerate(cpp_files, 1):
				project_name = f"hls_project_{i}"

				# Use different clock periods, as in the normal flow
				if i == 1:
					clock_period = self.graph_manager.cp_1
				elif i == 2:
					clock_period = self.graph_manager.cp_2
				else:
					clock_period = 10  # Default value

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
			if self.verbose:
				print(f"HLS compilation failed: {e}")
			return None

	def _generate_miter_circuit(self, verilog_files_groups):
		"""Generates the Miter circuit"""
		try:
			# verilog_files_groups is now a list of lists
			if len(verilog_files_groups) < 2:
				if self.verbose:
					print("Need at least 2 groups of Verilog files for miter generation")
				return None

			# Use the first and second groups of Verilog files
			verilog_files_1 = verilog_files_groups[0]
			verilog_files_2 = verilog_files_groups[1]

			if self.verbose:
				print(f"[DEBUG] Group 1 Verilog files: {verilog_files_1}")
				print(f"[DEBUG] Group 2 Verilog files: {verilog_files_2}")

			# Create MiterGenerator
			merged_verilog_folder = os.path.join(self.output_dir, "merged_verilog")
			os.makedirs(merged_verilog_folder, exist_ok=True)

			miter_generator = MiterGenerator(
				verilog_file_path_list_1=verilog_files_1,
				verilog_file_path_list_2=verilog_files_2,
				merged_verilog_folder_path=merged_verilog_folder,
				top_name="top"
			)

			# Generate Miter circuit, returns the top module name
			kairos_top = miter_generator.generate_miter()

			# Return the directory path containing the miter.v file
			if self.verbose:
				print(f"[DEBUG] Miter generation completed. Top module: {kairos_top}")
				print(f"[DEBUG] Miter directory: {merged_verilog_folder}")

			return merged_verilog_folder

		except Exception as e:
			if self.verbose:
				print(f"Miter generation failed: {e}")
				import traceback
				traceback.print_exc()
			return None

	def _convert_to_btor2(self, miter_file):
		"""Converts the Miter circuit to BTOR2 format"""
		try:
			# miter_file should be a directory path containing the miter.v file
			# If miter_file is a file path, we need to get its directory
			if os.path.isfile(miter_file):
				input_dir = os.path.dirname(miter_file)
			else:
				input_dir = miter_file

			if self.verbose:
				print(f"[DEBUG] Input directory for BTOR2 conversion: {input_dir}")
				print(f"[DEBUG] Files in input directory: {os.listdir(input_dir) if os.path.exists(input_dir) else 'Directory not found'}")

			# Call the conversion script with the correct parameter format
			cmd = [
				"python3",
				"script/miter_to_btor.py",
				input_dir,  # input_folder
				self.btor2_output_dir  # output_folder
			]

			if self.verbose:
				print(f"[DEBUG] Running BTOR2 conversion: {' '.join(cmd)}")

			result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

			if self.verbose:
				print(f"[DEBUG] BTOR2 conversion stdout: {result.stdout}")
				print(f"[DEBUG] BTOR2 conversion stderr: {result.stderr}")
				print(f"[DEBUG] BTOR2 conversion return code: {result.returncode}")

			if result.returncode == 0:
				# Check the generated BTOR2 file
				btor2_file = os.path.join(self.btor2_output_dir, "miter.btor2")
				if os.path.exists(btor2_file):
					return btor2_file
				else:
					# Look for any .btor2 file
					btor2_files = glob.glob(os.path.join(self.btor2_output_dir, "*.btor2"))
					if btor2_files:
						return btor2_files[0]
					else:
						if self.verbose:
							print("No BTOR2 files found in output directory")
						return None
			else:
				if self.verbose:
					print(f"BTOR2 conversion failed with return code {result.returncode}")
				return None
		except Exception as e:
			if self.verbose:
				print(f"BTOR2 conversion failed: {e}")
				import traceback
				traceback.print_exc()
			return None

	def _run_smt_sweeper(self, btor2_file):
		"""Runs SMT-Sweeper solver (Reference Solver - expected to be fast)"""
		try:
			cmd = [
				"./solver/smt-sweeper",
				"-f", btor2_file,
				"-i", "100",
				"-b", "300",
				"--dump_smt"
			]

			start_time = time.time()
			result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
			end_time = time.time()

			solve_time = end_time - start_time

			if result.returncode == 0:
				# Save the generated SMT file to the generate directory
				if result.stdout:
					timestamp = int(time.time() * 1000000)
					smt_file = os.path.join(self.generate_dir, f"{timestamp}.smt2")
					with open(smt_file, 'w') as f:
						f.write(result.stdout)
				return solve_time
			else:
				return float('inf')  # Solving failed

		except subprocess.TimeoutExpired:
			return float('inf')  # Timeout
		except Exception as e:
			if self.verbose:
				print(f"SMT-Sweeper failed: {e}")
			return float('inf')

	def _run_bitwuzla(self, btor2_file):
		"""Runs Bitwuzla solver (Target Solver - expected to be slow)"""
		try:
			# Find the most recently generated SMT file
			latest_smt_file = self._get_latest_smt_file()
			if not latest_smt_file:
				if self.verbose:
					print(f"No SMT file found for Bitwuzla, using BTOR2 file: {btor2_file}")
				# If no SMT file, use the BTOR2 file directly
				latest_smt_file = btor2_file

			cmd = [
				"./solver/bitwuzla/build/src/main/bitwuzla",
				latest_smt_file
			]

			start_time = time.time()
			result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
			end_time = time.time()

			solve_time = end_time - start_time

			if self.verbose:
				print(f"Bitwuzla result: {result.returncode}, time: {solve_time:.3f}s")

			return solve_time

		except subprocess.TimeoutExpired:
			if self.verbose:
				print("Bitwuzla timeout")
			return float('inf')  # Timeout
		except Exception as e:
			if self.verbose:
				print(f"Bitwuzla failed: {e}")
			return float('inf')

	def _get_latest_smt_file(self):
		"""Gets the latest SMT file in the generate directory"""
		try:
			# Simplified version of file lookup
			smt_files = glob.glob(os.path.join(self.generate_dir, "*.smt2"))
			if smt_files:
				# Sort by modification time, return the latest
				latest_file = max(smt_files, key=os.path.getmtime)
				return latest_file
			return None
		except Exception:
			return None

	def _fix_btor2_output_to_bad(self, btor2_file):
		"""Convert 'output' declarations to 'bad' in BTOR2 file for property checking"""
		try:
			if self.verbose:
				print("[DEBUG] Fixing BTOR2 file: converting output to bad...")

			# Read the BTOR2 file
			with open(btor2_file, 'r') as f:
				lines = f.readlines()

			# Process each line
			modified_lines = []
			output_count = 0
			for line in lines:
				# Check if this line declares an output
				if ' output ' in line:
					# Convert output to bad
					modified_line = line.replace(' output ', ' bad ')
					modified_lines.append(modified_line)
					output_count += 1
					if self.verbose:
						print(f"[DEBUG] Converted output to bad: {line.strip()} -> {modified_line.strip()}")
				else:
					modified_lines.append(line)

			# Write back the modified content
			with open(btor2_file, 'w') as f:
				f.writelines(modified_lines)

			if self.verbose:
				print(f"[DEBUG] Fixed {output_count} output declarations in BTOR2 file")

		except Exception as e:
			if self.verbose:
				print(f"[WARNING] Failed to fix BTOR2 file: {e}")
			# Don't fail the entire pipeline for this

	def fuzz(self):
		"""Main BanditFuzz fuzzing loop"""
		print("[INFO] Starting HLS BanditFuzz...")

		# Generate initial graph
		print("[INFO] Generating initial graph...")
		success = self.graph_manager.generate_random_graph(action_number_total=20)
		if not success:
			print("[ERROR] Failed to generate initial graph")
			return

		self.best_graph = copy.deepcopy(self.graph_manager.program_graph)
		self.best_performance_margin, success = self.run_hls_pipeline_and_evaluate(self.best_graph)

		if not success:
			print(f"[ERROR] Failed to evaluate initial graph. Performance margin: {self.best_performance_margin}")
			return

		print(f"[INFO] Initial performance margin: {self.best_performance_margin:.3f}")

		# Initialize working graph for incremental mutation
		self.current_working_graph = copy.deepcopy(self.best_graph)
		self.current_working_performance = self.best_performance_margin

		# Main loop
		for iteration in range(1, self.max_iter + 1):
			print(f"\n[INFO] Iteration {iteration}/{self.max_iter}")

			# Agent 2 decides strategy: 0=Generate new graph, 1=Mutate existing graph
			strategy = self.strategy_agent.select_action()

			# Store baseline performance for reward calculation
			baseline_performance = self.current_working_performance if strategy == 1 else self.best_performance_margin

			if strategy == 0:  # Generate new graph
				print("[INFO] Generating new graph...")
				success = self.graph_manager.generate_random_graph(action_number_total=20)
				if success:
					new_graph = copy.deepcopy(self.graph_manager.program_graph)
					# Reset working graph when generating new graph
					self._reset_working_graph()
				else:
					continue
			else:  # Incremental mutation of working graph
				print("[INFO] Incrementally mutating working graph...")

				# Check if we should reset due to stagnation
				if self.stagnation_counter >= self.max_stagnation:
					print(f"[INFO] Resetting working graph due to stagnation ({self.stagnation_counter} iterations)")
					self._reset_working_graph()

				action_idx = self.action_agent.select_action()
				new_graph = self._mutate_graph_incremental(action_idx)

				if self.verbose:
					print(f"[DEBUG] Mutation history length: {len(self.mutation_history)}")
					if self.mutation_history:
						recent_actions = [h['action_name'] for h in self.mutation_history[-3:]]
						print(f"[DEBUG] Recent actions: {recent_actions}")

			# Evaluate the new graph
			performance_margin, success = self.run_hls_pipeline_and_evaluate(new_graph)

			if not success:
				# Evaluation failed, give negative reward
				self.strategy_agent.reward(False)
				if strategy == 1:  # Only reward action agent if mutation
					self.action_agent.reward(False)
				continue

			# Calculate rewards based on improvement type
			global_improvement = performance_margin > self.best_performance_margin
			local_improvement = performance_margin > baseline_performance

			# For strategy agent: reward based on whether the chosen strategy led to any improvement
			strategy_reward = local_improvement

			# For action agent: reward based on local improvement when mutating
			action_reward = local_improvement if strategy == 1 else False

			if global_improvement:
				print(f"[GLOBAL IMPROVE] New best margin: {performance_margin:.3f} (was {self.best_performance_margin:.3f})")
				self.best_graph = new_graph
				self.best_performance_margin = performance_margin

				# Reset stagnation counter on global improvement
				self.stagnation_counter = 0

				# Update working graph to the new best graph for future mutations
				if strategy == 1:  # Only update working graph if this was a mutation
					self.current_working_graph = copy.deepcopy(new_graph)
					self.current_working_performance = performance_margin
					if self.verbose:
						print(f"[DEBUG] Updated working graph to new best. Mutation history preserved.")

				# Save best graph
				self._save_best_graph()
			elif local_improvement and strategy == 1:
				print(f"[LOCAL IMPROVE] Working graph improved: {performance_margin:.3f} (was {self.current_working_performance:.3f})")
				# Update working graph even if not globally best
				self.current_working_graph = copy.deepcopy(new_graph)
				self.current_working_performance = performance_margin
				# Reset stagnation counter on local improvement
				self.stagnation_counter = 0
			else:
				# Increment stagnation counter only if no improvement at all
				self.stagnation_counter += 1
				if self.verbose:
					improvement_type = "global" if not global_improvement else "local"
					print(f"[DEBUG] No {improvement_type} improvement. Stagnation counter: {self.stagnation_counter}/{self.max_stagnation}")

			# Reward agents with improved reward mechanism
			self.strategy_agent.reward(strategy_reward)
			if strategy == 1:  # Only reward action agent if mutation
				self.action_agent.reward(action_reward)

			print(f"[INFO] Current margin: {performance_margin:.3f}, Best: {self.best_performance_margin:.3f}")
			if strategy == 1 and self.verbose:
				print(f"[INFO] Working graph mutations applied: {len(self.mutation_history)}")

			# Print detailed mutation summary every 10 iterations
			if iteration % 10 == 0:
				self._print_mutation_summary()

		print(f"\n[INFO] BanditFuzz completed. Best performance margin: {self.best_performance_margin:.3f}")

		# Final summary
		print(f"[INFO] Final mutation summary:")
		print(f"  Total mutations in best path: {len(self.mutation_history)}")
		print(f"  Final stagnation counter: {self.stagnation_counter}")
		self._save_mutation_history()

	def _mutate_graph_incremental(self, action_idx):
		"""
		Incrementally mutates the current working graph.
		This implements true incremental mutation as described in BanditFuzz paper.
		"""
		try:
			# Use current working graph as base
			if self.current_working_graph is None:
				# Initialize working graph from best graph
				self.current_working_graph = copy.deepcopy(self.best_graph)
				self.mutation_history = []
				if self.verbose:
					print("[DEBUG] Initialized working graph from best graph")

			# Set the working graph in graph manager
			self.graph_manager.program_graph = copy.deepcopy(self.current_working_graph)

			# Execute the selected action
			action = self.actions[action_idx]
			action_name = action.__name__ if hasattr(action, '__name__') else f"action_{action_idx}"

			if self.verbose:
				print(f"[DEBUG] Applying action: {action_name}")

			success = action()

			if success:
				# Update working graph with the mutation result
				self.current_working_graph = copy.deepcopy(self.graph_manager.program_graph)

				# Track mutation history
				self.mutation_history.append({
					'action_idx': action_idx,
					'action_name': action_name,
					'iteration': len(self.mutation_history) + 1
				})

				if self.verbose:
					print(f"[DEBUG] Mutation successful. History length: {len(self.mutation_history)}")

				return self.current_working_graph
			else:
				if self.verbose:
					print(f"[DEBUG] Action {action_name} failed, returning unchanged graph")
				return self.current_working_graph

		except Exception as e:
			if self.verbose:
				print(f"Incremental mutation failed: {e}")
				import traceback
				traceback.print_exc()
			return self.current_working_graph if self.current_working_graph is not None else self.best_graph

	def _reset_working_graph(self):
		"""Reset working graph to best graph and clear mutation history"""
		self.current_working_graph = copy.deepcopy(self.best_graph)
		self.current_working_performance = self.best_performance_margin
		self.mutation_history = []
		self.stagnation_counter = 0
		if self.verbose:
			print("[DEBUG] Reset working graph to best graph")

	def _mutate_graph(self, base_graph, action_idx):
		"""
		Legacy mutation method - kept for compatibility.
		This method is now deprecated in favor of incremental mutation.
		"""
		try:
			# Copy the base graph
			self.graph_manager.program_graph = copy.deepcopy(base_graph)

			# Execute the selected action
			action = self.actions[action_idx]
			action()

			return copy.deepcopy(self.graph_manager.program_graph)
		except Exception as e:
			if self.verbose:
				print(f"Mutation failed: {e}")
			return base_graph

	def _save_best_graph(self):
		"""Saves information about the best graph"""
		try:
			best_info = {
				"performance_margin": self.best_performance_margin,
				"graph_nodes": self.best_graph.number_of_nodes(),
				"graph_edges": self.best_graph.number_of_edges(),
				"mutation_history_length": len(self.mutation_history),
				"stagnation_counter": self.stagnation_counter
			}

			info_file = os.path.join(self.output_dir, "best_graph_info.txt")
			with open(info_file, 'w') as f:
				for key, value in best_info.items():
					f.write(f"{key}: {value}\n")

			# Save detailed mutation history
			self._save_mutation_history()
		except Exception as e:
			if self.verbose:
				print(f"Failed to save best graph info: {e}")

	def _save_mutation_history(self):
		"""Save detailed mutation history for analysis"""
		try:
			history_file = os.path.join(self.output_dir, "mutation_history.txt")
			with open(history_file, 'w') as f:
				f.write(f"Best Performance Margin: {self.best_performance_margin:.3f}\n")
				f.write(f"Total Mutations Applied: {len(self.mutation_history)}\n")
				f.write(f"Current Stagnation Counter: {self.stagnation_counter}\n")
				f.write("=" * 50 + "\n")
				f.write("Mutation History:\n")

				for i, mutation in enumerate(self.mutation_history, 1):
					f.write(f"{i:3d}. Action {mutation['action_idx']:2d}: {mutation['action_name']}\n")

				if not self.mutation_history:
					f.write("No mutations applied yet.\n")

		except Exception as e:
			if self.verbose:
				print(f"Failed to save mutation history: {e}")

	def _print_mutation_summary(self):
		"""Print a summary of current mutation state"""
		if self.verbose:
			print(f"\n[MUTATION SUMMARY]")
			print(f"  Working graph nodes: {self.current_working_graph.number_of_nodes() if self.current_working_graph else 'N/A'}")
			print(f"  Working graph edges: {self.current_working_graph.number_of_edges() if self.current_working_graph else 'N/A'}")
			print(f"  Best graph nodes: {self.best_graph.number_of_nodes() if self.best_graph else 'N/A'}")
			print(f"  Best graph edges: {self.best_graph.number_of_edges() if self.best_graph else 'N/A'}")
			print(f"  Mutations applied: {len(self.mutation_history)}")
			print(f"  Stagnation counter: {self.stagnation_counter}/{self.max_stagnation}")

			if self.mutation_history:
				recent_count = min(3, len(self.mutation_history))
				recent_actions = [h['action_name'] for h in self.mutation_history[-recent_count:]]
				print(f"  Recent actions: {' -> '.join(recent_actions)}")
			print()


