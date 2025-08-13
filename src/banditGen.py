import os, glob, random, time, uuid, subprocess, copy
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
		self.action_agent = ThompsonSampling(n_actions=len(self.actions))
		self.strategy_agent = ThompsonSampling(n_actions=2)  # Generate new graph vs. Mutate existing graph

		# State management
		self.best_graph = None
		self.best_performance_margin = float('-inf')  # Goal is to maximize performance difference
		self.max_iter = 1000
		self.verbose = verbose

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

		# Main loop
		for iteration in range(1, self.max_iter + 1):
			print(f"\n[INFO] Iteration {iteration}/{self.max_iter}")

			# Agent 2 decides strategy: 0=Generate new graph, 1=Mutate existing graph
			strategy = self.strategy_agent.select_action()

			if strategy == 0:  # Generate new graph
				print("[INFO] Generating new graph...")
				success = self.graph_manager.generate_random_graph(action_number_total=20)
				if success:
					new_graph = copy.deepcopy(self.graph_manager.program_graph)
				else:
					continue
			else:  # Mutate existing graph
				print("[INFO] Mutating existing graph...")
				action_idx = self.action_agent.select_action()
				new_graph = self._mutate_graph(self.best_graph, action_idx)

			# Evaluate the new graph
			performance_margin, success = self.run_hls_pipeline_and_evaluate(new_graph)

			if not success:
				# Evaluation failed, give negative reward
				self.strategy_agent.reward(False)
				if strategy == 1:  # Only reward action agent if mutation
					self.action_agent.reward(False)
				continue

			# Check for improvement
			improved = performance_margin > self.best_performance_margin

			if improved:
				print(f"[IMPROVE] New best margin: {performance_margin:.3f} (was {self.best_performance_margin:.3f})")
				self.best_graph = new_graph
				self.best_performance_margin = performance_margin

				# Save best graph
				self._save_best_graph()

			# Reward agents
			self.strategy_agent.reward(improved)
			if strategy == 1:  # Only reward action agent if mutation
				self.action_agent.reward(improved)

			print(f"[INFO] Current margin: {performance_margin:.3f}, Best: {self.best_performance_margin:.3f}")

		print(f"\n[INFO] BanditFuzz completed. Best performance margin: {self.best_performance_margin:.3f}")

	def _mutate_graph(self, base_graph, action_idx):
		"""Mutates the graph structure"""
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
			}

			info_file = os.path.join(self.output_dir, "best_graph_info.txt")
			with open(info_file, 'w') as f:
				for key, value in best_info.items():
					f.write(f"{key}: {value}\n")
		except Exception as e:
			if self.verbose:
				print(f"Failed to save best graph info: {e}")


