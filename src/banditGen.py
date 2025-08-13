import os, glob, random, time, uuid, subprocess, copy
from agents import ThompsonSampling
from random_graph_manager import RandomGraphManager
from vitis_hls_compiler import VitisHLSCompiler
from miter_generator import MiterGenerator
from yosys_compiler import YosysCompiler

class HLSBanditFuzz:
	def __init__(self, output_dir="./output", seed=42, verbose=False):
		# HLS工具链组件
		self.graph_manager = RandomGraphManager(seed=seed)
		self.hls_compiler = VitisHLSCompiler(working_dir=output_dir)
		self.yosys_compiler = YosysCompiler()
		# MiterGenerator将在需要时动态创建

		# BanditFuzz组件
		self.actions = self.graph_manager.bandit_action_list
		self.action_agent = ThompsonSampling(n_actions=len(self.actions))
		self.strategy_agent = ThompsonSampling(n_actions=2)  # 生成新图 vs 变异现有图

		# 状态管理
		self.best_graph = None
		self.best_performance_margin = float('-inf')  # 目标是最大化性能差异
		self.max_iter = 1000
		self.verbose = verbose

		# 路径配置
		self.output_dir = output_dir
		self.btor2_output_dir = os.path.join(output_dir, "btor2")
		self.generate_dir = "./generate"
		os.makedirs(self.output_dir, exist_ok=True)
		os.makedirs(self.btor2_output_dir, exist_ok=True)
		os.makedirs(self.generate_dir, exist_ok=True)

	def run_hls_pipeline_and_evaluate(self, graph):
		"""
		运行完整的HLS流程并评估性能差异
		返回: (performance_margin, success)
		"""
		try:
			# 1. 生成C++代码
			if self.verbose:
				print("[DEBUG] Step 1: Generating C++ code...")
			cpp_files = self._generate_cpp_from_graph(graph)
			if not cpp_files:
				if self.verbose:
					print("[ERROR] Step 1 failed: C++ generation")
				return float('-inf'), False

			# 2. HLS编译生成Verilog
			if self.verbose:
				print("[DEBUG] Step 2: HLS compilation...")
			verilog_files = self._compile_with_hls(cpp_files)
			if not verilog_files:
				if self.verbose:
					print("[ERROR] Step 2 failed: HLS compilation")
				return float('-inf'), False

			# 3. 生成Miter电路
			if self.verbose:
				print("[DEBUG] Step 3: Miter generation...")
			miter_file = self._generate_miter_circuit(verilog_files)
			if not miter_file:
				if self.verbose:
					print("[ERROR] Step 3 failed: Miter generation")
				return float('-inf'), False

			# 4. 转换为BTOR2格式
			if self.verbose:
				print("[DEBUG] Step 4: BTOR2 conversion...")
			btor2_file = self._convert_to_btor2(miter_file)
			if not btor2_file:
				if self.verbose:
					print("[ERROR] Step 4 failed: BTOR2 conversion")
				return float('-inf'), False

			# 5. 运行双求解器测试
			if self.verbose:
				print("[DEBUG] Step 5: Running solvers...")
			smt_sweeper_time = self._run_smt_sweeper(btor2_file)
			bitwuzla_time = self._run_bitwuzla(btor2_file)

			# 6. 计算性能差异 (目标: bitwuzla慢, smt-sweeper快)
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
		"""从图生成C++代码"""
		try:
			# 设置图管理器的图
			self.graph_manager.program_graph = graph

			# 生成两个版本的C++代码用于比较
			cpp_file_1 = os.path.join(self.output_dir, "benchmark_1.cpp")
			cpp_file_2 = os.path.join(self.output_dir, "benchmark_2.cpp")

			# 使用dump_cpp_comparsion方法生成两个版本
			self.graph_manager.dump_cpp_comparsion(cpp_file_1, cpp_file_2)

			# 检查文件是否真的生成了
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
		"""使用HLS编译器编译C++代码"""
		try:
			# 返回分组的Verilog文件，而不是混合列表
			verilog_files_groups = []
			for i, cpp_file in enumerate(cpp_files, 1):
				project_name = f"hls_project_{i}"

				# 使用不同的时钟周期，就像正常流程一样
				if i == 1:
					clock_period = self.graph_manager.cp_1
				elif i == 2:
					clock_period = self.graph_manager.cp_2
				else:
					clock_period = 10  # 默认值

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
		"""生成Miter电路"""
		try:
			# verilog_files_groups现在是一个列表的列表
			if len(verilog_files_groups) < 2:
				if self.verbose:
					print("Need at least 2 groups of Verilog files for miter generation")
				return None

			# 使用第一组和第二组Verilog文件
			verilog_files_1 = verilog_files_groups[0]
			verilog_files_2 = verilog_files_groups[1]

			if self.verbose:
				print(f"[DEBUG] Group 1 Verilog files: {verilog_files_1}")
				print(f"[DEBUG] Group 2 Verilog files: {verilog_files_2}")

			# 创建MiterGenerator
			merged_verilog_folder = os.path.join(self.output_dir, "merged_verilog")
			os.makedirs(merged_verilog_folder, exist_ok=True)

			miter_generator = MiterGenerator(
				verilog_file_path_list_1=verilog_files_1,
				verilog_file_path_list_2=verilog_files_2,
				merged_verilog_folder_path=merged_verilog_folder,
				top_name="top"
			)

			# 生成Miter电路，返回的是top模块名称
			kairos_top = miter_generator.generate_miter()

			# 返回包含miter.v文件的目录路径
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
		"""将Miter电路转换为BTOR2格式"""
		try:
			# miter_file应该是一个目录路径，包含miter.v文件
			# 如果miter_file是文件路径，我们需要获取其目录
			if os.path.isfile(miter_file):
				input_dir = os.path.dirname(miter_file)
			else:
				input_dir = miter_file

			if self.verbose:
				print(f"[DEBUG] Input directory for BTOR2 conversion: {input_dir}")
				print(f"[DEBUG] Files in input directory: {os.listdir(input_dir) if os.path.exists(input_dir) else 'Directory not found'}")

			# 调用转换脚本，使用正确的参数格式
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
				# 检查生成的BTOR2文件
				btor2_file = os.path.join(self.btor2_output_dir, "miter.btor2")
				if os.path.exists(btor2_file):
					return btor2_file
				else:
					# 查找任何.btor2文件
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
		"""运行SMT-Sweeper求解器 (Reference Solver - 期望快)"""
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
				# 保存生成的SMT文件到generate目录
				if result.stdout:
					timestamp = int(time.time() * 1000000)
					smt_file = os.path.join(self.generate_dir, f"{timestamp}.smt2")
					with open(smt_file, 'w') as f:
						f.write(result.stdout)
				return solve_time
			else:
				return float('inf')  # 求解失败

		except subprocess.TimeoutExpired:
			return float('inf')  # 超时
		except Exception as e:
			if self.verbose:
				print(f"SMT-Sweeper failed: {e}")
			return float('inf')

	def _run_bitwuzla(self, btor2_file):
		"""运行Bitwuzla求解器 (Target Solver - 期望慢)"""
		try:
			# 找到最新生成的SMT文件
			latest_smt_file = self._get_latest_smt_file()
			if not latest_smt_file:
				if self.verbose:
					print(f"No SMT file found for Bitwuzla, using BTOR2 file: {btor2_file}")
				# 如果没有SMT文件，直接使用BTOR2文件
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
			return float('inf')  # 超时
		except Exception as e:
			if self.verbose:
				print(f"Bitwuzla failed: {e}")
			return float('inf')

	def _get_latest_smt_file(self):
		"""获取generate目录中最新的SMT文件"""
		try:
			# 简化版本的文件查找
			smt_files = glob.glob(os.path.join(self.generate_dir, "*.smt2"))
			if smt_files:
				# 按修改时间排序，返回最新的
				latest_file = max(smt_files, key=os.path.getmtime)
				return latest_file
			return None
		except Exception:
			return None

	def fuzz(self):
		"""主要的BanditFuzz模糊测试循环"""
		print("[INFO] Starting HLS BanditFuzz...")

		# 生成初始图
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

		# 主循环
		for iteration in range(1, self.max_iter + 1):
			print(f"\n[INFO] Iteration {iteration}/{self.max_iter}")

			# 智能体2决定策略: 0=生成新图, 1=变异现有图
			strategy = self.strategy_agent.select_action()

			if strategy == 0:  # 生成新图
				print("[INFO] Generating new graph...")
				success = self.graph_manager.generate_random_graph(action_number_total=20)
				if success:
					new_graph = copy.deepcopy(self.graph_manager.program_graph)
				else:
					continue
			else:  # 变异现有图
				print("[INFO] Mutating existing graph...")
				action_idx = self.action_agent.select_action()
				new_graph = self._mutate_graph(self.best_graph, action_idx)

			# 评估新图
			performance_margin, success = self.run_hls_pipeline_and_evaluate(new_graph)

			if not success:
				# 评估失败，给负奖励
				self.strategy_agent.reward(False)
				if strategy == 1:  # 只有变异时才奖励动作智能体
					self.action_agent.reward(False)
				continue

			# 检查是否改进
			improved = performance_margin > self.best_performance_margin

			if improved:
				print(f"[IMPROVE] New best margin: {performance_margin:.3f} (was {self.best_performance_margin:.3f})")
				self.best_graph = new_graph
				self.best_performance_margin = performance_margin

				# 保存最佳图
				self._save_best_graph()

			# 奖励智能体
			self.strategy_agent.reward(improved)
			if strategy == 1:  # 只有变异时才奖励动作智能体
				self.action_agent.reward(improved)

			print(f"[INFO] Current margin: {performance_margin:.3f}, Best: {self.best_performance_margin:.3f}")

		print(f"\n[INFO] BanditFuzz completed. Best performance margin: {self.best_performance_margin:.3f}")

	def _mutate_graph(self, base_graph, action_idx):
		"""变异图结构"""
		try:
			# 复制基础图
			self.graph_manager.program_graph = copy.deepcopy(base_graph)

			# 执行选定的动作
			action = self.actions[action_idx]
			action()

			return copy.deepcopy(self.graph_manager.program_graph)
		except Exception as e:
			if self.verbose:
				print(f"Mutation failed: {e}")
			return base_graph

	def _save_best_graph(self):
		"""保存最佳图的信息"""
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


