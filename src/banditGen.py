"""Hybrid Evolutionary Bandit Fuzzing"""

import os, time, subprocess, copy, random
from agents import ThompsonSampling
from random_graph_manager import RandomGraphManager
from vitis_hls_compiler import VitisHLSCompiler
from miter_generator import MiterGenerator
from yosys_compiler import YosysCompiler
from utils import BanditFuzzUtils


class HLSBanditFuzz:
    def __init__(self, output_dir="./output", seed=114513, verbose=False):
        # Components
        self.graph_mgr = RandomGraphManager(seed=seed)
        self.hls = VitisHLSCompiler(working_dir=output_dir)
        self.yosys = YosysCompiler()
        self.utils = BanditFuzzUtils(verbose=verbose, output_dir=output_dir)
        
        # State
        self.seed = seed
        self.gen_count = 0
        self.pool = []  # [(graph, perf, action_count), ...]
        
        # Agents
        self.actions = self.graph_mgr.bandit_action_list
        self.action_agent = ThompsonSampling(len(self.actions), 0.99, 10, 5)
        self.strategy_agent = ThompsonSampling(2, 0.99, 10, 5)  # 0=Evolve, 1=Inject
        
        # Config
        self.verbose = verbose
        self.out_dir = output_dir
        self.max_iter = 100
        self.timeout_value = 3600.0  # Use 3600s for timeout cases
        os.makedirs(output_dir, exist_ok=True)

    def fuzz(self):
        """Main loop."""
        print("="*60 + "\nHybrid Evolutionary Bandit Fuzzing\n" + "="*60)
        
        # Init pool
        if not self._init():
            print("Init failed")
            return
        
        print(f"Initial pool: {len(self.pool)} graphs")
        
        # Main loop
        ok, total = 0, 0
        while ok < self.max_iter:
            total += 1
            
            # Choose strategy
            strat = self.strategy_agent.select_action()
            print(f"\n[{ok+1}/{self.max_iter}] {'EVOLVE' if strat==0 else 'INJECT'}")
            
            # Execute
            graph, baseline, action_count = self._exec(strat)
            if graph is None:
                continue
            
            # Evaluate
            perf, status = self._eval(graph)
            if not status:
                # Technical failure - don't penalize agents, just skip
                print("  (evaluation failed, skipping)")
                continue
            
            # Update pool and give feedback based on performance
            ok += 1
            self._update(strat, graph, perf, baseline, action_count)
        
        self.utils.print_summary(self.pool, self.max_iter, ok, total)

    def _exec(self, strat):
        """Execute strategy: 0=Evolve, 1=Inject."""
        if strat == 0:  # Evolve
            if not self.pool:
                return None, -1, 0
            # Mask out timeout cases (>= timeout_value)
            pool_case2evolve = [(g, p, a) for g, p, a in self.pool if p < self.timeout_value]
            if not pool_case2evolve:
                return None, -1, 0
            parent, parent_perf, parent_actions = random.choice(pool_case2evolve)
            print(f"  Parent: {parent.number_of_nodes()} nodes, {parent_actions} actions, perf={parent_perf:.3f}s")
            child = self._mutate(parent)
            return child, parent_perf, parent_actions + 1
        else:  # Inject
            avg = sum(a for _, _, a in self.pool) / len(self.pool) if self.pool else 100
            target = max(30, int(avg)) 
            if not self._gen(target):
                return None, -1, 0
            fresh = copy.deepcopy(self.graph_mgr.program_graph)
            pool_avg = self._pool_avg()
            return fresh, pool_avg, target

    def _update(self, strat, graph, perf, baseline, action_count):
        """Update pool if improved, and give feedback based on performance."""
        improved = perf > baseline
        pstr = f"{perf:.3f}s" if perf < self.timeout_value else "timeout"
        bstr = f"{baseline:.3f}s" if baseline >= 0 and baseline < self.timeout_value else "timeout" if baseline >= self.timeout_value else "none"
        
        if improved:
            self.pool.append((graph, perf, action_count))
            print(f"  ✓ {pstr} > {bstr}, pool={len(self.pool)}, actions={action_count}")
            self.strategy_agent.reward(True)
            if strat == 0:
                self.action_agent.reward(True)
        else:
            print(f"  ✗ {pstr} <= {bstr}")
            self.strategy_agent.reward(False)
            if strat == 0:
                self.action_agent.reward(False)

    def _mutate(self, parent):
        """Mutate parent graph."""
        self.graph_mgr.program_graph = copy.deepcopy(parent)
        action_idx = self.action_agent.select_action()
        action = self.actions[action_idx]
        if action():
            self.graph_mgr._make_single_output()
        return copy.deepcopy(self.graph_mgr.program_graph)

    def _gen(self, n):
        """Generate fresh graph with n actions.
        
        First generation (init) uses fixed seed for reproducibility.
        Subsequent generations use truly random seeds.
        """
        try:
            with self.utils.suppress_output():
                self.gen_count += 1
                
                # Only first generation uses fixed seed
                if self.gen_count == 1:
                    random.seed(self.seed)
                    self.graph_mgr.seed = self.seed
                    print(f"[Gen {self.gen_count}] Using fixed seed: {self.seed}")
                else:
                    # Use truly random seed (system time-based)
                    random_seed = int(time.time() * 1000000) % (2**31)
                    random.seed(random_seed)
                    self.graph_mgr.seed = random_seed
                    if self.verbose:
                        print(f"[Gen {self.gen_count}] Using random seed: {random_seed}")
                
                self.graph_mgr._reset_all()
                ok = self.graph_mgr.generate_random_graph(action_number_total=n)
            return ok and len(self.graph_mgr._get_op_node_list()) >= 3
        except:
            return False

    def _init(self):
        """Initialize pool with one valid graph."""
        for _ in range(5):
            initial_actions = 20
            if not self._gen(initial_actions):
                continue
            g = copy.deepcopy(self.graph_mgr.program_graph)
            print(f"Begin eval")
            perf, status = self._eval(g)
            if not status:
                print(f"Eval failed, retry...")
                continue
            print(f"Success (perf={perf:.3f}s, actions={initial_actions})...Add to pool")
            self.pool.append((g, perf, initial_actions))
            return True
        return False

    def _pool_avg(self):
        """Average pool performance (exclude timeouts)."""
        perfs = [p for _, p, _ in self.pool if p < self.timeout_value]
        return sum(perfs) / len(perfs) if perfs else 0.1

    def _eval(self, graph):
        """Run HLS pipeline and evaluate.
        
        Returns:
            (perf, status) where:
            - perf: -1 for failure, 0.0-10.0 for fast solve, 3600.0 for timeout
            - status: False for failure, True for success (including timeout)
        """
        try:
            # C++
            cpp = self._cpp(graph)
            if not cpp:
                return -1, False
            
            # HLS
            vs = self._hls(cpp)
            if not vs:
                return -1, False
            
            # Miter
            m = self._miter(vs)
            if not m:
                return -1, False
            elif m == "COMB":
                # Combinational circuit, can't verify - retry
                return -1, False
            
            # AIG
            aig = self._aig(m)
            if not aig:
                return -1, False
            
            # rIC3
            perf = self._ric3(aig)
            if perf < 0:
                # Solver error
                return -1, False
            elif perf >= self.timeout_value:
                # Timeout - this is good!
                self.utils.dump_timeout(graph)
                return perf, True
            else:
                # Fast solve
                return perf, True
        except:
            return -1, False

    def _cpp(self, g):
        try:
            with self.utils.suppress_output():
                self.graph_mgr.program_graph = g
                f1, f2 = f"{self.out_dir}/b1.cpp", f"{self.out_dir}/b2.cpp"
                self.graph_mgr.dump_cpp_comparsion(f1, f2)
            return [f1, f2] if os.path.exists(f1) and os.path.exists(f2) else None
        except:
            return None

    def _hls(self, cpps):
        try:
            res = []
            cps = [self.graph_mgr.cp_1, self.graph_mgr.cp_2]
            for i, cpp in enumerate(cpps):
                with self.utils.suppress_output():
                    r = self.hls.compile(
                        project_name=f"p{i+1}",
                        top_name="top",
                        clock_period=cps[i] if i < len(cps) else 10,
                        cpp_file_list=[cpp]
                    )
                if r["success"]:
                    res.append(r["verilog_files"])
                else:
                    return None
            return res
        except:
            return None

    def _miter(self, vgroups):
        try:
            if len(vgroups) < 2:
                return None
            v1, v2 = vgroups[0], vgroups[1]
            mdir = f"{self.out_dir}/merged_verilog"
            os.makedirs(mdir, exist_ok=True)
            with self.utils.suppress_output():
                mg = MiterGenerator(v1, v2, mdir, "top")
                try:
                    mg.generate_miter(insert_assertions=False)
                    return mdir
                except ValueError as e:
                    if "ap_rst" in str(e) or "ap_clk" in str(e):
                        return "COMB"
                    raise
        except:
            return None

    def _aig(self, mdir):
        try:
            mfile = f"{mdir}/miter.v"
            if not os.path.exists(mfile):
                return None
            adir = f"{self.out_dir}/miter"
            os.makedirs(adir, exist_ok=True)
            afile = f"{adir}/miter.aig"
            with self.utils.suppress_output():
                self.yosys.execute(mfile, adir, afile, "top_A_times_top_B")
            return afile if os.path.exists(afile) else None
        except:
            return None

    def _ric3(self, aig):
        """Run rIC3 solver.
        
        Returns:
            - elapsed time (0.0-10.0) if solved
            - 3600.0 if timeout or no result
            - -1 if error
        """
        try:
            start = time.time()
            r = subprocess.run(
                ["../rIC3-code/target/release/rIC3", aig],
                capture_output=True, text=True, timeout=10
            )
            elapsed = time.time() - start
            
            # Check if solver gave a result
            if "SAT" in r.stdout or "UNSAT" in r.stdout:
                return elapsed
            
            # Solver ran but gave no result - treat as timeout
            return 3600.0
            
        except subprocess.TimeoutExpired:
            # Timeout after 10s - treat as very hard case
            return 3600.0
        except:
            # Other errors (solver crash, file not found, etc)
            return -1