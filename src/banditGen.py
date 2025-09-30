"""Hybrid Evolutionary Bandit Fuzzing - Minimal Version."""

import os, time, subprocess, copy, random
from agents import ThompsonSampling
from random_graph_manager import RandomGraphManager
from vitis_hls_compiler import VitisHLSCompiler
from miter_generator import MiterGenerator
from yosys_compiler import YosysCompiler
from utils import BanditFuzzUtils


class HLSBanditFuzz:
    def __init__(self, output_dir="./output", seed=114514, verbose=False):
        # Components
        self.graph_mgr = RandomGraphManager(seed=seed)
        self.hls = VitisHLSCompiler(working_dir=output_dir)
        self.yosys = YosysCompiler()
        self.utils = BanditFuzzUtils(verbose=verbose, output_dir=output_dir)
        
        # State
        self.seed = seed
        self.gen_count = 0
        self.pool = []  # [(graph, perf), ...]
        
        # Agents
        self.actions = self.graph_mgr.bandit_action_list
        self.action_agent = ThompsonSampling(len(self.actions), 0.99, 10, 5)
        self.strategy_agent = ThompsonSampling(2, 0.99, 10, 5)  # 0=Evolve, 1=Inject
        
        # Config
        self.verbose = verbose
        self.out_dir = output_dir
        self.max_iter = 100
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
            graph, baseline = self._exec(strat)
            if graph is None:
                continue
            
            # Evaluate
            perf, status = self._eval(graph)
            if status == "RETRY" or not status:
                self._neg_reward(strat)
                continue
            
            # Update pool
            ok += 1
            self._update(strat, graph, perf, baseline)
        
        self.utils.print_summary(self.pool, self.max_iter, ok, total)

    def _exec(self, strat):
        """Execute strategy: 0=Evolve, 1=Inject."""
        if strat == 0:  # Evolve
            if not self.pool:
                return None, float('-inf')
            parent, parent_perf = random.choice(self.pool)
            print(f"  Parent: {parent.number_of_nodes()} nodes")
            child = self._mutate(parent)
            return child, parent_perf
        else:  # Inject
            avg = sum(g.number_of_nodes() for g, _ in self.pool) / len(self.pool) if self.pool else 100
            target = max(50, int(avg))
            if not self._gen(target):
                return None, float('-inf')
            fresh = copy.deepcopy(self.graph_mgr.program_graph)
            pool_avg = self._pool_avg()
            return fresh, pool_avg

    def _update(self, strat, graph, perf, baseline):
        """Update pool if improved."""
        improved = perf > baseline
        pstr = f"{perf:.3f}s" if perf != float('inf') else "timeout"
        bstr = f"{baseline:.3f}s" if baseline != float('inf') else "timeout"
        
        if improved:
            self.pool.append((graph, perf))
            print(f"  ✓ {pstr} > {bstr}, pool={len(self.pool)}")
            self.strategy_agent.reward(True)
            if strat == 0:
                self.action_agent.reward(True)
        else:
            print(f"  ✗ {pstr} <= {bstr}")
            self._neg_reward(strat)

    def _neg_reward(self, strat):
        """Give negative reward."""
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
        """Generate fresh graph with n actions."""
        try:
            with self.utils.suppress_output():
                self.gen_count += 1
                random.seed(self.seed + self.gen_count)
                self.graph_mgr.seed = self.seed + self.gen_count
                self.graph_mgr._reset_all()
                ok = self.graph_mgr.generate_random_graph(action_number_total=n)
            return ok and len(self.graph_mgr._get_op_node_list()) >= 3
        except:
            return False

    def _init(self):
        """Initialize pool with one valid graph."""
        for _ in range(5):
            if not self._gen(100):
                continue
            g = copy.deepcopy(self.graph_mgr.program_graph)
            perf, status = self._eval(g)
            if status == "RETRY":
                continue
            elif status:
                self.pool.append((g, perf))
                return True
        return False

    def _pool_avg(self):
        """Average pool performance (exclude timeouts)."""
        perfs = [p for _, p in self.pool if p != float('inf')]
        return sum(perfs) / len(perfs) if perfs else float('-inf')

    def _eval(self, graph):
        """Run HLS pipeline and evaluate."""
        try:
            # C++
            cpp = self._cpp(graph)
            if not cpp:
                return float('-inf'), False
            
            # HLS
            vs = self._hls(cpp)
            if not vs:
                return float('-inf'), False
            
            # Miter
            m = self._miter(vs)
            if not m:
                return float('-inf'), False
            elif m == "COMB":
                return float('-inf'), "RETRY"
            
            # AIG
            aig = self._aig(m)
            if not aig:
                return float('-inf'), False
            
            # rIC3
            r = self._ric3(aig)
            if r == "TIMEOUT":
                self.utils.dump_timeout(graph)
                return float('inf'), True
            
            return r, True
        except:
            return float('-inf'), False

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
        try:
            start = time.time()
            r = subprocess.run(
                ["../rIC3-code/target/release/rIC3", aig],
                capture_output=True, text=True, timeout=10
            )
            elapsed = time.time() - start
            return elapsed if "SAT" in r.stdout or "UNSAT" in r.stdout else float('inf')
        except subprocess.TimeoutExpired:
            return "TIMEOUT"
        except:
            return float('inf')