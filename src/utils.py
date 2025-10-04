"""utilities for HLS BanditFuzz."""

import os, sys, shutil, datetime, pickle
from contextlib import contextmanager
from io import StringIO


class BanditFuzzUtils:
    def __init__(self, verbose=False, output_dir="./output"):
        self.verbose = verbose
        self.output_dir = output_dir
        self.timeout_dir = os.path.join(output_dir, "timeout_cases")
        os.makedirs(self.timeout_dir, exist_ok=True)

    @contextmanager
    def suppress_output(self):
        """Suppress stdout when not verbose."""
        if self.verbose:
            yield
        else:
            old = sys.stdout
            sys.stdout = StringIO()
            try:
                yield
            finally:
                sys.stdout = old

    def log(self, msg):
        """Log if verbose."""
        if self.verbose:
            print(f"[DEBUG] {msg}")

    def dump_timeout(self, graph):
        """Dump timeout case (good benchmark)."""
        try:
            aig = os.path.join(self.output_dir, "miter", "miter.aig")
            if not os.path.exists(aig):
                return
            
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = os.path.join(self.timeout_dir, f"timeout_{ts}")
            os.makedirs(folder, exist_ok=True)
            
            # Copy key files
            for item in ["benchmark_1.cpp", "benchmark_2.cpp", "miter"]:
                src = os.path.join(self.output_dir, item)
                if os.path.exists(src):
                    dst = os.path.join(folder, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
            
            # Save graph
            if graph:
                with open(os.path.join(folder, "graph.pkl"), 'wb') as f:
                    pickle.dump(graph, f)
            
            print(f"[TIMEOUT] Saved: {folder}")
        except:
            pass

    def print_summary(self, pool, target, successful, total):
        """Print final summary."""
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Completed: {successful}/{target} ({successful/total*100:.1f}% success)")
        print(f"Pool size: {len(pool)}")
        
        # Updated: unpack triple (graph, perf, action_count)
        perfs = [p for _, p, _ in pool if p != float('inf')]
        if perfs:
            print(f"Avg perf: {sum(perfs)/len(perfs):.3f}s, Best: {max(perfs):.3f}s")
            print(f"Good cases: {sum(1 for _, p, _ in pool if p == 3600.0)}")
        
        # Updated: node count statistics
        sizes = [g.number_of_nodes() for g, _, _ in pool]
        if sizes:
            print(f"Avg size: {sum(sizes)/len(sizes):.1f}, Range: {min(sizes)}-{max(sizes)}")
        
        # New: action count statistics
        actions = [a for _, _, a in pool]
        if actions:
            print(f"Avg actions: {sum(actions)/len(actions):.1f}, Range: {min(actions)}-{max(actions)}")
        
        print("="*60)