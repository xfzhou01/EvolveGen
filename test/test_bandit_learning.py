#!/usr/bin/env python3
"""
Test and observe the learning process of BanditFuzz
"""

import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('src')

from banditGen import HLSBanditFuzz

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class LearningObserver:
    """Class to observe and record the learning process"""
    
    def __init__(self):
        self.iteration_data = []
        self.strategy_choices = []
        self.action_choices = []
        self.performance_margins = []
        self.strategy_params = []
        self.action_params = []
    
    def record_iteration(self, iteration, strategy, action_idx, performance_margin, 
                        strategy_agent, action_agent, improved):
        """Record data for each iteration"""
        self.iteration_data.append({
            'iteration': iteration,
            'strategy': strategy,
            'action_idx': action_idx,
            'performance_margin': performance_margin,
            'improved': improved
        })
        
        self.strategy_choices.append(strategy)
        self.action_choices.append(action_idx)
        self.performance_margins.append(performance_margin)
        
        # Record the parameters of the agent
        self.strategy_params.append([
            strategy_agent.alpha_beta[0].copy(),
            strategy_agent.alpha_beta[1].copy()
        ])
        
        self.action_params.append([
            action_agent.alpha_beta[i].copy() for i in range(action_agent.n_actions)
        ])
    
    def plot_learning_progress(self, save_path="learning_analysis.png"):
        """Plot learning progress"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Performance marginal changes
        axes[0, 0].plot(self.performance_margins)
        axes[0, 0].set_title('Performance Margin Over Time')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Performance Margin')
        
        # 2. Policy selection distribution
        strategy_counts = [self.strategy_choices.count(0), self.strategy_choices.count(1)]
        axes[0, 1].bar(['Generate New', 'Mutate Best'], strategy_counts)
        axes[0, 1].set_title('Strategy Selection Distribution')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Action selection distribution
        action_counts = [self.action_choices.count(i) for i in range(4)]
        action_names = ['Add Input', 'Add Op', 'Add Loop', 'Add Branch']
        axes[0, 2].bar(action_names, action_counts)
        axes[0, 2].set_title('Action Selection Distribution')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Changes in policy agent parameters
        strategy_alpha_0 = [params[0][0] for params in self.strategy_params]
        strategy_alpha_1 = [params[1][0] for params in self.strategy_params]
        axes[1, 0].plot(strategy_alpha_0, label='Strategy 0 (Generate)')
        axes[1, 0].plot(strategy_alpha_1, label='Strategy 1 (Mutate)')
        axes[1, 0].set_title('Strategy Agent Alpha Parameters')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Alpha Value')
        axes[1, 0].legend()
        
        # 5. Changes in action agent parameters
        for i in range(4):
            action_alphas = [params[i][0] for params in self.action_params]
            axes[1, 1].plot(action_alphas, label=f'Action {i}')
        axes[1, 1].set_title('Action Agent Alpha Parameters')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Alpha Value')
        axes[1, 1].legend()
        
        # 6. Improvement rate
        improvements = [1 if data['improved'] else 0 for data in self.iteration_data]
        window_size = min(10, len(improvements))
        if len(improvements) >= window_size:
            moving_avg = np.convolve(improvements, np.ones(window_size)/window_size, mode='valid')
            axes[1, 2].plot(range(window_size-1, len(improvements)), moving_avg)
        axes[1, 2].set_title(f'Improvement Rate (Moving Avg, window={window_size})')
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Improvement Rate')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning analysis plot saved to: {save_path}")
        
    def save_data(self, save_path="learning_data.json"):
        """Save learning data"""
        data = {
            'iteration_data': self.iteration_data,
            'strategy_choices': self.strategy_choices,
            'action_choices': self.action_choices,
            'performance_margins': self.performance_margins
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        print(f"Learning data saved to: {save_path}")

def test_bandit_learning(max_iterations=20, verbose=True, run_full_pipeline=False):
    """Test the learning process of BanditFuzz"""
    print("=== BanditFuzz Learning Test ===")
    
    # Create an observer
    observer = LearningObserver()
    
    # Create a BanditFuzz instance
    output_dir = "./test_learning_output"
    os.makedirs(output_dir, exist_ok=True)
    
    bandit_fuzzer = HLSBanditFuzz(
        output_dir=output_dir,
        seed=42,
        verbose=verbose
    )
    
    # Set fewer iterations for testing
    bandit_fuzzer.max_iter = max_iterations
    
    print(f"Starting learning test for {max_iterations} iterations...")
    print("Initial agent states:")
    print(f"  Strategy Agent: {bandit_fuzzer.strategy_agent.alpha_beta}")
    print(f"  Action Agent: {bandit_fuzzer.action_agent.alpha_beta}")
    
    # Modify BanditFuzz's fuzz method to record the learning process
    original_fuzz = bandit_fuzzer.fuzz
    
    def monitored_fuzz():
        # Generate initial graph
        print("[INFO] Generating initial graph...")
        success = bandit_fuzzer.graph_manager.generate_random_graph(action_number_total=20)
        if not success:
            print("[ERROR] Failed to generate initial graph")
            return
        
        bandit_fuzzer.best_graph = bandit_fuzzer.graph_manager.program_graph.copy()
        bandit_fuzzer.best_performance_margin = -0.1  # Simulate initial performance
        
        print(f"[INFO] Initial performance margin: {bandit_fuzzer.best_performance_margin:.3f}")
        
        # Main loop
        for iteration in range(1, bandit_fuzzer.max_iter + 1):
            print(f"\n[INFO] Iteration {iteration}/{bandit_fuzzer.max_iter}")
            
            #Agent decision making
            strategy = bandit_fuzzer.strategy_agent.select_action()
            action_idx = bandit_fuzzer.action_agent.select_action() if strategy == 1 else -1
            
            # Choose between full pipeline or simulated evaluation
            if run_full_pipeline:
                # Run the complete HLS pipeline to get real performance data
                if strategy == 0:  # Generate new graph
                    print(f"[INFO] Generating new graph...")
                    success = bandit_fuzzer.graph_manager.generate_random_graph(action_number_total=20)
                    if not success:
                        print(f"[WARNING] Failed to generate new graph in iteration {iteration}")
                        performance_margin = float('-inf')
                        improved = False
                    else:
                        current_graph = bandit_fuzzer.graph_manager.program_graph.copy()
                        performance_margin, success = bandit_fuzzer.run_hls_pipeline_and_evaluate(current_graph)
                        improved = success and performance_margin > bandit_fuzzer.best_performance_margin
                        if improved:
                            bandit_fuzzer.best_graph = current_graph
                else:  # Mutate existing graph (strategy == 1)
                    print(f"[INFO] Mutating existing graph with action {action_idx}...")
                    # Apply the selected action to mutate the graph
                    bandit_fuzzer.graph_manager.program_graph = bandit_fuzzer.best_graph.copy()
                    action_success = bandit_fuzzer.actions[action_idx]()

                    if not action_success:
                        print(f"[WARNING] Action {action_idx} failed in iteration {iteration}")
                        performance_margin = float('-inf')
                        improved = False
                    else:
                        current_graph = bandit_fuzzer.graph_manager.program_graph.copy()
                        performance_margin, success = bandit_fuzzer.run_hls_pipeline_and_evaluate(current_graph)
                        improved = success and performance_margin > bandit_fuzzer.best_performance_margin
                        if improved:
                            bandit_fuzzer.best_graph = current_graph
            else:
                # Simulated performance evaluation for faster testing
                performance_margin = np.random.normal(-0.05, 0.02)  # Simulate performance data
                improved = performance_margin > bandit_fuzzer.best_performance_margin

            # Update best performance if improved
            if improved and performance_margin != float('-inf'):
                print(f"[IMPROVE] New best margin: {performance_margin:.3f} (was {bandit_fuzzer.best_performance_margin:.3f})")
                bandit_fuzzer.best_performance_margin = performance_margin
            
            # Rewards for Intelligents
            bandit_fuzzer.strategy_agent.reward(improved)
            if strategy == 1:  # The action weapon will be rewarded only when mutations are made
                bandit_fuzzer.action_agent.reward(improved)
            
            # Record data
            observer.record_iteration(
                iteration, strategy, action_idx, performance_margin,
                bandit_fuzzer.strategy_agent, bandit_fuzzer.action_agent, improved
            )
            
            print(f"[INFO] Strategy: {strategy}, Action: {action_idx}, Margin: {performance_margin:.3f}, Improved: {improved}")
    
    # Fuzzy test for running monitoring
    monitored_fuzz()
    
    print("\n=== Learning Result Analysis ===")
    
    #Analyze strategy selection
    strategy_counts = [observer.strategy_choices.count(0), observer.strategy_choices.count(1)]
    print(f"Strategy selection distribution: Generate New Graph={strategy_counts[0]}, Mutate Existing Graph={strategy_counts[1]}")
    
    #Analyze action selection
    action_counts = [observer.action_choices.count(i) for i in range(4)]
    action_names = ['Add Input', 'Add Op', 'Add Loop', 'Add Branch']
    print("Action selection distribution:")
    for name, count in zip(action_names, action_counts):
        print(f"  {name}: {count}")
    
    # Analytical performance improvements
    improvements = sum(1 for data in observer.iteration_data if data['improved'])
    print(f"Total improvements: {improvements}/{max_iterations} ({improvements/max_iterations*100:.1f}%)")
    
    # Final agent state
    print("\nFinal agent states:")
    print(f"  Strategy Agent: {bandit_fuzzer.strategy_agent.alpha_beta}")
    print(f"  Action Agent: {bandit_fuzzer.action_agent.alpha_beta}")
    
    # Save and draw results
    observer.save_data(os.path.join(output_dir, "learning_data.json"))
    observer.plot_learning_progress(os.path.join(output_dir, "learning_analysis.png"))
    
    return observer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test BanditFuzz learning process')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations to run')
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run full HLS pipeline (generates btor2 files, but slower)')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')

    args = parser.parse_args()

    if args.full_pipeline:
        print("Running with FULL HLS pipeline - this will generate btor2 files but may take longer...")
        observer = test_bandit_learning(
            max_iterations=args.iterations,
            verbose=args.verbose,
            run_full_pipeline=True
        )
    else:
        print("Running with SIMULATED evaluation - faster but no btor2 files generated...")
        observer = test_bandit_learning(
            max_iterations=args.iterations,
            verbose=args.verbose,
            run_full_pipeline=False
        )

    print(f"\nTest complete! Check the result files in ./test_learning_output/ directory.")
    if args.full_pipeline:
        print("BTOR2 files should be available in ./test_learning_output/btor2/ directory.")
