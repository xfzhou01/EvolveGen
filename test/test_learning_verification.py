#!/usr/bin/env python3
"""
Verify if the agents in BanditFuzz are actually learning
"""

import sys
import os
sys.path.append('src')

from agents.thompson import ThompsonSampling
import numpy as np

def test_thompson_sampling_learning():
    """Test if Thompson Sampling can learn"""
    print("=== Thompson Sampling Learning Verification ===")
    
    # Create an agent with 4 actions
    agent = ThompsonSampling(n_actions=4)
    
    print("Initial State:")
    for i in range(4):
        print(f"  Action {i}: alpha={agent.alpha_beta[i][0]:.3f}, beta={agent.alpha_beta[i][1]:.3f}")
    
    # Simulate learning process: assume action 2 is the best
    print("\nSimulating learning process (Action 2 always succeeds, others fail):")
    
    action_counts = [0, 0, 0, 0]
    success_counts = [0, 0, 0, 0]
    
    for iteration in range(50):
        # Select action
        action = agent.select_action()
        action_counts[action] += 1
        
        # Simulate reward: action 2 always succeeds, others fail
        if action == 2:
            reward = True
            success_counts[action] += 1
        else:
            reward = False
        
        # Give reward
        agent.reward(reward)
        
        # Print status every 10 iterations
        if (iteration + 1) % 10 == 0:
            print(f"\nIteration {iteration + 1}:")
            print(f"  Action selection counts: {action_counts}")
            print(f"  Success counts: {success_counts}")
            print("  Current parameters:")
            for i in range(4):
                print(f"    Action {i}: alpha={agent.alpha_beta[i][0]:.3f}, beta={agent.alpha_beta[i][1]:.3f}")
    
    print(f"\nFinal Result:")
    print(f"Action selection distribution: {[count/50 for count in action_counts]}")
    print("Has the agent learned to prefer Action 2?", action_counts[2] > max(action_counts[0], action_counts[1], action_counts[3]))

def test_bandit_fuzz_agents():
    """Test the agent configuration in BanditFuzz"""
    print("\n=== BanditFuzz Agent Configuration Verification ===")
    
    from banditGen import HLSBanditFuzz
    
    # Create BanditFuzz instance
    bandit_fuzzer = HLSBanditFuzz(
        output_dir="./test_learning_output",
        seed=42,
        verbose=True
    )
    
    print(f"Number of actions for action agent: {bandit_fuzzer.action_agent.n_actions}")
    print(f"Number of actions for strategy agent: {bandit_fuzzer.strategy_agent.n_actions}")
    print(f"Available actions list: {len(bandit_fuzzer.actions)}")
    
    # Test action selection
    print("\nTesting action selection:")
    for i in range(10):
        strategy = bandit_fuzzer.strategy_agent.select_action()
        action_idx = bandit_fuzzer.action_agent.select_action()
        print(f"  Iteration {i+1}: Strategy={strategy}, Action Index={action_idx}")
        
        # Simulate random reward
        reward = np.random.choice([True, False])
        bandit_fuzzer.strategy_agent.reward(reward)
        bandit_fuzzer.action_agent.reward(reward)

if __name__ == "__main__":
    test_thompson_sampling_learning()
    test_bandit_fuzz_agents()
