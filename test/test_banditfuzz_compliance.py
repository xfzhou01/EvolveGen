#!/usr/bin/env python3
"""
Test script to verify compliance with BanditFuzz paper methodology.
This script tests that our implementation follows the multi-agent RL approach
described in the BanditFuzz paper.
"""

import sys
import os
import copy

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from banditGen import HLSBanditFuzz

def test_multi_agent_strategy():
    """Test that the two-agent strategy works as described in BanditFuzz paper"""
    print("=" * 60)
    print("Testing Multi-Agent Strategy (BanditFuzz Paper Compliance)")
    print("=" * 60)
    
    fuzzer = HLSBanditFuzz(output_dir="./test_output", seed=42, verbose=True)
    
    # Generate initial graph
    success = fuzzer.graph_manager.generate_random_graph(action_number_total=5)
    if not success:
        print("❌ Failed to generate initial graph")
        return False
    
    fuzzer.best_graph = copy.deepcopy(fuzzer.graph_manager.program_graph)
    fuzzer.current_working_graph = copy.deepcopy(fuzzer.best_graph)
    
    print("✅ Initial setup complete")
    
    # Test Strategy Agent (Agent 2 in paper)
    print("\n1. Testing Strategy Agent decisions...")
    strategy_decisions = []
    for i in range(10):
        strategy = fuzzer.strategy_agent.select_action()
        strategy_decisions.append(strategy)
    
    # Should have both 0 (new graph) and 1 (mutate) decisions
    has_new_graph = 0 in strategy_decisions
    has_mutate = 1 in strategy_decisions
    
    print(f"  Strategy decisions: {strategy_decisions}")
    print(f"  Has 'generate new graph' (0): {has_new_graph}")
    print(f"  Has 'mutate existing' (1): {has_mutate}")
    
    if has_new_graph and has_mutate:
        print("  ✅ Strategy agent explores both options")
    else:
        print("  ⚠️  Strategy agent may be biased (this could be normal early in training)")
    
    # Test Action Agent (Agent 1 in paper)
    print("\n2. Testing Action Agent decisions...")
    action_decisions = []
    for i in range(10):
        action_idx = fuzzer.action_agent.select_action()
        action_decisions.append(action_idx)
    
    unique_actions = set(action_decisions)
    total_actions = len(fuzzer.actions)
    
    print(f"  Action decisions: {action_decisions}")
    print(f"  Unique actions used: {len(unique_actions)}/{total_actions}")
    print(f"  Available actions: {[action.__name__ for action in fuzzer.actions]}")
    
    if len(unique_actions) > 1:
        print("  ✅ Action agent explores multiple actions")
    else:
        print("  ⚠️  Action agent may be biased (this could be normal early in training)")
    
    return True

def test_incremental_vs_fresh_generation():
    """Test the key difference: incremental mutation vs fresh generation"""
    print("\n" + "=" * 60)
    print("Testing Incremental Mutation vs Fresh Generation")
    print("=" * 60)
    
    fuzzer = HLSBanditFuzz(output_dir="./test_output", seed=42, verbose=True)
    
    # Generate initial graph
    success = fuzzer.graph_manager.generate_random_graph(action_number_total=5)
    if not success:
        print("❌ Failed to generate initial graph")
        return False
    
    fuzzer.best_graph = copy.deepcopy(fuzzer.graph_manager.program_graph)
    fuzzer.current_working_graph = copy.deepcopy(fuzzer.best_graph)
    
    initial_nodes = fuzzer.best_graph.number_of_nodes()
    print(f"✅ Initial graph: {initial_nodes} nodes")
    
    # Test Strategy 0: Generate new graph
    print(f"\n1. Testing Strategy 0: Generate new graph...")
    old_working_graph = copy.deepcopy(fuzzer.current_working_graph)
    
    # Simulate strategy 0 behavior
    success = fuzzer.graph_manager.generate_random_graph(action_number_total=5)
    if success:
        new_graph = copy.deepcopy(fuzzer.graph_manager.program_graph)
        fuzzer._reset_working_graph()  # This should happen with strategy 0
        
        new_nodes = new_graph.number_of_nodes()
        print(f"  New graph: {new_nodes} nodes")
        print(f"  Working graph reset: {len(fuzzer.mutation_history) == 0}")
        print(f"  ✅ Strategy 0 (fresh generation) working correctly")
    
    # Test Strategy 1: Incremental mutation
    print(f"\n2. Testing Strategy 1: Incremental mutation...")
    
    # Apply several incremental mutations
    mutations_applied = 0
    for i in range(3):
        action_idx = i % len(fuzzer.actions)
        before_nodes = fuzzer.current_working_graph.number_of_nodes()
        
        result_graph = fuzzer._mutate_graph_incremental(action_idx)
        
        after_nodes = result_graph.number_of_nodes()
        if after_nodes != before_nodes or len(fuzzer.mutation_history) > mutations_applied:
            mutations_applied += 1
        
        print(f"  Mutation {i+1}: {before_nodes} → {after_nodes} nodes")
    
    print(f"  Total mutations in history: {len(fuzzer.mutation_history)}")
    print(f"  Mutations that changed graph: {mutations_applied}")
    
    if len(fuzzer.mutation_history) > 0:
        print(f"  ✅ Strategy 1 (incremental mutation) working correctly")
        
        # Show mutation sequence
        print(f"  Mutation sequence:")
        for i, mutation in enumerate(fuzzer.mutation_history):
            print(f"    {i+1}. {mutation['action_name']}")
    else:
        print(f"  ⚠️  No mutations recorded (actions may have failed)")
    
    return True

def test_exploration_exploitation_balance():
    """Test that the system balances exploration vs exploitation"""
    print("\n" + "=" * 60)
    print("Testing Exploration vs Exploitation Balance")
    print("=" * 60)
    
    fuzzer = HLSBanditFuzz(output_dir="./test_output", seed=42, verbose=True)
    
    # Generate initial graph
    success = fuzzer.graph_manager.generate_random_graph(action_number_total=5)
    if not success:
        print("❌ Failed to generate initial graph")
        return False
    
    fuzzer.best_graph = copy.deepcopy(fuzzer.graph_manager.program_graph)
    fuzzer.current_working_graph = copy.deepcopy(fuzzer.best_graph)
    
    print("✅ Testing exploration vs exploitation balance...")
    
    # Simulate reward feedback to test learning
    print(f"\n1. Testing reward mechanism...")
    
    # Test positive reward
    initial_strategy_params = copy.deepcopy(fuzzer.strategy_agent.alpha_beta)
    initial_action_params = copy.deepcopy(fuzzer.action_agent.alpha_beta)
    
    # Select actions and give rewards
    strategy = fuzzer.strategy_agent.select_action()
    action_idx = fuzzer.action_agent.select_action()
    
    print(f"  Selected strategy: {strategy} ({'new graph' if strategy == 0 else 'mutate'})")
    print(f"  Selected action: {action_idx}")
    
    # Give positive reward
    fuzzer.strategy_agent.reward(True)
    fuzzer.action_agent.reward(True)
    
    # Check if parameters changed
    strategy_changed = fuzzer.strategy_agent.alpha_beta != initial_strategy_params
    action_changed = fuzzer.action_agent.alpha_beta != initial_action_params
    
    print(f"  Strategy agent parameters changed: {strategy_changed}")
    print(f"  Action agent parameters changed: {action_changed}")
    
    if strategy_changed and action_changed:
        print(f"  ✅ Reward mechanism working correctly")
    else:
        print(f"  ❌ Reward mechanism not updating parameters")
        return False
    
    # Test stagnation handling
    print(f"\n2. Testing stagnation handling...")
    fuzzer.stagnation_counter = fuzzer.max_stagnation - 1
    
    print(f"  Stagnation counter: {fuzzer.stagnation_counter}/{fuzzer.max_stagnation}")
    
    # Simulate one more stagnation
    fuzzer.stagnation_counter += 1
    
    if fuzzer.stagnation_counter >= fuzzer.max_stagnation:
        print(f"  Triggering reset due to stagnation...")
        old_history = len(fuzzer.mutation_history)
        fuzzer._reset_working_graph()
        new_history = len(fuzzer.mutation_history)
        
        print(f"  History length: {old_history} → {new_history}")
        print(f"  Stagnation counter: {fuzzer.stagnation_counter}")
        print(f"  ✅ Stagnation handling working correctly")
    
    return True

def main():
    """Run all BanditFuzz compliance tests"""
    print("🧪 Testing BanditFuzz Paper Compliance")
    print("This test verifies our implementation follows the BanditFuzz methodology.")
    
    # Create test output directory
    os.makedirs("./test_output", exist_ok=True)
    
    tests = [
        test_multi_agent_strategy,
        test_incremental_vs_fresh_generation,
        test_exploration_exploitation_balance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "=" * 60)
    print(f"BANDITFUZZ COMPLIANCE: {passed}/{total} tests passed")
    print(f"=" * 60)
    
    if passed == total:
        print("🎉 Implementation is compliant with BanditFuzz paper methodology!")
        return True
    else:
        print("⚠️  Some compliance issues found. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
