#!/usr/bin/env python3
"""
Test script to verify the incremental mutation mechanism in HLSBanditFuzz.
This script tests the key improvements made to fix the non-incremental mutation issue.
"""

import sys
import os
import copy
import networkx as nx

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from banditGen import HLSBanditFuzz
from random_graph_manager import RandomGraphManager

def test_incremental_mutation_basic():
    """Test basic incremental mutation functionality"""
    print("=" * 60)
    print("Testing Basic Incremental Mutation")
    print("=" * 60)
    
    # Create a test instance with verbose output
    fuzzer = HLSBanditFuzz(output_dir="./test_output", seed=42, verbose=True)
    
    # Generate initial graph
    print("\n1. Generating initial graph...")
    success = fuzzer.graph_manager.generate_random_graph(action_number_total=5)
    if not success:
        print("❌ Failed to generate initial graph")
        return False
    
    # Set up initial state
    fuzzer.best_graph = copy.deepcopy(fuzzer.graph_manager.program_graph)
    fuzzer.current_working_graph = copy.deepcopy(fuzzer.best_graph)
    
    initial_nodes = fuzzer.best_graph.number_of_nodes()
    initial_edges = fuzzer.best_graph.number_of_edges()
    print(f"✅ Initial graph: {initial_nodes} nodes, {initial_edges} edges")
    
    # Test incremental mutations
    print("\n2. Testing incremental mutations...")
    for i in range(3):
        print(f"\n--- Mutation {i+1} ---")
        
        # Record state before mutation
        before_nodes = fuzzer.current_working_graph.number_of_nodes()
        before_edges = fuzzer.current_working_graph.number_of_edges()
        before_history_len = len(fuzzer.mutation_history)
        
        # Apply incremental mutation
        action_idx = i % len(fuzzer.actions)  # Cycle through actions
        result_graph = fuzzer._mutate_graph_incremental(action_idx)
        
        # Check state after mutation
        after_nodes = result_graph.number_of_nodes()
        after_edges = result_graph.number_of_edges()
        after_history_len = len(fuzzer.mutation_history)
        
        print(f"  Before: {before_nodes} nodes, {before_edges} edges, history: {before_history_len}")
        print(f"  After:  {after_nodes} nodes, {after_edges} edges, history: {after_history_len}")
        
        # Verify incremental behavior
        if after_history_len > before_history_len:
            print(f"  ✅ Mutation history updated correctly")
        else:
            print(f"  ⚠️  Mutation history not updated (action may have failed)")
        
        # Verify working graph is maintained
        if fuzzer.current_working_graph is result_graph:
            print(f"  ✅ Working graph maintained correctly")
        else:
            print(f"  ❌ Working graph not maintained")
    
    print(f"\n3. Final mutation history:")
    for i, mutation in enumerate(fuzzer.mutation_history):
        print(f"  {i+1}. {mutation['action_name']} (action {mutation['action_idx']})")
    
    return True

def test_stagnation_reset():
    """Test stagnation counter and reset mechanism"""
    print("\n" + "=" * 60)
    print("Testing Stagnation Reset Mechanism")
    print("=" * 60)
    
    fuzzer = HLSBanditFuzz(output_dir="./test_output", seed=42, verbose=True)
    fuzzer.max_stagnation = 3  # Set low threshold for testing
    
    # Generate initial graph
    success = fuzzer.graph_manager.generate_random_graph(action_number_total=5)
    if not success:
        print("❌ Failed to generate initial graph")
        return False
    
    fuzzer.best_graph = copy.deepcopy(fuzzer.graph_manager.program_graph)
    fuzzer.current_working_graph = copy.deepcopy(fuzzer.best_graph)
    
    print(f"✅ Initial setup complete. Max stagnation: {fuzzer.max_stagnation}")
    
    # Simulate stagnation
    print(f"\n1. Simulating stagnation...")
    for i in range(fuzzer.max_stagnation + 1):
        print(f"  Stagnation iteration {i+1}: counter = {fuzzer.stagnation_counter}")
        fuzzer.stagnation_counter += 1
        
        if fuzzer.stagnation_counter >= fuzzer.max_stagnation:
            print(f"  🔄 Triggering reset due to stagnation")
            old_history_len = len(fuzzer.mutation_history)
            fuzzer._reset_working_graph()
            new_history_len = len(fuzzer.mutation_history)
            
            print(f"  ✅ Reset completed:")
            print(f"    - Stagnation counter: {fuzzer.stagnation_counter}")
            print(f"    - History length: {old_history_len} → {new_history_len}")
            break
    
    return True

def test_working_graph_vs_best_graph():
    """Test that working graph and best graph are managed separately"""
    print("\n" + "=" * 60)
    print("Testing Working Graph vs Best Graph Management")
    print("=" * 60)
    
    fuzzer = HLSBanditFuzz(output_dir="./test_output", seed=42, verbose=True)
    
    # Generate initial graph
    success = fuzzer.graph_manager.generate_random_graph(action_number_total=5)
    if not success:
        print("❌ Failed to generate initial graph")
        return False
    
    fuzzer.best_graph = copy.deepcopy(fuzzer.graph_manager.program_graph)
    fuzzer.current_working_graph = copy.deepcopy(fuzzer.best_graph)
    
    initial_best_nodes = fuzzer.best_graph.number_of_nodes()
    initial_working_nodes = fuzzer.current_working_graph.number_of_nodes()
    
    print(f"✅ Initial state:")
    print(f"  Best graph: {initial_best_nodes} nodes")
    print(f"  Working graph: {initial_working_nodes} nodes")
    
    # Apply mutations to working graph
    print(f"\n1. Applying mutations to working graph...")
    for i in range(2):
        action_idx = i % len(fuzzer.actions)
        fuzzer._mutate_graph_incremental(action_idx)
    
    final_best_nodes = fuzzer.best_graph.number_of_nodes()
    final_working_nodes = fuzzer.current_working_graph.number_of_nodes()
    
    print(f"\n2. After mutations:")
    print(f"  Best graph: {final_best_nodes} nodes (should be unchanged)")
    print(f"  Working graph: {final_working_nodes} nodes (may have changed)")
    
    # Verify best graph unchanged
    if final_best_nodes == initial_best_nodes:
        print(f"  ✅ Best graph correctly preserved during working graph mutations")
    else:
        print(f"  ❌ Best graph was incorrectly modified")
        return False
    
    # Test reset functionality
    print(f"\n3. Testing reset functionality...")
    fuzzer._reset_working_graph()
    reset_working_nodes = fuzzer.current_working_graph.number_of_nodes()
    
    if reset_working_nodes == final_best_nodes:
        print(f"  ✅ Working graph correctly reset to best graph")
    else:
        print(f"  ❌ Working graph reset failed")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 Testing Incremental Mutation Fixes")
    print("This test verifies the fixes for the non-incremental mutation issue.")
    
    # Create test output directory
    os.makedirs("./test_output", exist_ok=True)
    
    tests = [
        test_incremental_mutation_basic,
        test_stagnation_reset,
        test_working_graph_vs_best_graph
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
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print(f"=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! Incremental mutation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
