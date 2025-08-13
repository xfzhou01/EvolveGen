#!/usr/bin/env python3
"""
Simplified scripts for testing BanditFuzz HLS benchmark generator
"""

import sys
import os
sys.path.append('src')

from banditGen import HLSBanditFuzz

def test_bandit_fuzz():
    """Test the BanditFuzz function"""
    print("[INFO] Testing HLS BanditFuzz...")
    
    # Create an output directory
    output_dir = "./test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize BanditFuzz
    bandit_fuzzer = HLSBanditFuzz(
        output_dir=output_dir,
        seed=123,
        verbose=True
    )
    
    # Set fewer iterations for testing
    bandit_fuzzer.max_iter = 5
    
    try:
        # Run BanditFuzz
        bandit_fuzzer.fuzz()
        print("[SUCCESS] BanditFuzz test completed")
        return True
    except Exception as e:
        print(f"[ERROR] BanditFuzz test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bandit_fuzz()
    sys.exit(0 if success else 1)
