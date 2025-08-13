#!/usr/bin/env python3
"""
测试BanditFuzz HLS基准生成器的简化脚本
"""

import sys
import os
sys.path.append('src')

from banditGen import HLSBanditFuzz

def test_bandit_fuzz():
    """测试BanditFuzz功能"""
    print("[INFO] Testing HLS BanditFuzz...")
    
    # 创建输出目录
    output_dir = "./test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化BanditFuzz
    bandit_fuzzer = HLSBanditFuzz(
        output_dir=output_dir,
        seed=123,
        verbose=True
    )
    
    # 设置较少的迭代次数用于测试
    bandit_fuzzer.max_iter = 5
    
    try:
        # 运行BanditFuzz
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
