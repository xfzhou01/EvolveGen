#!/usr/bin/env python3
"""
验证BanditFuzz中的智能体是否真的在学习
"""

import sys
import os
sys.path.append('src')

from agents.thompson import ThompsonSampling
import numpy as np

def test_thompson_sampling_learning():
    """测试Thompson Sampling是否能学习"""
    print("=== Thompson Sampling学习验证 ===")
    
    # 创建一个有4个动作的智能体
    agent = ThompsonSampling(n_actions=4)
    
    print("初始状态:")
    for i in range(4):
        print(f"  动作{i}: alpha={agent.alpha_beta[i][0]:.3f}, beta={agent.alpha_beta[i][1]:.3f}")
    
    # 模拟学习过程：假设动作2是最好的
    print("\n模拟学习过程（动作2总是成功，其他动作失败）:")
    
    action_counts = [0, 0, 0, 0]
    success_counts = [0, 0, 0, 0]
    
    for iteration in range(50):
        # 选择动作
        action = agent.select_action()
        action_counts[action] += 1
        
        # 模拟奖励：动作2总是成功，其他失败
        if action == 2:
            reward = True
            success_counts[action] += 1
        else:
            reward = False
        
        # 给予奖励
        agent.reward(reward)
        
        # 每10次迭代打印一次状态
        if (iteration + 1) % 10 == 0:
            print(f"\n迭代 {iteration + 1}:")
            print(f"  动作选择次数: {action_counts}")
            print(f"  成功次数: {success_counts}")
            print("  当前参数:")
            for i in range(4):
                print(f"    动作{i}: alpha={agent.alpha_beta[i][0]:.3f}, beta={agent.alpha_beta[i][1]:.3f}")
    
    print(f"\n最终结果:")
    print(f"动作选择分布: {[count/50 for count in action_counts]}")
    print("智能体是否学会偏好动作2？", action_counts[2] > max(action_counts[0], action_counts[1], action_counts[3]))

def test_bandit_fuzz_agents():
    """测试BanditFuzz中的智能体配置"""
    print("\n=== BanditFuzz智能体配置验证 ===")
    
    from banditGen import HLSBanditFuzz
    
    # 创建BanditFuzz实例
    bandit_fuzzer = HLSBanditFuzz(
        output_dir="./test_learning_output",
        seed=42,
        verbose=True
    )
    
    print(f"动作智能体动作数量: {bandit_fuzzer.action_agent.n_actions}")
    print(f"策略智能体动作数量: {bandit_fuzzer.strategy_agent.n_actions}")
    print(f"可用动作列表: {len(bandit_fuzzer.actions)}")
    
    # 测试动作选择
    print("\n测试动作选择:")
    for i in range(10):
        strategy = bandit_fuzzer.strategy_agent.select_action()
        action_idx = bandit_fuzzer.action_agent.select_action()
        print(f"  迭代{i+1}: 策略={strategy}, 动作索引={action_idx}")
        
        # 模拟随机奖励
        reward = np.random.choice([True, False])
        bandit_fuzzer.strategy_agent.reward(reward)
        bandit_fuzzer.action_agent.reward(reward)

if __name__ == "__main__":
    test_thompson_sampling_learning()
    test_bandit_fuzz_agents()
