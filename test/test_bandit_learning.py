#!/usr/bin/env python3
"""
测试和观察BanditFuzz的学习过程
"""

import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('src')

from banditGen import HLSBanditFuzz

class LearningObserver:
    """观察和记录学习过程的类"""
    
    def __init__(self):
        self.iteration_data = []
        self.strategy_choices = []
        self.action_choices = []
        self.performance_margins = []
        self.strategy_params = []
        self.action_params = []
    
    def record_iteration(self, iteration, strategy, action_idx, performance_margin, 
                        strategy_agent, action_agent, improved):
        """记录每次迭代的数据"""
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
        
        # 记录智能体参数
        self.strategy_params.append([
            strategy_agent.alpha_beta[0].copy(),
            strategy_agent.alpha_beta[1].copy()
        ])
        
        self.action_params.append([
            action_agent.alpha_beta[i].copy() for i in range(action_agent.n_actions)
        ])
    
    def plot_learning_progress(self, save_path="learning_analysis.png"):
        """绘制学习进度图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 性能边际变化
        axes[0, 0].plot(self.performance_margins)
        axes[0, 0].set_title('Performance Margin Over Time')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Performance Margin')
        
        # 2. 策略选择分布
        strategy_counts = [self.strategy_choices.count(0), self.strategy_choices.count(1)]
        axes[0, 1].bar(['Generate New', 'Mutate Best'], strategy_counts)
        axes[0, 1].set_title('Strategy Selection Distribution')
        axes[0, 1].set_ylabel('Count')
        
        # 3. 动作选择分布
        action_counts = [self.action_choices.count(i) for i in range(4)]
        action_names = ['Add Input', 'Add Op', 'Add Loop', 'Add Branch']
        axes[0, 2].bar(action_names, action_counts)
        axes[0, 2].set_title('Action Selection Distribution')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. 策略智能体参数变化
        strategy_alpha_0 = [params[0][0] for params in self.strategy_params]
        strategy_alpha_1 = [params[1][0] for params in self.strategy_params]
        axes[1, 0].plot(strategy_alpha_0, label='Strategy 0 (Generate)')
        axes[1, 0].plot(strategy_alpha_1, label='Strategy 1 (Mutate)')
        axes[1, 0].set_title('Strategy Agent Alpha Parameters')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Alpha Value')
        axes[1, 0].legend()
        
        # 5. 动作智能体参数变化
        for i in range(4):
            action_alphas = [params[i][0] for params in self.action_params]
            axes[1, 1].plot(action_alphas, label=f'Action {i}')
        axes[1, 1].set_title('Action Agent Alpha Parameters')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Alpha Value')
        axes[1, 1].legend()
        
        # 6. 改进率
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
        print(f"学习分析图保存到: {save_path}")
        
    def save_data(self, save_path="learning_data.json"):
        """保存学习数据"""
        data = {
            'iteration_data': self.iteration_data,
            'strategy_choices': self.strategy_choices,
            'action_choices': self.action_choices,
            'performance_margins': self.performance_margins
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"学习数据保存到: {save_path}")

def test_bandit_learning(max_iterations=20, verbose=True):
    """测试BanditFuzz的学习过程"""
    print("=== BanditFuzz学习测试 ===")
    
    # 创建观察器
    observer = LearningObserver()
    
    # 创建BanditFuzz实例
    output_dir = "./test_learning_output"
    os.makedirs(output_dir, exist_ok=True)
    
    bandit_fuzzer = HLSBanditFuzz(
        output_dir=output_dir,
        seed=42,
        verbose=verbose
    )
    
    # 设置较少的迭代次数用于测试
    bandit_fuzzer.max_iter = max_iterations
    
    print(f"开始{max_iterations}次迭代的学习测试...")
    print("初始智能体状态:")
    print(f"  策略智能体: {bandit_fuzzer.strategy_agent.alpha_beta}")
    print(f"  动作智能体: {bandit_fuzzer.action_agent.alpha_beta}")
    
    # 修改BanditFuzz的fuzz方法来记录学习过程
    original_fuzz = bandit_fuzzer.fuzz
    
    def monitored_fuzz():
        # 生成初始图
        print("[INFO] Generating initial graph...")
        success = bandit_fuzzer.graph_manager.generate_random_graph(action_number_total=20)
        if not success:
            print("[ERROR] Failed to generate initial graph")
            return
        
        bandit_fuzzer.best_graph = bandit_fuzzer.graph_manager.program_graph.copy()
        bandit_fuzzer.best_performance_margin = -0.1  # 模拟初始性能
        
        print(f"[INFO] Initial performance margin: {bandit_fuzzer.best_performance_margin:.3f}")
        
        # 主循环
        for iteration in range(1, bandit_fuzzer.max_iter + 1):
            print(f"\n[INFO] Iteration {iteration}/{bandit_fuzzer.max_iter}")
            
            # 智能体决策
            strategy = bandit_fuzzer.strategy_agent.select_action()
            action_idx = bandit_fuzzer.action_agent.select_action() if strategy == 1 else -1
            
            # 模拟性能评估（实际中这里会运行完整的HLS流程）
            # 为了测试，我们模拟一个性能边际
            performance_margin = np.random.normal(-0.05, 0.02)  # 模拟性能数据
            
            # 检查是否改进
            improved = performance_margin > bandit_fuzzer.best_performance_margin
            
            if improved:
                print(f"[IMPROVE] New best margin: {performance_margin:.3f} (was {bandit_fuzzer.best_performance_margin:.3f})")
                bandit_fuzzer.best_performance_margin = performance_margin
            
            # 奖励智能体
            bandit_fuzzer.strategy_agent.reward(improved)
            if strategy == 1:  # 只有变异时才奖励动作智能体
                bandit_fuzzer.action_agent.reward(improved)
            
            # 记录数据
            observer.record_iteration(
                iteration, strategy, action_idx, performance_margin,
                bandit_fuzzer.strategy_agent, bandit_fuzzer.action_agent, improved
            )
            
            print(f"[INFO] Strategy: {strategy}, Action: {action_idx}, Margin: {performance_margin:.3f}, Improved: {improved}")
    
    # 运行监控的模糊测试
    monitored_fuzz()
    
    print("\n=== 学习结果分析 ===")
    
    # 分析策略选择
    strategy_counts = [observer.strategy_choices.count(0), observer.strategy_choices.count(1)]
    print(f"策略选择分布: 生成新图={strategy_counts[0]}, 变异现有图={strategy_counts[1]}")
    
    # 分析动作选择
    action_counts = [observer.action_choices.count(i) for i in range(4)]
    action_names = ['Add Input', 'Add Op', 'Add Loop', 'Add Branch']
    print("动作选择分布:")
    for i, (name, count) in enumerate(zip(action_names, action_counts)):
        print(f"  {name}: {count}")
    
    # 分析性能改进
    improvements = sum(1 for data in observer.iteration_data if data['improved'])
    print(f"总改进次数: {improvements}/{max_iterations} ({improvements/max_iterations*100:.1f}%)")
    
    # 最终智能体状态
    print("\n最终智能体状态:")
    print(f"  策略智能体: {bandit_fuzzer.strategy_agent.alpha_beta}")
    print(f"  动作智能体: {bandit_fuzzer.action_agent.alpha_beta}")
    
    # 保存和绘制结果
    observer.save_data(os.path.join(output_dir, "learning_data.json"))
    observer.plot_learning_progress(os.path.join(output_dir, "learning_analysis.png"))
    
    return observer

if __name__ == "__main__":
    # 运行学习测试
    observer = test_bandit_learning(max_iterations=30, verbose=True)
    print("\n测试完成！查看 ./test_learning_output/ 目录中的结果文件。")
