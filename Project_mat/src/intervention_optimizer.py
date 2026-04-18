# 动态规划求解器模块
import pandas as pd
import numpy as np
from config.constants import COST_MAPPING

def calculate_intervention_cost(intervention_type):
    """计算干预成本"""
    return COST_MAPPING.get(intervention_type, 0)

def calculate_risk_reduction(risk_level, intervention_type):
    """计算风险降低程度"""
    # 假设不同干预类型对不同风险等级的降低效果
    reduction_mapping = {
        1: {'intervention_type_1': 0.1, 'intervention_type_2': 0.2, 'intervention_type_3': 0.3},
        2: {'intervention_type_1': 0.15, 'intervention_type_2': 0.25, 'intervention_type_3': 0.35},
        3: {'intervention_type_1': 0.2, 'intervention_type_2': 0.3, 'intervention_type_3': 0.4}
    }
    return reduction_mapping.get(risk_level, {}).get(intervention_type, 0)

def dynamic_programming_optimizer(risk_levels, budget):
    """动态规划求解最优干预策略"""
    n = len(risk_levels)
    max_budget = budget
    
    # 初始化DP表
    dp = [[0] * (max_budget + 1) for _ in range(n + 1)]
    
    # 遍历每个个体
    for i in range(1, n + 1):
        risk_level = risk_levels[i - 1]
        
        # 遍历每个预算
        for b in range(max_budget + 1):
            # 不干预的情况
            dp[i][b] = dp[i - 1][b]
            
            # 尝试不同干预类型
            for intervention in COST_MAPPING.keys():
                cost = calculate_intervention_cost(intervention)
                if cost <= b:
                    # 计算风险降低收益
                    reduction = calculate_risk_reduction(risk_level, intervention)
                    # 更新DP表
                    if dp[i - 1][b - cost] + reduction > dp[i][b]:
                        dp[i][b] = dp[i - 1][b - cost] + reduction
    
    # 回溯找到最优策略
    optimal_strategy = []
    current_budget = max_budget
    
    for i in range(n, 0, -1):
        risk_level = risk_levels[i - 1]
        
        # 不干预的情况
        if dp[i][current_budget] == dp[i - 1][current_budget]:
            optimal_strategy.append('no_intervention')
        else:
            # 找到对应的干预类型
            for intervention in COST_MAPPING.keys():
                cost = calculate_intervention_cost(intervention)
                if current_budget >= cost and dp[i][current_budget] == dp[i - 1][current_budget - cost] + calculate_risk_reduction(risk_level, intervention):
                    optimal_strategy.append(intervention)
                    current_budget -= cost
                    break
    
    # 反转策略列表，使其与个体顺序一致
    optimal_strategy = optimal_strategy[::-1]
    
    return optimal_strategy, dp[n][max_budget]

def optimize_interventions(input_path, budget):
    """完整干预优化流程"""
    # 加载数据（包含风险等级）
    df = pd.read_pickle(input_path)
    
    # 提取风险等级
    risk_levels = df['risk_level'].tolist()
    
    # 使用动态规划求解最优策略
    optimal_strategy, total_reduction = dynamic_programming_optimizer(risk_levels, budget)
    
    # 将策略添加到数据框
    df['intervention_strategy'] = optimal_strategy
    
    # 计算总成本
    total_cost = sum(calculate_intervention_cost(strategy) for strategy in optimal_strategy if strategy != 'no_intervention')
    
    print(f"优化结果：")
    print(f"总风险降低: {total_reduction:.4f}")
    print(f"总成本: {total_cost}")
    print(f"剩余预算: {budget - total_cost}")
    
    return df, optimal_strategy, total_reduction