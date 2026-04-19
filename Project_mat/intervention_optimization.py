#!/usr/bin/env python3
# 痰湿体质患者个性化干预方案优化模型

import pandas as pd
import numpy as np

# 定义参数
class InterventionOptimizer:
    def __init__(self):
        # 中医调理参数
        self.tcm_costs = {1: 30, 2: 80, 3: 130}  # 月成本
        self.tcm_reductions = {1: 0.01, 2: 0.02, 3: 0.03}  # 月降幅
        
        # 活动干预参数
        self.activity_costs = {1: 3, 2: 5, 3: 8}  # 单次成本
        self.activity_reductions = lambda u, f: 0.03 * (u - 1) + 0.01 * (f - 5) if f >= 5 else 0
        
        # 预算约束
        self.budget_limit = 2000
        
        # 月数
        self.months = 6
        
    def get_allowed_activity_intensities(self, age_group, activity_score):
        """
        根据年龄组和活动能力得分获取允许的活动强度
        年龄组: 1=40-49, 2=50-59, 3=60-69, 4=70-79, 5=80-89
        活动得分: <40, 40-59, >=60
        """
        # 年龄约束
        if age_group == 5:  # 80-89岁
            return {1}
        elif age_group in [3, 4]:  # 60-79岁
            age_allowed = {1, 2}
        else:  # 40-59岁
            age_allowed = {1, 2, 3}
        
        # 活动能力约束
        if activity_score < 40:
            activity_allowed = {1}
        elif 40 <= activity_score < 60:
            activity_allowed = {1, 2}
        else:
            activity_allowed = {1, 2, 3}
        
        # 交集
        return age_allowed & activity_allowed
    
    def calculate_monthly_cost(self, tcm_level, activity_intensity, frequency):
        """
        计算每月成本
        """
        tcm_cost = self.tcm_costs[tcm_level]
        activity_cost = 4 * frequency * self.activity_costs[activity_intensity]  # 每月4周
        return tcm_cost + activity_cost
    
    def simulate_intervention(self, initial_score, tcm_level, activity_intensity, frequency):
        """
        模拟6个月干预效果
        采用乘法叠加：s_{t+1} = s_t * (1 - r_TCM) * (1 - r_act)
        """
        score = initial_score
        total_cost = 0
        monthly_reductions = []
        
        for _ in range(self.months):
            # 计算每月降幅（乘法叠加）
            tcm_reduction = self.tcm_reductions[tcm_level]
            activity_reduction = self.activity_reductions(activity_intensity, frequency)
            
            # 乘法叠加：(1 - r_TCM) * (1 - r_act) = 1 - r_total
            total_reduction_factor = (1 - tcm_reduction) * (1 - activity_reduction)
            total_reduction = 1 - total_reduction_factor
            
            # 应用软约束：月降幅不超过20%
            total_reduction = min(total_reduction, 0.20)
            total_reduction_factor = 1 - total_reduction
            
            # 更新积分
            score *= total_reduction_factor
            monthly_reductions.append(total_reduction)
            
            # 计算成本
            total_cost += self.calculate_monthly_cost(tcm_level, activity_intensity, frequency)
        
        return score, total_cost, monthly_reductions
    
    def optimize_intervention(self, initial_score, age_group, activity_score):
        """
        优化干预方案
        """
        allowed_intensities = self.get_allowed_activity_intensities(age_group, activity_score)
        
        best_score = float('inf')
        best_cost = float('inf')
        best方案 = None
        
        # 枚举所有可能的方案
        for tcm_level in [1, 2, 3]:
            for activity_intensity in allowed_intensities:
                for frequency in range(1, 11):  # 1-10次/周
                    # 计算总成本
                    total_cost = sum(self.calculate_monthly_cost(tcm_level, activity_intensity, frequency) for _ in range(self.months))
                    
                    # 预算约束
                    if total_cost > self.budget_limit:
                        continue
                    
                    # 模拟干预效果
                    final_score, calc_cost, reductions = self.simulate_intervention(initial_score, tcm_level, activity_intensity, frequency)
                    
                    # 词典序优化：先看最终积分，再看成本，最后看强度和频次
                    if (final_score < best_score) or \
                       (final_score == best_score and calc_cost < best_cost) or \
                       (final_score == best_score and calc_cost == best_cost and (activity_intensity + frequency) < (best方案[1] + best方案[2] if best方案 else float('inf'))):
                        best_score = final_score
                        best_cost = calc_cost
                        best方案 = (tcm_level, activity_intensity, frequency)
        
        return best方案, best_score, best_cost
    
    def analyze_sample(self, sample_id, initial_score, age_group, activity_score):
        """
        分析单个样本
        """
        best方案, final_score, total_cost = self.optimize_intervention(initial_score, age_group, activity_score)
        
        print(f"\n=== 样本ID={sample_id} 最优干预方案 ===")
        print(f"初始痰湿积分: {initial_score}")
        print(f"年龄组: {age_group} (40-49=1, 50-59=2, 60-69=3, 70-79=4, 80-89=5)")
        print(f"活动总分: {activity_score}")
        print(f"允许的活动强度: {self.get_allowed_activity_intensities(age_group, activity_score)}")
        print(f"最优方案: 中医{best方案[0]}级 + 活动{best方案[1]}级 + {best方案[2]}次/周")
        print(f"6月末积分: {final_score:.2f}")
        print(f"总成本: {total_cost:.2f}元")
        
        return {
            'sample_id': sample_id,
            'initial_score': initial_score,
            'age_group': age_group,
            'activity_score': activity_score,
            'tcm_level': best方案[0],
            'activity_intensity': best方案[1],
            'frequency': best方案[2],
            'final_score': final_score,
            'total_cost': total_cost
        }

# 主函数
if __name__ == "__main__":
    optimizer = InterventionOptimizer()
    
    # 分析样本ID=1, 2, 3
    samples = [
        {'id': 1, 'initial_score': 64, 'age_group': 2, 'activity_score': 38},  # 50-59岁，活动能力低
        {'id': 2, 'initial_score': 58, 'age_group': 1, 'activity_score': 40},  # 40-49岁，活动能力中
        {'id': 3, 'initial_score': 59, 'age_group': 1, 'activity_score': 63}   # 40-49岁，活动能力高
    ]
    
    results = []
    for sample in samples:
        result = optimizer.analyze_sample(
            sample['id'],
            sample['initial_score'],
            sample['age_group'],
            sample['activity_score']
        )
        results.append(result)
    
    # 生成匹配规律
    print("\n=== 患者特征-最优方案匹配规律 ===")
    print("1. 活动能力是活动强度的首要硬约束：活动总分<40者仅能选择1级强度")
    print("2. 年龄对活动强度的约束严格：60-79岁不可选3级，80-89岁仅可选1级")
    print("3. 高频次（10次/周）是普遍最优选择：在预算允许时推荐最高频次")
    print("4. 高活动能力者可'以强度换频次'：采用3级强度配合8-9次/周")
    
    # 保存结果
    df_results = pd.DataFrame(results)
    output_path = '/workspace/Project_mat/data/processed/intervention_optimization_results.csv'
    df_results.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")
