#!/usr/bin/env python3
# 训练干预优化模型，完成问题三第一步

import pandas as pd
import numpy as np
from intervention_optimization import InterventionOptimizer

# 加载预处理数据
data_path = '/workspace/Project_mat/data/processed/preprocessed_data.pkl'
print(f"加载数据: {data_path}")
df = pd.read_pickle(data_path)

print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns[:20])}...")

# 筛选痰湿体质患者（假设体质标签为5表示痰湿质）
# 检查是否有体质标签列
if '体质标签' in df.columns:
    print("\n体质标签分布:")
    print(df['体质标签'].value_counts())
    
    # 筛选痰湿体质患者
    tan_shi_patients = df[df['体质标签'] == 5].copy()
    print(f"\n痰湿体质患者数: {len(tan_shi_patients)}")
else:
    # 如果没有体质标签，使用痰湿质得分
    print("\n使用痰湿质得分筛选患者")
    print(f"痰湿质得分范围: {df['痰湿质'].min()} - {df['痰湿质'].max()}")
    
    # 筛选痰湿质得分≥60的患者
    tan_shi_patients = df[df['痰湿质'] >= 60].copy()
    print(f"痰湿质≥60的患者数: {len(tan_shi_patients)}")

# 检查必要列
required_columns = ['年龄组', '活动量表总分（ADL总分+IADL总分）', '痰湿质']
for col in required_columns:
    if col not in tan_shi_patients.columns:
        print(f"警告: 缺少列 {col}")

# 年龄组映射（假设年龄组编码：1=40-49, 2=50-59, 3=60-69, 4=70-79, 5=80-89）
def map_age_group(age):
    if 40 <= age < 50:
        return 1
    elif 50 <= age < 60:
        return 2
    elif 60 <= age < 70:
        return 3
    elif 70 <= age < 80:
        return 4
    elif 80 <= age < 90:
        return 5
    else:
        return 3  # 默认值

# 如果没有年龄组列，从年龄计算
if '年龄组' not in tan_shi_patients.columns and '年龄' in tan_shi_patients.columns:
    tan_shi_patients['年龄组'] = tan_shi_patients['年龄'].apply(map_age_group)
    print("\n从年龄计算年龄组")
    print(tan_shi_patients['年龄组'].value_counts())

# 初始化优化器
optimizer = InterventionOptimizer()

# 分析所有痰湿体质患者
results = []
print("\n分析痰湿体质患者的最优干预方案...")

for idx, row in tan_shi_patients.iterrows():
    sample_id = row.get('样本ID', idx)
    initial_score = row['痰湿质']
    age_group = row.get('年龄组', 3)  # 默认3
    activity_score = row.get('活动量表总分（ADL总分+IADL总分）', 50)  # 默认50
    
    # 计算最优方案
    best方案, final_score, total_cost = optimizer.optimize_intervention(
        initial_score, age_group, activity_score
    )
    
    results.append({
        '样本ID': sample_id,
        '初始痰湿积分': initial_score,
        '年龄组': age_group,
        '活动总分': activity_score,
        '中医调理等级': best方案[0],
        '活动强度': best方案[1],
        '每周频次': best方案[2],
        '6月末积分': final_score,
        '总成本': total_cost
    })

# 保存结果
df_results = pd.DataFrame(results)
output_path = '/workspace/Project_mat/data/processed/tan_shi_intervention_results.csv'
df_results.to_csv(output_path, index=False)
print(f"\n结果已保存到: {output_path}")

# 分析结果
print("\n=== 痰湿体质患者干预方案分析 ===")
print(f"总分析患者数: {len(df_results)}")
print("\n干预方案分布:")
print(df_results.groupby(['中医调理等级', '活动强度', '每周频次']).size().sort_values(ascending=False).head(10))

print("\n年龄组分布:")
print(df_results['年龄组'].value_counts())

print("\n活动能力分布:")
df_results['活动能力等级'] = pd.cut(
    df_results['活动总分'],
    bins=[0, 40, 60, 100],
    labels=['低', '中', '高']
)
print(df_results['活动能力等级'].value_counts())

print("\n平均结果:")
print(f"平均初始积分: {df_results['初始痰湿积分'].mean():.2f}")
print(f"平均6月末积分: {df_results['6月末积分'].mean():.2f}")
print(f"平均积分下降: {(df_results['初始痰湿积分'].mean() - df_results['6月末积分'].mean()):.2f}")
print(f"平均总成本: {df_results['总成本'].mean():.2f}元")
