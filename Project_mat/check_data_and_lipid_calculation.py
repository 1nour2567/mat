#!/usr/bin/env python3
# 检查数据基本信息和血脂异常计算

import pandas as pd
import numpy as np

# 加载数据
data_path = '/workspace/Project_mat/data/processed/preprocessed_data.pkl'
print(f"加载数据: {data_path}")
df = pd.read_pickle(data_path)

print(f"数据形状: {df.shape}")
print(f"总样本数: {len(df)}")

# 检查血脂相关列
lipid_columns = ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）']
print("\n=== 血脂列检查 ===")
for col in lipid_columns:
    if col in df.columns:
        print(f"{col}: 存在")
        print(f"  最小值: {df[col].min():.4f}")
        print(f"  最大值: {df[col].max():.4f}")
        print(f"  均值: {df[col].mean():.4f}")
    else:
        print(f"{col}: 不存在")

# 计算血脂异常项数
def calc_lipid_abnormal_count(row):
    """
    计算血脂异常项数
    """
    # 严格执行临床正常范围
    checks = [
        row['TC（总胆固醇）'] > 6.2 or row['TC（总胆固醇）'] < 3.1,
        row['TG（甘油三酯）'] > 1.7 or row['TG（甘油三酯）'] < 0.56,
        row['LDL-C（低密度脂蛋白）'] > 3.1 or row['LDL-C（低密度脂蛋白）'] < 2.07,
        row['HDL-C（高密度脂蛋白）'] < 1.04 or row['HDL-C（高密度脂蛋白）'] > 1.55
    ]
    return sum(checks)

# 应用计算
df['血脂异常项数'] = df.apply(calc_lipid_abnormal_count, axis=1)
df['临床确诊高风险'] = (df['血脂异常项数'] >= 1).astype(int)

# 统计结果
print("\n=== 血脂异常统计 ===")
print(f"血脂异常项数分布:")
print(df['血脂异常项数'].value_counts().sort_index())

print("\n=== 临床确诊高风险统计 ===")
print(f"临床确诊高风险样本数: {df['临床确诊高风险'].sum()}")
print(f"临床确诊高风险占比: {(df['临床确诊高风险'].sum() / len(df)) * 100:.1f}%")

# 检查具体的异常情况
print("\n=== 具体异常情况 ===")
for col in lipid_columns:
    if col in df.columns:
        if col == 'TC（总胆固醇）':
            abnormal = ((df[col] > 6.2) | (df[col] < 3.1)).sum()
        elif col == 'TG（甘油三酯）':
            abnormal = ((df[col] > 1.7) | (df[col] < 0.56)).sum()
        elif col == 'LDL-C（低密度脂蛋白）':
            abnormal = ((df[col] > 3.1) | (df[col] < 2.07)).sum()
        elif col == 'HDL-C（高密度脂蛋白）':
            abnormal = ((df[col] < 1.04) | (df[col] > 1.55)).sum()
        print(f"{col} 异常样本数: {abnormal}")

# 显示前10行数据的血脂情况
print("\n=== 前10行数据血脂情况 ===")
print(df[['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）', '血脂异常项数', '临床确诊高风险']].head(10))
