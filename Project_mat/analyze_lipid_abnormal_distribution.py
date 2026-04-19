#!/usr/bin/env python3
# 分析确诊人数中血脂异常分布

import pandas as pd
import numpy as np

# 加载数据
data_path = '/workspace/Project_mat/data/processed/preprocessed_data.pkl'
print(f"加载数据: {data_path}")
df = pd.read_pickle(data_path)

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
        row['HDL-C（高密度脂蛋白）'] < 0.9 or row['HDL-C（高密度脂蛋白）'] > 1.55
    ]
    return sum(checks)

# 应用计算
df['血脂异常项数'] = df.apply(calc_lipid_abnormal_count, axis=1)
df['临床确诊高风险'] = (df['血脂异常项数'] >= 1).astype(int)

# 筛选确诊样本
confirmed_df = df[df['临床确诊高风险'] == 1]
print(f"\n=== 确诊样本基本信息 ===")
print(f"确诊样本数: {len(confirmed_df)}")
print(f"占总样本比例: {(len(confirmed_df) / len(df)) * 100:.1f}%")

# 分析各项血脂异常分布
print("\n=== 确诊样本中各项血脂异常分布 ===")

# TC异常
tc_abnormal = ((confirmed_df['TC（总胆固醇）'] > 6.2) | (confirmed_df['TC（总胆固醇）'] < 3.1)).sum()
print(f"TC（总胆固醇）异常: {tc_abnormal} ({(tc_abnormal / len(confirmed_df)) * 100:.1f}%)")

# TG异常
tg_abnormal = ((confirmed_df['TG（甘油三酯）'] > 1.7) | (confirmed_df['TG（甘油三酯）'] < 0.56)).sum()
print(f"TG（甘油三酯）异常: {tg_abnormal} ({(tg_abnormal / len(confirmed_df)) * 100:.1f}%)")

# LDL-C异常
ldl_abnormal = ((confirmed_df['LDL-C（低密度脂蛋白）'] > 3.1) | (confirmed_df['LDL-C（低密度脂蛋白）'] < 2.07)).sum()
print(f"LDL-C（低密度脂蛋白）异常: {ldl_abnormal} ({(ldl_abnormal / len(confirmed_df)) * 100:.1f}%)")

# HDL-C异常
hdl_abnormal = ((confirmed_df['HDL-C（高密度脂蛋白）'] < 0.9) | (confirmed_df['HDL-C（高密度脂蛋白）'] > 1.55)).sum()
print(f"HDL-C（高密度脂蛋白）异常: {hdl_abnormal} ({(hdl_abnormal / len(confirmed_df)) * 100:.1f}%)")

# 分析血脂异常项数分布
print("\n=== 确诊样本中血脂异常项数分布 ===")
abnormal_count_dist = confirmed_df['血脂异常项数'].value_counts().sort_index()
for count, num_samples in abnormal_count_dist.items():
    percentage = (num_samples / len(confirmed_df)) * 100
    print(f"{count}项异常: {num_samples} ({percentage:.1f}%)")

# 分析同时异常的组合
print("\n=== 常见异常组合分析 ===")

# 定义异常标志
confirmed_df['TC异常'] = ((confirmed_df['TC（总胆固醇）'] > 6.2) | (confirmed_df['TC（总胆固醇）'] < 3.1)).astype(int)
confirmed_df['TG异常'] = ((confirmed_df['TG（甘油三酯）'] > 1.7) | (confirmed_df['TG（甘油三酯）'] < 0.56)).astype(int)
confirmed_df['LDL异常'] = ((confirmed_df['LDL-C（低密度脂蛋白）'] > 3.1) | (confirmed_df['LDL-C（低密度脂蛋白）'] < 2.07)).astype(int)
confirmed_df['HDL异常'] = ((confirmed_df['HDL-C（高密度脂蛋白）'] < 0.9) | (confirmed_df['HDL-C（高密度脂蛋白）'] > 1.55)).astype(int)

# 计算常见组合
print("前10种最常见的异常组合:")
combination_counts = confirmed_df.groupby(['TC异常', 'TG异常', 'LDL异常', 'HDL异常']).size().sort_values(ascending=False).head(10)

for i, (combination, count) in enumerate(combination_counts.items()):
    tc, tg, ldl, hdl = combination
    abnormal_items = []
    if tc:
        abnormal_items.append('TC')
    if tg:
        abnormal_items.append('TG')
    if ldl:
        abnormal_items.append('LDL')
    if hdl:
        abnormal_items.append('HDL')
    combination_str = '+'.join(abnormal_items) if abnormal_items else '无异常'
    percentage = (count / len(confirmed_df)) * 100
    print(f"{i+1}. {combination_str}: {count} ({percentage:.1f}%)")
