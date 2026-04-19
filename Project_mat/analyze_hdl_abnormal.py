import pandas as pd
import numpy as np

# 读取预处理数据
df = pd.read_pickle('data/processed/preprocessed_data.pkl')

# 计算HDL异常情况
df['HDL低异常'] = (df['HDL-C（高密度脂蛋白）'] < 1.04).astype(int)
df['HDL高异常'] = (df['HDL-C（高密度脂蛋白）'] > 1.55).astype(int)
df['HDL异常'] = (df['HDL低异常'] | df['HDL高异常']).astype(int)

# 计算血脂异常项数（修改前的逻辑）
def calc_lipid_abnormal_count_old(row):
    checks = [
        row['TC（总胆固醇）'] > 6.2 or row['TC（总胆固醇）'] < 3.1,
        row['TG（甘油三酯）'] > 1.7 or row['TG（甘油三酯）'] < 0.56,
        row['LDL-C（低密度脂蛋白）'] > 3.1 or row['LDL-C（低密度脂蛋白）'] < 2.07,
        row['HDL-C（高密度脂蛋白）'] < 1.04  # 仅检查低异常
    ]
    return sum(checks)

# 计算血脂异常项数（修改后的逻辑）
def calc_lipid_abnormal_count_new(row):
    checks = [
        row['TC（总胆固醇）'] > 6.2 or row['TC（总胆固醇）'] < 3.1,
        row['TG（甘油三酯）'] > 1.7 or row['TG（甘油三酯）'] < 0.56,
        row['LDL-C（低密度脂蛋白）'] > 3.1 or row['LDL-C（低密度脂蛋白）'] < 2.07,
        row['HDL-C（高密度脂蛋白）'] < 1.04 or row['HDL-C（高密度脂蛋白）'] > 1.55  # 检查低异常和高异常
    ]
    return sum(checks)

# 应用两种计算方法
df['血脂异常项数_old'] = df.apply(calc_lipid_abnormal_count_old, axis=1)
df['血脂异常项数_new'] = df.apply(calc_lipid_abnormal_count_new, axis=1)

# 计算高风险人数
high_risk_old = (df['血脂异常项数_old'] >= 1).sum()
high_risk_new = (df['血脂异常项数_new'] >= 1).sum()

# 统计HDL异常情况
hdl_abnormal_total = df['HDL异常'].sum()
hdl_low_abnormal = df['HDL低异常'].sum()
hdl_high_abnormal = df['HDL高异常'].sum()

# 找出因HDL高异常而新增的高风险样本
new_high_risk_samples = df[(df['血脂异常项数_old'] == 0) & (df['血脂异常项数_new'] >= 1)]
new_high_risk_count = len(new_high_risk_samples)

# 输出结果
print("=== HDL异常分析 ===")
print(f"HDL异常总人数: {hdl_abnormal_total}")
print(f"HDL低异常人数 (<1.04): {hdl_low_abnormal}")
print(f"HDL高异常人数 (>1.55): {hdl_high_abnormal}")
print()
print("=== 血脂异常人数分析 ===")
print(f"修改前血脂异常人数: {high_risk_old}")
print(f"修改后血脂异常人数: {high_risk_new}")
print(f"新增血脂异常人数: {high_risk_new - high_risk_old}")
print()
print("=== 新增高风险样本分析 ===")
print(f"因HDL高异常新增的高风险样本数: {new_high_risk_count}")
print()
print("新增高风险样本详情:")
if new_high_risk_count > 0:
    print(new_high_risk_samples[[
        'HDL-C（高密度脂蛋白）', 'HDL低异常', 'HDL高异常', 
        '血脂异常项数_old', '血脂异常项数_new'
    ]].head())
else:
    print("无新增高风险样本")
