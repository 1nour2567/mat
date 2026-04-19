#!/usr/bin/env python3
# 分析痰湿体质高风险人群的核心特征组合

import pandas as pd
import numpy as np

# 加载训练结果数据
data_path = '/workspace/Project_mat/data/processed/three_layer_result_from_preprocessed.pkl'
print(f"加载数据: {data_path}")
df = pd.read_pickle(data_path)

# 筛选痰湿体质较高的样本（痰湿质≥60）
tan_shi_high = df[df['痰湿质'] >= 60]
print(f"\n=== 痰湿质≥60的样本分析 ===")
print(f"样本数: {len(tan_shi_high)}")
print(f"占总样本比例: {(len(tan_shi_high) / len(df)) * 100:.1f}%")

# 分析痰湿质高样本的风险等级分布
print("\n痰湿质高样本的风险等级分布:")
risk_dist = tan_shi_high['最终风险等级'].value_counts()
for risk, count in risk_dist.items():
    percentage = (count / len(tan_shi_high)) * 100
    print(f"  {risk}: {count} ({percentage:.1f}%)")

# 分析痰湿质高样本的活动能力
print("\n痰湿质高样本的活动能力分布:")
tan_shi_high['活动能力等级'] = pd.cut(
    tan_shi_high['活动量表总分（ADL总分+IADL总分）'],
    bins=[0, 40, 60, 100],
    labels=['低活动能力', '中等活动能力', '高活动能力']
)
activity_dist = tan_shi_high['活动能力等级'].value_counts()
for activity, count in activity_dist.items():
    percentage = (count / len(tan_shi_high)) * 100
    print(f"  {activity}: {count} ({percentage:.1f}%)")

# 分析痰湿质高样本的血脂异常情况
print("\n痰湿质高样本的血脂异常分布:")
tan_shi_high['血脂异常'] = tan_shi_high['血脂异常项数'] >= 1
lipid_abnormal_dist = tan_shi_high['血脂异常'].value_counts()
for status, count in lipid_abnormal_dist.items():
    status_str = "异常" if status else "正常"
    percentage = (count / len(tan_shi_high)) * 100
    print(f"  血脂{status_str}: {count} ({percentage:.1f}%)")

# 分析核心特征组合
print("\n=== 核心特征组合分析 ===")

# 1. 痰湿质高 + 低活动能力 + 血脂异常
combination1 = tan_shi_high[(tan_shi_high['活动量表总分（ADL总分+IADL总分）'] < 40) & (tan_shi_high['血脂异常项数'] >= 1)]
print(f"1. 痰湿质高 + 低活动能力 + 血脂异常: {len(combination1)}人 ({(len(combination1)/len(tan_shi_high))*100:.1f}%)")

# 2. 痰湿质高 + 低活动能力 + 血脂正常
combination2 = tan_shi_high[(tan_shi_high['活动量表总分（ADL总分+IADL总分）'] < 40) & (tan_shi_high['血脂异常项数'] == 0)]
print(f"2. 痰湿质高 + 低活动能力 + 血脂正常: {len(combination2)}人 ({(len(combination2)/len(tan_shi_high))*100:.1f}%)")

# 3. 痰湿质高 + 中等活动能力 + 血脂异常
combination3 = tan_shi_high[(tan_shi_high['活动量表总分（ADL总分+IADL总分）'] >= 40) & (tan_shi_high['活动量表总分（ADL总分+IADL总分）'] < 60) & (tan_shi_high['血脂异常项数'] >= 1)]
print(f"3. 痰湿质高 + 中等活动能力 + 血脂异常: {len(combination3)}人 ({(len(combination3)/len(tan_shi_high))*100:.1f}%)")

# 4. 痰湿质高 + 高活动能力 + 血脂异常
combination4 = tan_shi_high[(tan_shi_high['活动量表总分（ADL总分+IADL总分）'] >= 60) & (tan_shi_high['血脂异常项数'] >= 1)]
print(f"4. 痰湿质高 + 高活动能力 + 血脂异常: {len(combination4)}人 ({(len(combination4)/len(tan_shi_high))*100:.1f}%)")

# 分析其他相关特征
print("\n=== 其他相关特征分析 ===")

# BMI分布
print("BMI分布:")
tan_shi_high['BMI等级'] = pd.cut(
    tan_shi_high['BMI'],
    bins=[0, 18.5, 24, 28, 100],
    labels=['体重过轻', '正常体重', '超重', '肥胖']
)
bmi_dist = tan_shi_high['BMI等级'].value_counts()
for bmi, count in bmi_dist.items():
    percentage = (count / len(tan_shi_high)) * 100
    print(f"  {bmi}: {count} ({percentage:.1f}%)")

# 空腹血糖分布
print("\n空腹血糖分布:")
tan_shi_high['血糖等级'] = pd.cut(
    tan_shi_high['空腹血糖'],
    bins=[0, 3.9, 6.1, 100],
    labels=['低血糖', '正常血糖', '高血糖']
)
glucose_dist = tan_shi_high['血糖等级'].value_counts()
for glucose, count in glucose_dist.items():
    percentage = (count / len(tan_shi_high)) * 100
    print(f"  {glucose}: {count} ({percentage:.1f}%)")

# 血尿酸分布
print("\n血尿酸分布:")
tan_shi_high['尿酸等级'] = pd.cut(
    tan_shi_high['血尿酸'],
    bins=[0, 208, 357, 428, 1000],
    labels=['低尿酸', '正常尿酸', '尿酸偏高', '高尿酸']
)
uric_acid_dist = tan_shi_high['尿酸等级'].value_counts()
for uric, count in uric_acid_dist.items():
    percentage = (count / len(tan_shi_high)) * 100
    print(f"  {uric}: {count} ({percentage:.1f}%)")

# 分析中医体质组合
print("\n=== 中医体质组合分析 ===")
# 计算其他体质得分的平均值
other_constitutions = ['平和质', '气虚质', '阳虚质', '阴虚质', '湿热质', '血瘀质', '气郁质', '特禀质']
print("其他中医体质平均得分:")
for constitution in other_constitutions:
    avg_score = tan_shi_high[constitution].mean()
    print(f"  {constitution}: {avg_score:.2f}")

# 识别高风险的核心特征组合
print("\n=== 核心特征组合识别 ===")
print("基于以上分析，痰湿体质高风险人群的核心特征组合包括：")
print("1. 痰湿质≥60 + 活动能力<40 + 血脂异常")
print("2. 痰湿质≥60 + 活动能力<40 + 血脂正常（中医预警高风险）")
print("3. 痰湿质≥60 + 活动能力≥40 + 血脂异常")
