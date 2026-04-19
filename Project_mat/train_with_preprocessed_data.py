#!/usr/bin/env python3
# 使用 preprocessed_data.pkl 训练三层架构模型

import pandas as pd
import numpy as np
from src.three_layer_architecture import TripleLayerPredictor

# 加载数据
data_path = '/workspace/Project_mat/data/processed/preprocessed_data.pkl'
print(f"加载数据: {data_path}")
df = pd.read_pickle(data_path)

print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")

# 检查必要的列
required_columns = [
    '平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质',
    'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）', '年龄组', '性别', '吸烟史', '饮酒史',
    '空腹血糖', '血尿酸', 'BMI',
    'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 'TC（总胆固醇）',
    '高血脂症二分类标签'
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"警告: 缺少必要列: {missing_columns}")
else:
    print("所有必要列都存在")

# 初始化模型
print("\n初始化三层架构模型...")
predictor = TripleLayerPredictor()

# 训练模型
print("\n训练模型...")
predictor.fit(df)

# 预测
print("\n进行预测...")
df_result = predictor.predict(df)

# 分析结果
print("\n=== 结果分析 ===")
risk_distribution = df_result['最终风险等级'].value_counts()
print("风险等级分布:")
for risk_level, count in risk_distribution.items():
    percentage = (count / len(df_result)) * 100
    print(f"  {risk_level}: {count} ({percentage:.1f}%)")

# 保存结果
output_path = '/workspace/Project_mat/data/processed/three_layer_result_from_preprocessed.pkl'
print(f"\n保存结果到: {output_path}")
df_result.to_pickle(output_path)

# 显示前几行结果
print("\n=== 预测结果示例 ===")
print(df_result[['痰湿质', '活动量表总分（ADL总分+IADL总分）', '血脂异常项数', '模型预测概率', '最终风险等级', '高血脂症二分类标签']].head())

print("\n训练完成！")
