
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.three_layer_architecture import TripleLayerPredictor

# 加载数据
df = pd.read_pickle('data/processed/preprocessed_data.pkl')

# 首先查看血脂正常的样本（临床确诊高风险是血脂异常项数≥1）
from src.three_layer_architecture import ClinicalRuleLayer
df['血脂异常项数'] = df.apply(ClinicalRuleLayer.calc_lipid_abnormal_count, axis=1)
normal_lipid_df = df[df['血脂异常项数'] == 0].copy()

print(f"血脂正常的样本数: {len(normal_lipid_df)}")

# 查看痰湿质分布
print("\n血脂正常样本的痰湿质分布:")
print(normal_lipid_df['痰湿质'].describe())

# 查看痰湿质≥49的样本数
tcm_high_risk = normal_lipid_df[normal_lipid_df['痰湿质'] >= 49]
print(f"\n痰湿质≥49的血脂正常样本数: {len(tcm_high_risk)}")
if len(tcm_high_risk) > 0:
    print(tcm_high_risk[['痰湿质', '活动量表总分（ADL总分+IADL总分）']])

# 现在我们训练模型并查看这些样本的预测概率
print("\n=== 训练模型 ===")
predictor = TripleLayerPredictor()
predictor.fit(df)

# 预测
df_result = predictor.predict(df)

# 查看血脂正常样本的预测结果
normal_lipid_result = df_result[df_result['血脂异常项数'] == 0]
print("\n血脂正常样本的风险等级分布:")
print(normal_lipid_result['最终风险等级'].value_counts())

# 查看痰湿质≥49的血脂正常样本的预测结果
if len(tcm_high_risk) > 0:
    tcm_high_risk_result = df_result[df_result.index.isin(tcm_high_risk.index)]
    print("\n痰湿质≥49的血脂正常样本的预测结果:")
    print(tcm_high_risk_result[['痰湿质', '最终风险等级', '模型预测概率']])

# 查看所有样本的预测概率分布
print("\n所有血脂正常样本的模型预测概率分布:")
print(normal_lipid_result['模型预测概率'].describe())
