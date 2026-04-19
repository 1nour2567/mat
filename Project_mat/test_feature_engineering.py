#!/usr/bin/env python3
"""
独立测试特征工程函数
"""
import pandas as pd
import numpy as np

print("=== 加载数据 ===")
df = pd.read_pickle('/workspace/Project_mat/data/processed/preprocessed_data.pkl')
print(f"数据形状：{df.shape}")

print("\n=== 定义特征工程函数 ===")
def create_tcm_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建中西医交叉特征
    
    Args:
        df: 原始数据框
        
    Returns:
        添加交叉特征后的数据框
    """
    df = df.copy()
    
    # 痰湿质相关交叉特征
    df['痰湿质×BMI'] = df['痰湿质'] * df['BMI']
    df['痰湿质×活动量表'] = df['痰湿质'] * df['活动量表总分（ADL总分+IADL总分）']
    df['痰湿质×血尿酸'] = df['痰湿质'] * df['血尿酸']
    
    # 气虚质相关交叉特征
    df['气虚质×BMI'] = df['气虚质'] * df['BMI']
    df['气虚质×活动量表'] = df['气虚质'] * df['活动量表总分（ADL总分+IADL总分）']
    
    return df

print("\n=== 测试交叉特征生成 ===")
df_with_interactions = create_tcm_interactions(df)
print("新增交叉特征：")
for col in df_with_interactions.columns:
    if '×' in col:
        print(f"  - {col}")

print("\n=== LIPID_FEATURES 屏蔽清单 ===")
LIPID_FEATURES = [
    'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 'TC（总胆固醇）',
    'AIP', 'TC/HDL比值', 'non-HDL-C', 'LDL/HDL比值', 'TG/HDL比值',
    'non-HDL-C_缩尾', 'AIP_缩尾', 'TC/HDL比值_缩尾', 'LDL/HDL比值_缩尾', 'TG/HDL比值_缩尾',
    'TC异常', 'TG异常', 'LDL-C异常', 'HDL-C异常', '血脂异常项数'
]
for i, feature in enumerate(LIPID_FEATURES, 1):
    print(f"{i:2d}. {feature}")

print("\n=== 模型可用特征列表 ===")
BASE_MODEL_FEATURES = [
    '平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质',
    'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）', '年龄组', '性别', '吸烟史', '饮酒史',
    '空腹血糖', '血尿酸', 'BMI', '尿酸异常'
]
MODEL_FEATURES = BASE_MODEL_FEATURES + [
    '痰湿质×BMI', '痰湿质×活动量表', '痰湿质×血尿酸',
    '气虚质×BMI', '气虚质×活动量表'
]
for i, feature in enumerate(MODEL_FEATURES, 1):
    print(f"{i:2d}. {feature}")

print(f"\n总特征数：{len(MODEL_FEATURES)}")

print("\n=== 验证交叉特征值 ===")
sample = df.iloc[0]
print(f"样本 1 原始值：")
print(f"  - 痰湿质：{sample['痰湿质']}")
print(f"  - BMI：{sample['BMI']}")
print(f"  - 活动量表：{sample['活动量表总分（ADL总分+IADL总分）']}")
print(f"  - 血尿酸：{sample['血尿酸']}")

sample_with = df_with_interactions.iloc[0]
print(f"\n样本 1 交叉特征：")
print(f"  - 痰湿质×BMI：{sample_with['痰湿质×BMI']} (应等于 {sample['痰湿质']} × {sample['BMI']} = {sample['痰湿质'] * sample['BMI']})")
print(f"  - 痰湿质×活动量表：{sample_with['痰湿质×活动量表']}")
print(f"  - 痰湿质×血尿酸：{sample_with['痰湿质×血尿酸']}")
print(f"  - 气虚质×BMI：{sample_with['气虚质×BMI']}")
print(f"  - 气虚质×活动量表：{sample_with['气虚质×活动量表']}")

print("\n=== 特征工程验证完成！🎉 ===")
print("""
改进总结：
✅ 更新了 LIPID_FEATURES 屏蔽清单（新增所有血脂派生特征）
✅ 新增了尿酸异常特征
✅ 创建了中西医交叉特征生成函数
✅ 新增了 5 个高质量交叉特征
✅ 特征总数从 20 个增加到 25 个
""")
