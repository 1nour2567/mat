#!/usr/bin/env python3
# 分析性别和年龄差异
import pandas as pd
import numpy as np
import sys

sys.path.append('/workspace/Project_mat')

from src.feature_engineering import analyze_constitution_contribution

def analyze_gender_differences(df):
    """
    分析不同性别之间的体质贡献度差异
    """
    print("\n=== 性别差异分析 ===")
    
    # 按性别分组
    gender_groups = df.groupby('性别')
    
    for gender, group in gender_groups:
        gender_name = '女' if gender == 0 else '男'
        print(f"\n{gender_name}（性别代码：{gender}）体质贡献度分析：")
        
        # 分析体质贡献度
        contributions = analyze_constitution_contribution(group, '高血脂症二分类标签')
        
        if contributions:
            print(f"{gender_name}群体中贡献度最高的体质：{contributions[0][0]}（{contributions[0][1]:.4f}）")

def analyze_age_differences(df):
    """
    分析不同年龄组之间的体质贡献度差异
    """
    print("\n=== 年龄组差异分析 ===")
    
    # 按年龄组分组
    age_groups = df.groupby('年龄组')
    
    for age_group, group in age_groups:
        age_group_name = group['年龄组名称'].iloc[0] if '年龄组名称' in group.columns else f"年龄组{age_group}"
        print(f"\n{age_group_name}（年龄组代码：{age_group}）体质贡献度分析：")
        
        # 分析体质贡献度
        contributions = analyze_constitution_contribution(group, '高血脂症二分类标签')
        
        if contributions:
            print(f"{age_group_name}群体中贡献度最高的体质：{contributions[0][0]}（{contributions[0][1]:.4f}）")

def main():
    """
    主分析函数
    """
    print("=== 性别和年龄差异分析 ===")
    
    # 加载分析后的数据
    data_path = '/workspace/Project_mat/data/processed/analyzed_data_with_constitution_rules.pkl'
    try:
        df = pd.read_pickle(data_path)
        print(f"成功加载数据，样本数：{len(df)}")
    except Exception as e:
        print(f"加载数据失败：{e}")
        return
    
    # 检查必要的列
    required_cols = ['性别', '年龄组', '高血脂症二分类标签']
    for col in required_cols:
        if col not in df.columns:
            print(f"缺少必要的列：{col}")
            return
    
    # 分析性别差异
    analyze_gender_differences(df)
    
    # 分析年龄差异
    analyze_age_differences(df)

if __name__ == "__main__":
    main()
