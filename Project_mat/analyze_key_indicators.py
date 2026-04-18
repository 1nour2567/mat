#!/usr/bin/env python3
# 分析能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标
import pandas as pd
import numpy as np
import sys

sys.path.append('/workspace/Project_mat')

from src.feature_engineering import calculate_spearman_correlation, calculate_mutual_info, calculate_pls_loadings, entropy_weight_method

def analyze_key_indicators(df, gender=None, age_group=None):
    """
    分析能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标
    
    Args:
        df: 数据框
        gender: 性别（0=女，1=男），None表示所有性别
        age_group: 年龄组代码，None表示所有年龄组
    
    Returns:
        关键指标排序列表
    """
    # 筛选数据
    filtered_df = df.copy()
    if gender is not None:
        filtered_df = filtered_df[filtered_df['性别'] == gender]
    if age_group is not None:
        filtered_df = filtered_df[filtered_df['年龄组'] == age_group]
    
    # 血常规相关指标
    blood_indicators = ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 
                       'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI']
    
    # 派生指标
    derived_indicators = ['non-HDL-C', 'AIP', 'TC/HDL比值', 'LDL/HDL比值', 'TG/HDL比值']
    
    all_indicators = blood_indicators + derived_indicators
    
    # 计算Spearman相关系数（痰湿表征能力）
    spearman_scores = calculate_spearman_correlation(filtered_df, all_indicators, '痰湿质')
    
    # 计算互信息（风险预警能力）
    mi_scores = calculate_mutual_info(filtered_df, all_indicators, '高血脂症二分类标签')
    
    # 计算PLS联合结构载荷（双目标整合能力）
    pls_scores = calculate_pls_loadings(filtered_df, all_indicators, '痰湿质', '高血脂症二分类标签')
    
    # 熵权法综合评分
    weights = entropy_weight_method(spearman_scores, mi_scores, pls_scores)
    
    # 计算综合得分
    common_features = list(set(spearman_scores.keys()) & set(mi_scores.keys()) & set(pls_scores.keys()))
    combined_scores = {}
    
    for feature in common_features:
        score = (weights['spearman'] * spearman_scores[feature] +
                weights['mi'] * mi_scores[feature] +
                weights['pls'] * pls_scores[feature])
        combined_scores[feature] = score
    
    # 排序
    sorted_indicators = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_indicators

def main():
    """
    主分析函数
    """
    print("=== 分析能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标 ===")
    
    # 加载数据
    data_path = '/workspace/预处理后数据.csv'
    encodings = ['utf-8', 'gbk', 'latin1']
    for encoding in encodings:
        try:
            df = pd.read_csv(data_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码加载数据")
            break
        except:
            continue
    else:
        raise Exception("无法加载数据，请检查文件编码")
    
    # 1. 总体分析
    print("\n1. 总体分析（所有人群）：")
    overall_indicators = analyze_key_indicators(df)
    print("前10个关键指标：")
    for i, (indicator, score) in enumerate(overall_indicators[:10]):
        print(f"   {i+1}. {indicator}: {score:.4f}")
    
    # 2. 性别差异分析
    print("\n2. 性别差异分析：")
    for gender in [0, 1]:
        gender_name = '女' if gender == 0 else '男'
        print(f"\n{gender_name}群体：")
        gender_indicators = analyze_key_indicators(df, gender=gender)
        print("前10个关键指标：")
        for i, (indicator, score) in enumerate(gender_indicators[:10]):
            print(f"   {i+1}. {indicator}: {score:.4f}")
    
    # 3. 年龄组差异分析
    print("\n3. 年龄组差异分析：")
    age_group_mapping = {1: '40-49岁', 2: '50-59岁', 3: '60-69岁', 4: '70-79岁', 5: '80-89岁'}
    for age_group, age_group_name in age_group_mapping.items():
        print(f"\n{age_group_name}群体：")
        age_indicators = analyze_key_indicators(df, age_group=age_group)
        print("前10个关键指标：")
        for i, (indicator, score) in enumerate(age_indicators[:10]):
            print(f"   {i+1}. {indicator}: {score:.4f}")
    
    # 4. 分析结论
    print("\n=== 分析结论 ===")
    print("1. 总体而言，最能表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    print(f"   - 第1名：{overall_indicators[0][0]}（{overall_indicators[0][1]:.4f}）")
    print(f"   - 第2名：{overall_indicators[1][0]}（{overall_indicators[1][1]:.4f}）")
    print(f"   - 第3名：{overall_indicators[2][0]}（{overall_indicators[2][1]:.4f}）")
    
    # 比较男女差异
    female_top = analyze_key_indicators(df, gender=0)[:3]
    male_top = analyze_key_indicators(df, gender=1)[:3]
    print("\n2. 性别差异：")
    print("   女性群体前3名指标：")
    for i, (indicator, score) in enumerate(female_top):
        print(f"      {i+1}. {indicator}: {score:.4f}")
    print("   男性群体前3名指标：")
    for i, (indicator, score) in enumerate(male_top):
        print(f"      {i+1}. {indicator}: {score:.4f}")
    
    # 比较年龄组差异
    print("\n3. 年龄组差异：")
    for age_group, age_group_name in age_group_mapping.items():
        age_top = analyze_key_indicators(df, age_group=age_group)[:3]
        print(f"   {age_group_name}前3名指标：")
        for i, (indicator, score) in enumerate(age_top):
            print(f"      {i+1}. {indicator}: {score:.4f}")

if __name__ == "__main__":
    main()
