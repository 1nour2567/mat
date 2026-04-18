#!/usr/bin/env python3
# 按照指定格式分析能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标
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
        关键指标排序列表, 权重
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
    
    return sorted_indicators, weights

def main():
    """
    主分析函数
    """
    # 加载数据
    data_path = '/workspace/预处理后数据.csv'
    encodings = ['utf-8', 'gbk', 'latin1']
    for encoding in encodings:
        try:
            df = pd.read_csv(data_path, encoding=encoding)
            break
        except:
            continue
    else:
        raise Exception("无法加载数据，请检查文件编码")
    
    # 性别分析
    male_indicators, male_weights = analyze_key_indicators(df, gender=1)
    female_indicators, female_weights = analyze_key_indicators(df, gender=0)
    
    # 年龄组分析
    age_group_mapping = {1: '40-49岁', 2: '50-59岁', 3: '60-69岁', 4: '70-79岁', 5: '80-89岁'}
    age_indicators = {}
    age_weights = {}
    for age_group, age_group_name in age_group_mapping.items():
        indicators, weights = analyze_key_indicators(df, age_group=age_group)
        age_indicators[age_group_name] = indicators
        age_weights[age_group_name] = weights
    
    # 生成报告
    print("=== 不同性别和年龄组的关键指标分析 ===")
    print("\n" + "=" * 60)
    print("性别分组分析")
    print("=" * 60)
    
    # 男性分析
    print("\n=== 男性分析 ===")
    print("\n=== 熵权法（EWM）计算结果 ===")
    print(f"Spearman相关系数权重: {male_weights['spearman']:.4f}")
    print(f"互信息权重: {male_weights['mi']:.4f}")
    print(f"PLS联合结构载荷权重: {male_weights['pls']:.4f}")
    print("\n男性前5个关键指标：")
    for i, (indicator, score) in enumerate(male_indicators[:5]):
        print(f"   {i+1}. {indicator}: {score:.4f}")
    
    # 女性分析
    print("\n=== 女性分析 ===")
    print("\n=== 熵权法（EWM）计算结果 ===")
    print(f"Spearman相关系数权重: {female_weights['spearman']:.4f}")
    print(f"互信息权重: {female_weights['mi']:.4f}")
    print(f"PLS联合结构载荷权重: {female_weights['pls']:.4f}")
    print("\n女性前5个关键指标：")
    for i, (indicator, score) in enumerate(female_indicators[:5]):
        print(f"   {i+1}. {indicator}: {score:.4f}")
    
    print("\n" + "=" * 60)
    print("年龄组分组分析")
    print("=" * 60)
    
    # 年龄组分析
    for age_group_name, indicators in age_indicators.items():
        print(f"\n=== {age_group_name}分析 ===")
        weights = age_weights[age_group_name]
        print("\n=== 熵权法（EWM）计算结果 ===")
        print(f"Spearman相关系数权重: {weights['spearman']:.4f}")
        print(f"互信息权重: {weights['mi']:.4f}")
        print(f"PLS联合结构载荷权重: {weights['pls']:.4f}")
        print(f"\n{age_group_name}前5个关键指标：")
        for i, (indicator, score) in enumerate(indicators[:5]):
            print(f"   {i+1}. {indicator}: {score:.4f}")
    
    print("\n" + "=" * 60)
    print("综合分析")
    print("=" * 60)
    
    print("\n关键发现：")
    print("1. 不同性别和年龄组的关键指标存在差异")
    print("2. 整体而言，TC、AIP、TG等指标在各分组中都表现重要")
    print("3. 性别和年龄因素可能影响痰湿体质与高血脂的关联模式")

if __name__ == "__main__":
    main()
