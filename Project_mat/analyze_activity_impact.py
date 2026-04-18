#!/usr/bin/env python3
# 研究活动量对高血脂症的影响（男女和不同年龄组）
import pandas as pd
import numpy as np
import sys

sys.path.append('/workspace/Project_mat')

from src.feature_engineering import calculate_spearman_correlation, calculate_mutual_info, calculate_pls_loadings, entropy_weight_method

def analyze_activity_indicators(df, gender=None, age_group=None):
    """
    分析活动量指标对高血脂症的影响
    
    Args:
        df: 数据框
        gender: 性别（0=女，1=男），None表示所有性别
        age_group: 年龄组代码，None表示所有年龄组
    
    Returns:
        活动量指标排序列表, 权重
    """
    # 筛选数据
    filtered_df = df.copy()
    if gender is not None:
        filtered_df = filtered_df[filtered_df['性别'] == gender]
    if age_group is not None:
        filtered_df = filtered_df[filtered_df['年龄组'] == age_group]
    
    # 活动量指标
    activity_indicators = [
        'ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
        'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
        '活动量表总分（ADL总分+IADL总分）'
    ]
    
    # 计算Spearman相关系数（痰湿表征能力）
    spearman_scores = calculate_spearman_correlation(filtered_df, activity_indicators, '痰湿质')
    
    # 计算互信息（风险预警能力）
    mi_scores = calculate_mutual_info(filtered_df, activity_indicators, '高血脂症二分类标签')
    
    # 计算PLS联合结构载荷（双目标整合能力）
    pls_scores = calculate_pls_loadings(filtered_df, activity_indicators, '痰湿质', '高血脂症二分类标签')
    
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
    print("=== 研究活动量对高血脂症的影响 ===")
    
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
    
    # 检查必要的列
    required_cols = ['性别', '年龄组', '高血脂症二分类标签', '痰湿质']
    activity_cols = [
        'ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
        'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
        '活动量表总分（ADL总分+IADL总分）'
    ]
    
    for col in required_cols + activity_cols:
        if col not in df.columns:
            print(f"缺少必要的列：{col}")
            return
    
    # 总体分析
    print("\n=== 总体分析（所有人群） ===")
    overall_indicators, overall_weights = analyze_activity_indicators(df)
    print("\n=== 熵权法（EWM）计算结果 ===")
    print(f"Spearman相关系数权重: {overall_weights['spearman']:.4f}")
    print(f"互信息权重: {overall_weights['mi']:.4f}")
    print(f"PLS联合结构载荷权重: {overall_weights['pls']:.4f}")
    print("\n总体前10个关键指标：")
    for i, (indicator, score) in enumerate(overall_indicators[:10]):
        print(f"   {i+1}. {indicator}: {score:.4f}")
    
    # 性别分析
    print("\n" + "=" * 60)
    print("性别分组分析")
    print("=" * 60)
    
    for gender in [0, 1]:
        gender_name = '女' if gender == 0 else '男'
        print(f"\n=== {gender_name}分析 ===")
        indicators, weights = analyze_activity_indicators(df, gender=gender)
        print("\n=== 熵权法（EWM）计算结果 ===")
        print(f"Spearman相关系数权重: {weights['spearman']:.4f}")
        print(f"互信息权重: {weights['mi']:.4f}")
        print(f"PLS联合结构载荷权重: {weights['pls']:.4f}")
        print(f"\n{gender_name}前10个关键指标：")
        for i, (indicator, score) in enumerate(indicators[:10]):
            print(f"   {i+1}. {indicator}: {score:.4f}")
    
    # 年龄组分析
    print("\n" + "=" * 60)
    print("年龄组分组分析")
    print("=" * 60)
    
    age_group_mapping = {1: '40-49岁', 2: '50-59岁', 3: '60-69岁', 4: '70-79岁', 5: '80-89岁'}
    for age_group, age_group_name in age_group_mapping.items():
        print(f"\n=== {age_group_name}分析 ===")
        indicators, weights = analyze_activity_indicators(df, age_group=age_group)
        print("\n=== 熵权法（EWM）计算结果 ===")
        print(f"Spearman相关系数权重: {weights['spearman']:.4f}")
        print(f"互信息权重: {weights['mi']:.4f}")
        print(f"PLS联合结构载荷权重: {weights['pls']:.4f}")
        print(f"\n{age_group_name}前10个关键指标：")
        for i, (indicator, score) in enumerate(indicators[:10]):
            print(f"   {i+1}. {indicator}: {score:.4f}")
    
    # 综合分析
    print("\n" + "=" * 60)
    print("综合分析")
    print("=" * 60)
    
    print("\n关键发现：")
    print("1. 总体而言，最能表征痰湿体质和预警高血脂的活动量指标：")
    print(f"   - 第1名：{overall_indicators[0][0]}（{overall_indicators[0][1]:.4f}）")
    print(f"   - 第2名：{overall_indicators[1][0]}（{overall_indicators[1][1]:.4f}）")
    print(f"   - 第3名：{overall_indicators[2][0]}（{overall_indicators[2][1]:.4f}）")
    
    # 性别差异比较
    female_indicators, _ = analyze_activity_indicators(df, gender=0)
    male_indicators, _ = analyze_activity_indicators(df, gender=1)
    
    print("\n2. 性别差异：")
    print("   女性群体前3名指标：")
    for i, (indicator, score) in enumerate(female_indicators[:3]):
        print(f"      {i+1}. {indicator}: {score:.4f}")
    print("   男性群体前3名指标：")
    for i, (indicator, score) in enumerate(male_indicators[:3]):
        print(f"      {i+1}. {indicator}: {score:.4f}")
    
    # 年龄组差异比较
    print("\n3. 年龄组差异：")
    for age_group, age_group_name in age_group_mapping.items():
        age_indicators, _ = analyze_activity_indicators(df, age_group=age_group)
        print(f"   {age_group_name}前3名指标：")
        for i, (indicator, score) in enumerate(age_indicators[:3]):
            print(f"      {i+1}. {indicator}: {score:.4f}")
    
    print("\n4. 结论：")
    print("   - 活动量对高血脂症的影响在不同性别和年龄组之间存在差异")
    print("   - 不同人群的关键活动量指标有所不同，需要针对性地进行干预")
    print("   - 性别和年龄因素也会影响活动量与高血脂的关联模式")

if __name__ == "__main__":
    main()
