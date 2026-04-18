#!/usr/bin/env python3
# 基于新体质判断规则的分析脚本
import pandas as pd
import numpy as np
import sys

sys.path.append('/workspace/Project_mat')

from src.preprocessing import load_raw_data, clean_data, feature_derivation
from src.feature_engineering import calculate_spearman_correlation, calculate_mutual_info, calculate_pls_loadings, entropy_weight_method
from src.feature_engineering import analyze_constitution_contribution

def reclassify_constitution(df):
    """
    根据新规则重新判断体质
    规则：
    1. 平和质：平和分达到阈值（假设≥60），同时其他失衡体质分数不能太高（假设≤40）
    2. 偏颇体质：分数最高且达到成立阈值（假设≥60）
    """
    constitution_cols = ['平和质', '气虚质', '阳虚质', '阴虚质', 
                       '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
    
    # 体质标签映射
    label_mapping = {
        '平和质': 1, '气虚质': 2, '阳虚质': 3, '阴虚质': 4,
        '痰湿质': 5, '湿热质': 6, '血瘀质': 7, '气郁质': 8, '特禀质': 9
    }
    
    reclassified_labels = []
    
    for idx, row in df.iterrows():
        # 检查平和质条件
        if row['平和质'] >= 60:
            other_scores = [row[col] for col in constitution_cols if col != '平和质']
            if all(score <= 40 for score in other_scores):
                reclassified_labels.append(label_mapping['平和质'])
                continue
        
        # 检查偏颇体质条件
        max_score = row[constitution_cols].max()
        max_constitution = row[constitution_cols].idxmax()
        if max_score >= 60:
            reclassified_labels.append(label_mapping[max_constitution])
        else:
            # 没有达到任何体质的成立阈值
            reclassified_labels.append(0)  # 0表示未确定体质
    
    df['重新判断体质标签'] = reclassified_labels
    return df

def analyze_blood_routine_indicators(df):
    """
    分析血常规体检指标与痰湿体质和高血脂的关系
    """
    print("\n=== 血常规体检指标分析 ===")
    
    # 血常规相关指标
    blood_indicators = ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 
                       'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI']
    
    # 派生指标
    derived_indicators = ['non-HDL-C', 'AIP', 'TC/HDL比值', 'LDL/HDL比值', 'TG/HDL比值']
    
    all_indicators = blood_indicators + derived_indicators
    
    # 计算Spearman相关系数（痰湿表征能力）
    print("\n1. 血常规指标与痰湿质的Spearman相关系数：")
    spearman_scores = calculate_spearman_correlation(df, all_indicators, '痰湿质')
    sorted_spearman = sorted(spearman_scores.items(), key=lambda x: x[1], reverse=True)
    for indicator, score in sorted_spearman:
        print(f"   {indicator}: {score:.4f}")
    
    # 计算互信息（风险预警能力）
    print("\n2. 血常规指标的互信息（风险预警能力）：")
    mi_scores = calculate_mutual_info(df, all_indicators, '高血脂症二分类标签')
    sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
    for indicator, score in sorted_mi:
        print(f"   {indicator}: {score:.4f}")
    
    # 计算PLS联合结构载荷（双目标整合能力）
    print("\n3. 血常规指标的PLS联合结构载荷：")
    pls_scores = calculate_pls_loadings(df, all_indicators, '痰湿质', '高血脂症二分类标签')
    sorted_pls = sorted(pls_scores.items(), key=lambda x: x[1], reverse=True)
    for indicator, score in sorted_pls:
        print(f"   {indicator}: {score:.4f}")
    
    # 熵权法综合评分
    print("\n4. 熵权法综合评分：")
    weights = entropy_weight_method(spearman_scores, mi_scores, pls_scores)
    
    combined_scores = {}
    for indicator in all_indicators:
        if indicator in spearman_scores and indicator in mi_scores and indicator in pls_scores:
            score = (weights['spearman'] * spearman_scores[indicator] +
                    weights['mi'] * mi_scores[indicator] +
                    weights['pls'] * pls_scores[indicator])
            combined_scores[indicator] = score
    
    sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    print("\n5. 血常规指标综合排名：")
    for i, (indicator, score) in enumerate(sorted_combined[:10]):
        print(f"   {i+1}. {indicator}: {score:.4f}")
    
    return sorted_combined

def analyze_activity_scale(df):
    """
    分析中老年人活动量表评分与痰湿体质和高血脂的关系
    """
    print("\n=== 中老年人活动量表评分分析 ===")
    
    # 活动量表相关指标
    activity_indicators = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                         'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                         '活动量表总分（ADL总分+IADL总分）', '活动能力分层']
    
    # 计算Spearman相关系数（痰湿表征能力）
    print("\n1. 活动量表与痰湿质的Spearman相关系数：")
    spearman_scores = calculate_spearman_correlation(df, activity_indicators, '痰湿质')
    sorted_spearman = sorted(spearman_scores.items(), key=lambda x: x[1], reverse=True)
    for indicator, score in sorted_spearman:
        print(f"   {indicator}: {score:.4f}")
    
    # 计算互信息（风险预警能力）
    print("\n2. 活动量表的互信息（风险预警能力）：")
    mi_scores = calculate_mutual_info(df, activity_indicators, '高血脂症二分类标签')
    sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
    for indicator, score in sorted_mi:
        print(f"   {indicator}: {score:.4f}")
    
    # 计算PLS联合结构载荷（双目标整合能力）
    print("\n3. 活动量表的PLS联合结构载荷：")
    pls_scores = calculate_pls_loadings(df, activity_indicators, '痰湿质', '高血脂症二分类标签')
    sorted_pls = sorted(pls_scores.items(), key=lambda x: x[1], reverse=True)
    for indicator, score in sorted_pls:
        print(f"   {indicator}: {score:.4f}")
    
    # 熵权法综合评分
    print("\n4. 熵权法综合评分：")
    weights = entropy_weight_method(spearman_scores, mi_scores, pls_scores)
    
    combined_scores = {}
    for indicator in activity_indicators:
        if indicator in spearman_scores and indicator in mi_scores and indicator in pls_scores:
            score = (weights['spearman'] * spearman_scores[indicator] +
                    weights['mi'] * mi_scores[indicator] +
                    weights['pls'] * pls_scores[indicator])
            combined_scores[indicator] = score
    
    sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    print("\n5. 活动量表综合排名：")
    for i, (indicator, score) in enumerate(sorted_combined[:10]):
        print(f"   {i+1}. {indicator}: {score:.4f}")
    
    return sorted_combined

def main():
    """主分析函数"""
    print("=== 基于新体质判断规则的分析 ===")
    
    # 加载数据
    raw_data_path = '/workspace/Project_mat/data/raw/附件1：样例数据.xlsx'
    df = load_raw_data(raw_data_path)
    
    # 数据清洗
    df = clean_data(df)
    
    # 特征衍生
    df = feature_derivation(df)
    
    # 处理字符串类型的分层变量
    if '活动能力分层' in df.columns:
        activity_mapping = {'<40': 0, '40-59': 1, '≥60': 2}
        df['活动能力分层'] = df['活动能力分层'].map(activity_mapping).fillna(0)
    
    if '痰湿积分分层' in df.columns:
        phlegm_mapping = {'≤58': 0, '59-61': 1, '≥62': 2}
        df['痰湿积分分层'] = df['痰湿积分分层'].map(phlegm_mapping).fillna(0)
    
    # 基于新规则重新判断体质
    print("\n=== 基于新规则重新判断体质 ===")
    df = reclassify_constitution(df)
    
    # 分析血常规体检指标
    blood_indicators_ranking = analyze_blood_routine_indicators(df)
    
    # 分析中老年人活动量表评分
    activity_indicators_ranking = analyze_activity_scale(df)
    
    # 九种体质风险贡献度分析（使用重新判断的体质标签）
    print("\n=== 九种体质风险贡献度分析 ===")
    
    # 保存结果
    output_path = '/workspace/Project_mat/data/processed/analyzed_data_with_constitution_rules.pkl'
    df.to_pickle(output_path)
    print(f"\n分析结果已保存到: {output_path}")
    
    # 输出关键发现
    print("\n=== 关键发现 ===")
    print("1. 血常规体检指标中最能表征痰湿体质和预警高血脂的关键指标：")
    for i, (indicator, score) in enumerate(blood_indicators_ranking[:5]):
        print(f"   {i+1}. {indicator}: {score:.4f}")
    
    print("\n2. 中老年人活动量表评分中最能表征痰湿体质和预警高血脂的关键指标：")
    for i, (indicator, score) in enumerate(activity_indicators_ranking[:5]):
        print(f"   {i+1}. {indicator}: {score:.4f}")
    
    print("\n3. 体质判断规则更新：")
    print("   - 平和质：平和分≥60且其他体质分≤40")
    print("   - 偏颇体质：分数最高且≥60")
    print("   - 未达到阈值的样本标记为未确定体质")

if __name__ == "__main__":
    main()
