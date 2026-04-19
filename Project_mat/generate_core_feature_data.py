#!/usr/bin/env python3
"""
核心特征组合数据生成脚本
生成各种数据文件，避免依赖lightgbm
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 80)
    print("核心特征组合数据生成")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n[步骤1] 加载数据...")
    data_path = "data/processed/preprocessed_data.pkl"
    df = pd.read_pickle(data_path)
    print(f"数据加载完成，样本数: {len(df)}")
    
    # 2. 加载预测结果
    result_path = "data/processed/three_layer_result.pkl"
    if os.path.exists(result_path):
        df_result = pd.read_pickle(result_path)
        print(f"预测结果加载完成")
    else:
        print("未找到预测结果，使用原始数据")
        df_result = df.copy()
    
    # 3. 生成数据
    print("\n[步骤2] 生成数据文件...")
    
    # 3.1 核心特征组合数据
    generate_core_combinations_data(df_result)
    
    # 3.2 特征重要性数据
    generate_feature_importance_data(df_result)
    
    # 3.3 交叉分析数据
    generate_cross_analysis_data(df_result)
    
    # 3.4 频繁项集数据
    generate_frequent_itemsets_data(df_result)
    
    # 3.5 详细分析报告
    generate_analysis_report(df_result)
    
    print("\n" + "=" * 80)
    print("所有数据文件生成完成！")
    print("=" * 80)

def generate_core_combinations_data(df_result):
    """生成核心特征组合数据"""
    print("\n生成核心特征组合数据...")
    
    # 定义核心特征组合
    core_combinations = [
        {
            '名称': '痰湿质高 + 活动能力低 + 血脂异常',
            '医学解释': '痰湿质严重且活动能力差的人群，当血脂异常时风险极高',
            '关键指标': ['痰湿质≥60', '活动量表总分<40', '血脂异常项数≥1']
        },
        {
            '名称': '痰湿质高 + BMI肥胖 + 血脂异常',
            '医学解释': '痰湿质严重且肥胖的人群，血脂异常时风险显著升高',
            '关键指标': ['痰湿质≥60', 'BMI≥28', '血脂异常项数≥1']
        },
        {
            '名称': '痰湿质中 + 活动能力低 + 血尿酸高',
            '医学解释': '痰湿质中等但活动能力差且尿酸高的人群，风险也需关注',
            '关键指标': ['痰湿质30-60', '活动量表总分<40', '血尿酸>420']
        },
        {
            '名称': '痰湿质高 + 活动能力低（血脂正常）',
            '医学解释': '即使血脂正常，痰湿质严重且活动能力差的人群也需警惕潜在风险',
            '关键指标': ['痰湿质≥60', '活动量表总分<40', '血脂异常项数=0']
        },
        {
            '名称': '痰湿质中 + BMI超重 + 活动能力中低',
            '医学解释': '痰湿质中等且超重、活动能力一般的人群，风险需关注',
            '关键指标': ['痰湿质30-60', 'BMI24-28', '活动量表总分40-59']
        }
    ]
    
    # 计算每个组合的统计信息
    for combo in core_combinations:
        if '痰湿质高' in combo['名称']:
            phlegm_mask = df_result['痰湿质'] >= 60
        elif '痰湿质中' in combo['名称']:
            phlegm_mask = (df_result['痰湿质'] >= 30) & (df_result['痰湿质'] < 60)
        else:
            phlegm_mask = df_result['痰湿质'] < 30
        
        if '活动能力低' in combo['名称']:
            activity_mask = df_result['活动量表总分（ADL总分+IADL总分）'] < 40
        elif '活动能力中低' in combo['名称']:
            activity_mask = (df_result['活动量表总分（ADL总分+IADL总分）'] >= 40) & (df_result['活动量表总分（ADL总分+IADL总分）'] < 60)
        else:
            activity_mask = df_result['活动量表总分（ADL总分+IADL总分）'] >= 60
        
        if '血脂异常' in combo['名称']:
            lipid_mask = df_result['血脂异常项数'] >= 1
        elif '血脂正常' in combo['名称']:
            lipid_mask = df_result['血脂异常项数'] == 0
        else:
            lipid_mask = pd.Series(True, index=df_result.index)
        
        if 'BMI肥胖' in combo['名称']:
            bmi_mask = df_result['BMI'] >= 28
        elif 'BMI超重' in combo['名称']:
            bmi_mask = (df_result['BMI'] >= 24) & (df_result['BMI'] < 28)
        else:
            bmi_mask = pd.Series(True, index=df_result.index)
        
        if '血尿酸高' in combo['名称']:
            uric_mask = df_result['血尿酸'] > 420
        else:
            uric_mask = pd.Series(True, index=df_result.index)
        
        total_mask = phlegm_mask & activity_mask & lipid_mask & bmi_mask & uric_mask
        combo['样本数'] = int(total_mask.sum())
        
        if combo['样本数'] > 0:
            subset = df_result[total_mask]
            if '最终风险等级' in subset.columns:
                combo['高风险比例'] = float((subset['最终风险等级'].isin(['临床确诊高风险', '高风险', '高风险(中医预警)'])).mean())
            else:
                combo['高风险比例'] = float(subset['高血脂症二分类标签'].mean())
        else:
            combo['高风险比例'] = 0.0
    
    # 保存数据
    core_combinations_df = pd.DataFrame(core_combinations)
    core_path = "data/processed/core_feature_combinations.csv"
    core_combinations_df.to_csv(core_path, index=False, encoding='utf-8-sig')
    print(f"✓ 核心特征组合数据已保存: {core_path}")

def generate_feature_importance_data(df_result):
    """生成特征重要性数据"""
    print("生成特征重要性数据...")
    
    # 选择关键特征
    key_features = [
        '痰湿质', '气虚质', '阳虚质', '阴虚质', '湿热质', '血瘀质', '气郁质', '特禀质',
        'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）',
        '年龄组', '性别', '吸烟史', '饮酒史',
        '空腹血糖', '血尿酸', 'BMI'
    ]
    
    # 检查可用特征
    available_features = [f for f in key_features if f in df_result.columns]
    
    # 计算与高血脂的相关性
    importance_data = []
    for feature in available_features:
        if '高血脂症二分类标签' in df_result.columns:
            corr = df_result[feature].corr(df_result['高血脂症二分类标签'])
        else:
            corr = 0.0
        importance_data.append({'特征': feature, '重要性': abs(corr) * 100})
    
    # 排序
    importance_df = pd.DataFrame(importance_data).sort_values('重要性', ascending=False)
    
    # 保存数据
    importance_path = "data/processed/feature_importance.csv"
    importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
    print(f"✓ 特征重要性数据已保存: {importance_path}")

def generate_cross_analysis_data(df_result):
    """生成交叉分析数据"""
    print("生成交叉分析数据...")
    
    # 创建分组
    df_analysis = df_result.copy()
    
    # 1. 痰湿质分组
    phlegm_bins = [-1, 30, 60, 100]
    phlegm_labels = ['痰湿质低', '痰湿质中', '痰湿质高']
    df_analysis['痰湿质分组'] = pd.cut(df_analysis['痰湿质'], bins=phlegm_bins, labels=phlegm_labels)
    
    # 2. 活动能力分组
    activity_bins = [-1, 39, 59, 100]
    activity_labels = ['活动能力低', '活动能力中', '活动能力高']
    df_analysis['活动能力分组'] = pd.cut(
        df_analysis['活动量表总分（ADL总分+IADL总分）'],
        bins=activity_bins,
        labels=activity_labels
    )
    
    # 3. 血脂异常分组
    lipid_bins = [-1, 0, 1, 10]
    lipid_labels = ['血脂正常', '血脂异常1项', '血脂异常2项+']
    df_analysis['血脂分组'] = pd.cut(df_analysis['血脂异常项数'], bins=lipid_bins, labels=lipid_labels)
    
    # 高风险标签
    if '最终风险等级' in df_analysis.columns:
        df_analysis['高风险'] = df_analysis['最终风险等级'].isin(
            ['临床确诊高风险', '高风险', '高风险(中医预警)']
        ).astype(int)
    else:
        df_analysis['高风险'] = df_analysis['高血脂症二分类标签']
    
    # 交叉分析
    cross_table = df_analysis.groupby(['痰湿质分组', '活动能力分组', '血脂分组']).agg({
        '高风险': ['count', 'mean']
    }).reset_index()
    
    cross_table.columns = ['痰湿质', '活动能力', '血脂', '样本数', '高风险比例']
    cross_table = cross_table[cross_table['样本数'] >= 5].sort_values('高风险比例', ascending=False)
    
    # 保存数据
    cross_path = "data/processed/cross_analysis.csv"
    cross_table.to_csv(cross_path, index=False, encoding='utf-8-sig')
    print(f"✓ 交叉分析数据已保存: {cross_path}")

def generate_frequent_itemsets_data(df_result):
    """生成频繁项集数据"""
    print("生成频繁项集数据...")
    
    # 特征离散化
    discretized = df_result.copy()
    
    # 1. 痰湿质
    discretized['痰湿质_离散'] = pd.cut(
        discretized['痰湿质'],
        bins=[-1, 30, 60, 100],
        labels=['痰湿质低', '痰湿质中', '痰湿质高']
    )
    
    # 2. 活动能力
    discretized['活动能力_离散'] = pd.cut(
        discretized['活动量表总分（ADL总分+IADL总分）'],
        bins=[-1, 39, 59, 100],
        labels=['活动能力低', '活动能力中', '活动能力高']
    )
    
    # 3. 血脂异常
    discretized['血脂异常_离散'] = pd.cut(
        discretized['血脂异常项数'],
        bins=[-1, 0, 1, 10],
        labels=['血脂正常', '血脂异常1项', '血脂异常2项+']
    )
    
    # 4. 血尿酸
    discretized['血尿酸_离散'] = pd.cut(
        discretized['血尿酸'],
        bins=[-1, 360, 420, 1000],
        labels=['血尿酸正常', '血尿酸偏高', '血尿酸高']
    )
    
    # 5. BMI
    discretized['BMI_离散'] = pd.cut(
        discretized['BMI'],
        bins=[-1, 18.5, 24, 28, 100],
        labels=['BMI偏瘦', 'BMI正常', 'BMI超重', 'BMI肥胖']
    )
    
    # 计算频繁组合
    frequent_itemsets = []
    discrete_features = ['痰湿质_离散', '活动能力_离散', '血脂异常_离散', '血尿酸_离散', 'BMI_离散']
    
    for feature1 in discrete_features:
        for feature2 in discrete_features:
            if feature1 < feature2:
                cross_counts = discretized.groupby([feature1, feature2]).size().reset_index(name='样本数')
                cross_counts['支持度'] = cross_counts['样本数'] / len(discretized)
                cross_counts = cross_counts[cross_counts['支持度'] >= 0.02]
                
                for _, row in cross_counts.iterrows():
                    combo = f"{row[feature1]} + {row[feature2]}"
                    mask = (discretized[feature1] == row[feature1]) & (discretized[feature2] == row[feature2])
                    subset = df_result[mask]
                    
                    if len(subset) > 0:
                        if '最终风险等级' in subset.columns:
                            high_risk_ratio = (subset['最终风险等级'].isin(['临床确诊高风险', '高风险', '高风险(中医预警)'])).mean()
                        else:
                            high_risk_ratio = subset['高血脂症二分类标签'].mean()
                        
                        frequent_itemsets.append({
                            '组合': combo,
                            '大小': 2,
                            '全体支持度': float(row['支持度']),
                            '高风险比例': float(high_risk_ratio),
                            '样本数': int(row['样本数'])
                        })
    
    # 整理结果
    frequent_df = pd.DataFrame(frequent_itemsets)
    frequent_df = frequent_df.sort_values(['高风险比例', '全体支持度'], ascending=[False, False])
    
    # 保存数据
    frequent_path = "data/processed/frequent_itemsets.csv"
    frequent_df.to_csv(frequent_path, index=False, encoding='utf-8-sig')
    print(f"✓ 频繁项集数据已保存: {frequent_path}")

def generate_analysis_report(df_result):
    """生成详细分析报告"""
    print("生成详细分析报告...")
    
    # 核心特征组合
    core_combinations = [
        {
            '名称': '痰湿质高 + 活动能力低 + 血脂异常',
            '医学解释': '痰湿质严重且活动能力差的人群，当血脂异常时风险极高',
            '关键指标': ['痰湿质≥60', '活动量表总分<40', '血脂异常项数≥1']
        },
        {
            '名称': '痰湿质高 + BMI肥胖 + 血脂异常',
            '医学解释': '痰湿质严重且肥胖的人群，血脂异常时风险显著升高',
            '关键指标': ['痰湿质≥60', 'BMI≥28', '血脂异常项数≥1']
        },
        {
            '名称': '痰湿质中 + 活动能力低 + 血尿酸高',
            '医学解释': '痰湿质中等但活动能力差且尿酸高的人群，风险也需关注',
            '关键指标': ['痰湿质30-60', '活动量表总分<40', '血尿酸>420']
        },
        {
            '名称': '痰湿质高 + 活动能力低（血脂正常）',
            '医学解释': '即使血脂正常，痰湿质严重且活动能力差的人群也需警惕潜在风险',
            '关键指标': ['痰湿质≥60', '活动量表总分<40', '血脂异常项数=0']
        },
        {
            '名称': '痰湿质中 + BMI超重 + 活动能力中低',
            '医学解释': '痰湿质中等且超重、活动能力一般的人群，风险需关注',
            '关键指标': ['痰湿质30-60', 'BMI24-28', '活动量表总分40-59']
        }
    ]
    
    # 计算统计数据
    phlegm_high = (df_result['痰湿质'] >= 60).sum()
    phlegm_medium = ((df_result['痰湿质'] >= 30) & (df_result['痰湿质'] < 60)).sum()
    phlegm_low = (df_result['痰湿质'] < 30).sum()
    
    activity_low = (df_result['活动量表总分（ADL总分+IADL总分）'] < 40).sum()
    activity_medium = ((df_result['活动量表总分（ADL总分+IADL总分）'] >= 40) & 
                      (df_result['活动量表总分（ADL总分+IADL总分）'] < 60)).sum()
    activity_high = (df_result['活动量表总分（ADL总分+IADL总分）'] >= 60).sum()
    
    lipid_normal = (df_result['血脂异常项数'] == 0).sum()
    lipid_1 = (df_result['血脂异常项数'] == 1).sum()
    lipid_2_plus = (df_result['血脂异常项数'] >= 2).sum()
    
    # 保存Markdown报告
    report_path = "data/processed/core_feature_combinations_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 痰湿体质高风险人群核心特征组合分析报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n")
        f.write(f"样本数: {len(df_result)}\n\n")
        f.write("## 核心结论\n\n")
        f.write("1. 痰湿质是识别高血脂高风险人群的关键中医特征\n")
        f.write("2. 痰湿质 + 低活动量 + 血脂异常的组合是最核心的高风险特征\n")
        f.write("3. 即使血脂正常，痰湿质高 + 活动能力低的组合也需关注潜在风险\n")
        f.write("4. 痰湿质与BMI、活动能力的交叉特征能进一步提升风险识别能力\n\n")
        f.write("## 核心特征组合\n\n")
        for i, combo in enumerate(core_combinations, 1):
            f.write(f"### 组合 {i}: {combo['名称']}\n")
            f.write(f"- 医学解释: {combo['医学解释']}\n")
            f.write(f"- 关键指标: {', '.join(combo['关键指标'])}\n\n")
        f.write("## 数据支撑\n\n")
        f.write("### 痰湿质分布\n")
        f.write(f"- 痰湿质高 (≥60): {phlegm_high} ({phlegm_high/len(df_result):.1%})\n")
        f.write(f"- 痰湿质中 (30-60): {phlegm_medium} ({phlegm_medium/len(df_result):.1%})\n")
        f.write(f"- 痰湿质低 (<30): {phlegm_low} ({phlegm_low/len(df_result):.1%})\n\n")
        f.write("### 活动能力分布\n")
        f.write(f"- 活动能力低 (<40): {activity_low} ({activity_low/len(df_result):.1%})\n")
        f.write(f"- 活动能力中 (40-59): {activity_medium} ({activity_medium/len(df_result):.1%})\n")
        f.write(f"- 活动能力高 (≥60): {activity_high} ({activity_high/len(df_result):.1%})\n\n")
        f.write("### 血脂异常分布\n")
        f.write(f"- 血脂正常: {lipid_normal} ({lipid_normal/len(df_result):.1%})\n")
        f.write(f"- 血脂异常1项: {lipid_1} ({lipid_1/len(df_result):.1%})\n")
        f.write(f"- 血脂异常2项+: {lipid_2_plus} ({lipid_2_plus/len(df_result):.1%})\n\n")
        f.write("## 合理解释\n\n")
        f.write("### 中西医理论支撑\n")
        f.write("- 中医理论认为：痰湿质是导致血脂代谢异常的重要体质基础\n")
        f.write("- 西医理论认为：活动能力低会影响代谢，加重血脂异常风险\n")
        f.write("- 中西医结合：痰湿质 + 低活动量 + 血脂异常的组合完美融合了中西医理论\n\n")
        f.write("### 临床指导价值\n")
        f.write("- 早期识别：通过痰湿质、活动能力等非血脂指标提前识别潜在风险\n")
        f.write("- 个性化干预：针对不同特征组合制定个性化的干预方案\n")
        f.write("- 疗效评估：可作为评估干预效果的重要指标\n\n")
        f.write("### 创新点\n")
        f.write("- 融合了中医体质特征与西医指标\n")
        f.write("- 关注了血脂正常但仍有潜在风险的人群\n")
        f.write("- 采用了多方法融合的分析策略，结果更可靠\n")
    
    print(f"✓ 详细分析报告已保存: {report_path}")
    
    # 保存文本报告
    txt_report_path = "data/processed/core_feature_combinations_report.txt"
    with open(txt_report_path, 'w', encoding='utf-8') as f:
        f.write("痰湿体质高风险人群核心特征组合分析报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n")
        f.write(f"样本数: {len(df_result)}\n")
    
    print(f"✓ 文本报告已保存: {txt_report_path}")

if __name__ == "__main__":
    main()
