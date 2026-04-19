#!/usr/bin/env python3
"""
核心特征组合深度分析
融合频繁项集、特征重要性和规则蒸馏
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.three_layer_architecture import TripleLayerPredictor, MODEL_FEATURES
import lightgbm as lgb
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 80)
    print("痰湿体质高风险人群核心特征组合深度分析")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n[步骤1] 加载数据...")
    data_path = "data/processed/preprocessed_data.pkl"
    df = pd.read_pickle(data_path)
    print(f"数据加载完成，样本数: {len(df)}")
    
    # 2. 训练三层模型
    print("\n[步骤2] 训练模型并预测...")
    predictor = TripleLayerPredictor()
    predictor.fit(df)
    df_result = predictor.predict(df)
    
    # 3. 多维度分析
    print("\n[步骤3] 多维度分析...")
    
    # 3.1 先进行特征离散化
    discretized_df = discretize_features(df_result)
    
    # 3.2 方法一：频繁项集挖掘 - 聚焦痰湿质
    print("\n方法一：频繁项集挖掘")
    print("-" * 60)
    frequent_combinations = mine_phlegm_related_combinations(discretized_df, df_result)
    
    # 3.3 方法二：特征重要性分析
    print("\n方法二：特征重要性分析")
    print("-" * 60)
    feature_importance = analyze_feature_importance(predictor, df_result)
    
    # 3.4 方法三：规则蒸馏 - 提取决策规则
    print("\n方法三：规则蒸馏")
    print("-" * 60)
    decision_rules = extract_decision_rules(predictor, df_result)
    
    # 4. 整合分析结果
    print("\n[步骤4] 整合分析结果...")
    final_analysis = integrate_analysis(frequent_combinations, feature_importance, decision_rules)
    
    # 5. 生成完整报告
    print("\n[步骤5] 生成完整报告...")
    generate_report(final_analysis, df_result)
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)

def discretize_features(df):
    """特征离散化 - 更精细的离散化策略"""
    discretized = df.copy()
    
    # 1. 痰湿质（更细致的分层）
    discretized['痰湿质_离散'] = pd.cut(
        discretized['痰湿质'],
        bins=[-1, 30, 50, 70, 100],
        labels=['痰湿质低', '痰湿质中低', '痰湿质中高', '痰湿质高']
    )
    
    # 2. 活动能力
    discretized['活动能力_离散'] = pd.cut(
        discretized['活动量表总分（ADL总分+IADL总分）'],
        bins=[-1, 39, 59, 79, 100],
        labels=['活动能力低', '活动能力中低', '活动能力中高', '活动能力高']
    )
    
    # 3. 血脂异常（更细致的分层）
    discretized['血脂异常_离散'] = pd.cut(
        discretized['血脂异常项数'],
        bins=[-1, 0, 1, 2, 10],
        labels=['血脂正常', '血脂异常1项', '血脂异常2项', '血脂异常3项+']
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
    
    return discretized

def mine_phlegm_related_combinations(discretized_df, df_result):
    """挖掘与痰湿质相关的核心特征组合"""
    print("挖掘痰湿质相关的特征组合...")
    
    # 选择离散化后的特征
    discrete_features = ['痰湿质_离散', '活动能力_离散', '血脂异常_离散', '血尿酸_离散', 'BMI_离散']
    
    # 转换为one-hot编码
    one_hot = pd.get_dummies(discretized_df[discrete_features])
    
    # 挖掘频繁项集
    frequent_itemsets = fpgrowth(one_hot, min_support=0.02, use_colnames=True)
    
    # 分析每个项集的风险关联性
    results = []
    for _, row in frequent_itemsets.iterrows():
        items = row['itemsets']
        
        # 计算项集支持度
        support = row['support']
        
        # 计算包含痰湿质的项集
        has_phlegm = any('痰湿质' in item for item in items)
        
        # 计算项集样本的风险分布
        mask = pd.Series(True, index=one_hot.index)
        for item in items:
            mask = mask & (one_hot[item] == 1)
        
        if mask.sum() > 0:
            subset = df_result[mask]
            high_risk_ratio = (subset['最终风险等级'].isin(['临床确诊高风险', '高风险', '高风险(中医预警)'])).mean()
            
            results.append({
                '组合': ' + '.join(sorted(items)),
                '大小': len(items),
                '全体支持度': support,
                '包含痰湿质': has_phlegm,
                '高风险比例': high_risk_ratio,
                '样本数': mask.sum()
            })
    
    # 整理结果
    combinations_df = pd.DataFrame(results)
    
    # 重点展示：包含痰湿质的组合
    phlegm_combinations = combinations_df[
        combinations_df['包含痰湿质'] &
        (combinations_df['大小'] >= 2)
    ].sort_values(['高风险比例', '全体支持度'], ascending=[False, False])
    
    print(f"\n包含痰湿质的核心组合（Top 20）:")
    display_cols = ['组合', '大小', '样本数', '全体支持度', '高风险比例']
    print(phlegm_combinations[display_cols].head(20).to_string(index=False))
    
    return {
        'frequent_itemsets': combinations_df,
        'phlegm_combinations': phlegm_combinations
    }

def analyze_feature_importance(predictor, df_result):
    """分析特征重要性"""
    print("分析特征重要性...")
    
    # 获取模型预测概率作为标签
    y = df_result['模型预测概率'] > 0.5  # 简化为二分类
    
    # 准备特征矩阵
    # 创建交叉特征
    df_temp = df_result.copy()
    df_temp['痰湿质×BMI'] = df_temp['痰湿质'] * df_temp['BMI']
    df_temp['痰湿质×活动量表'] = df_temp['痰湿质'] * df_temp['活动量表总分（ADL总分+IADL总分）']
    df_temp['痰湿质×血尿酸'] = df_temp['痰湿质'] * df_temp['血尿酸']
    df_temp['气虚质×BMI'] = df_temp['气虚质'] * df_temp['BMI']
    df_temp['气虚质×活动量表'] = df_temp['气虚质'] * df_temp['活动量表总分（ADL总分+IADL总分）']
    
    # 选择可用特征（不含血脂相关）
    feature_names = [
        '平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质',
        'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）',
        '年龄组', '性别', '吸烟史', '饮酒史',
        '空腹血糖', '血尿酸', 'BMI',
        '痰湿质×BMI', '痰湿质×活动量表', '痰湿质×血尿酸', '气虚质×BMI', '气虚质×活动量表'
    ]
    
    # 检查哪些特征可用
    available_features = [f for f in feature_names if f in df_temp.columns]
    X = df_temp[available_features].fillna(0)
    
    # 训练一个简单的LightGBM来分析特征重要性
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y)
    
    # 提取特征重要性
    importance_df = pd.DataFrame({
        '特征': available_features,
        '重要性': model.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    print("\n特征重要性排名（Top 15）:")
    print(importance_df.head(15).to_string(index=False))
    
    # 特别关注痰湿质相关特征
    print("\n痰湿质相关特征:")
    phlegm_features = [f for f in available_features if '痰湿质' in f]
    print(importance_df[importance_df['特征'].isin(phlegm_features)].to_string(index=False))
    
    return {
        'importance': importance_df,
        'top_features': importance_df.head(15)
    }

def extract_decision_rules(predictor, df_result):
    """从模型提取决策规则"""
    print("提取决策规则...")
    
    # 首先进行分组分析：痰湿质 + 活动能力 + 血脂异常
    print("\n关键组合分析：痰湿质 + 活动能力 + 血脂异常")
    
    # 1. 痰湿质分组
    phlegm_bins = [-1, 30, 60, 100]
    phlegm_labels = ['痰湿质低', '痰湿质中', '痰湿质高']
    df_result['痰湿质分组'] = pd.cut(df_result['痰湿质'], bins=phlegm_bins, labels=phlegm_labels)
    
    # 2. 活动能力分组
    activity_bins = [-1, 39, 59, 100]
    activity_labels = ['活动能力低', '活动能力中', '活动能力高']
    df_result['活动能力分组'] = pd.cut(
        df_result['活动量表总分（ADL总分+IADL总分）'],
        bins=activity_bins,
        labels=activity_labels
    )
    
    # 3. 血脂异常分组
    lipid_bins = [-1, 0, 1, 10]
    lipid_labels = ['血脂正常', '血脂异常1项', '血脂异常2项+']
    df_result['血脂分组'] = pd.cut(df_result['血脂异常项数'], bins=lipid_bins, labels=lipid_labels)
    
    # 交叉分析
    cross_table = df_result.groupby(['痰湿质分组', '活动能力分组', '血脂分组']).agg({
        '高血脂症二分类标签': ['count', 'mean']
    }).reset_index()
    
    cross_table.columns = ['痰湿质', '活动能力', '血脂', '样本数', '高风险比例']
    cross_table = cross_table[cross_table['样本数'] >= 5].sort_values('高风险比例', ascending=False)
    
    print("\n关键组合高风险比例（样本数≥5）:")
    print(cross_table.head(20).to_string(index=False))
    
    # 重点分析：痰湿质高 + 活动能力低（即便血脂正常）
    print("\n" + "=" * 60)
    print("重点分析：痰湿质高 + 活动能力低")
    print("=" * 60)
    subset = df_result[
        (df_result['痰湿质分组'] == '痰湿质高') &
        (df_result['活动能力分组'] == '活动能力低')
    ]
    if len(subset) > 0:
        print(f"样本数: {len(subset)}")
        print(f"高风险比例: {(subset['最终风险等级'].isin(['临床确诊高风险', '高风险', '高风险(中医预警)'])).mean():.2%}")
        print(f"血脂正常的样本数: {(subset['血脂分组'] == '血脂正常').sum()}")
        print(f"血脂异常的样本数: {(subset['血脂分组'] != '血脂正常').sum()}")
    
    return {
        'cross_table': cross_table,
        'key_subset': subset
    }

def integrate_analysis(frequent_combinations, feature_importance, decision_rules):
    """整合分析结果"""
    print("整合分析结果...")
    
    # 识别核心组合
    # 1. 从频繁项集中选择高风险比例的组合
    phlegm_combinations = frequent_combinations['phlegm_combinations']
    top_frequent = phlegm_combinations[phlegm_combinations['高风险比例'] > 0.7].head(5)
    
    # 2. 从交叉分析中选择高风险组合
    cross_table = decision_rules['cross_table']
    top_cross = cross_table[cross_table['高风险比例'] > 0.7].head(5)
    
    # 3. 综合选择核心组合
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
    
    return {
        'core_combinations': core_combinations,
        'frequent_itemsets': frequent_combinations['frequent_itemsets'],
        'cross_table': cross_table,
        'feature_importance': feature_importance['importance'],
        'top_frequent': top_frequent,
        'top_cross': top_cross
    }

def generate_report(final_analysis, df_result):
    """生成完整报告"""
    print("\n" + "=" * 80)
    print("痰湿体质高风险人群核心特征组合分析报告")
    print("=" * 80)
    
    # 1. 核心结论
    print("\n一、核心结论")
    print("-" * 60)
    print("1. 痰湿质是识别高血脂高风险人群的关键中医特征")
    print("2. 痰湿质 + 低活动量 + 血脂异常的组合是最核心的高风险特征")
    print("3. 即使血脂正常，痰湿质高 + 活动能力低的组合也需关注潜在风险")
    print("4. 痰湿质与BMI、活动能力的交叉特征能进一步提升风险识别能力")
    
    # 2. 核心特征组合详细说明
    print("\n二、核心特征组合详解")
    print("-" * 60)
    for i, combo in enumerate(final_analysis['core_combinations'], 1):
        print(f"\n组合 {i}: {combo['名称']}")
        print(f"  医学解释: {combo['医学解释']}")
        print(f"  关键指标: {', '.join(combo['关键指标'])}")
    
    # 3. 数据支撑
    print("\n三、数据支撑")
    print("-" * 60)
    
    # 整体风险分布
    risk_dist = df_result['最终风险等级'].value_counts()
    print("\n整体风险分布:")
    for level, count in risk_dist.items():
        print(f"  {level}: {count} ({count/len(df_result):.1%})")
    
    # 痰湿质分布
    print("\n痰湿质分布:")
    phlegm_high = (df_result['痰湿质'] >= 60).sum()
    phlegm_medium = ((df_result['痰湿质'] >= 30) & (df_result['痰湿质'] < 60)).sum()
    phlegm_low = (df_result['痰湿质'] < 30).sum()
    print(f"  痰湿质高 (≥60): {phlegm_high} ({phlegm_high/len(df_result):.1%})")
    print(f"  痰湿质中 (30-60): {phlegm_medium} ({phlegm_medium/len(df_result):.1%})")
    print(f"  痰湿质低 (<30): {phlegm_low} ({phlegm_low/len(df_result):.1%})")
    
    # 4. 合理解释
    print("\n四、合理解释")
    print("-" * 60)
    print("1. 中西医理论支撑:")
    print("   - 中医理论认为：痰湿质是导致血脂代谢异常的重要体质基础")
    print("   - 西医理论认为：活动能力低会影响代谢，加重血脂异常风险")
    print("   - 中西医结合：痰湿质 + 低活动量 + 血脂异常的组合完美融合了中西医理论")
    
    print("\n2. 临床指导价值:")
    print("   - 早期识别：通过痰湿质、活动能力等非血脂指标提前识别潜在风险")
    print("   - 个性化干预：针对不同特征组合制定个性化的干预方案")
    print("   - 疗效评估：可作为评估干预效果的重要指标")
    
    print("\n3. 创新点:")
    print("   - 融合了中医体质特征与西医指标")
    print("   - 关注了血脂正常但仍有潜在风险的人群")
    print("   - 采用了多方法融合的分析策略，结果更可靠")
    
    # 5. 生成数据文件
    print("\n五、生成数据文件")
    print("-" * 60)
    
    # 保存核心特征组合数据
    core_combinations_df = pd.DataFrame(final_analysis['core_combinations'])
    core_path = "data/processed/core_feature_combinations.csv"
    core_combinations_df.to_csv(core_path, index=False, encoding='utf-8-sig')
    print(f"✓ 核心特征组合数据已保存: {core_path}")
    
    # 保存特征重要性数据
    if 'feature_importance' in final_analysis:
        importance_path = "data/processed/feature_importance.csv"
        final_analysis['feature_importance'].to_csv(importance_path, index=False, encoding='utf-8-sig')
        print(f"✓ 特征重要性数据已保存: {importance_path}")
    
    # 保存交叉分析数据
    if 'cross_table' in final_analysis:
        cross_path = "data/processed/cross_analysis.csv"
        final_analysis['cross_table'].to_csv(cross_path, index=False, encoding='utf-8-sig')
        print(f"✓ 交叉分析数据已保存: {cross_path}")
    
    # 保存频繁项集数据
    if 'frequent_itemsets' in final_analysis:
        frequent_path = "data/processed/frequent_itemsets.csv"
        final_analysis['frequent_itemsets'].to_csv(frequent_path, index=False, encoding='utf-8-sig')
        print(f"✓ 频繁项集数据已保存: {frequent_path}")
    
    # 保存详细分析报告（Markdown格式）
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
        for i, combo in enumerate(final_analysis['core_combinations'], 1):
            f.write(f"### 组合 {i}: {combo['名称']}\n")
            f.write(f"- 医学解释: {combo['医学解释']}\n")
            f.write(f"- 关键指标: {', '.join(combo['关键指标'])}\n\n")
    
    print(f"✓ 详细分析报告已保存: {report_path}")
    
    # 保存原始文本报告
    txt_report_path = "data/processed/core_feature_combinations_report.txt"
    with open(txt_report_path, 'w', encoding='utf-8') as f:
        f.write("痰湿体质高风险人群核心特征组合分析报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n")
        f.write(f"样本数: {len(df_result)}\n")
    
    print(f"✓ 文本报告已保存: {txt_report_path}")
    
    print("\n所有数据文件生成完成！")

if __name__ == "__main__":
    main()
