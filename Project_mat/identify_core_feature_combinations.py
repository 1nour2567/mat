# 核心特征组合识别：规则蒸馏 + 频繁项集挖掘
# 修正版：解决置信度、提升度计算问题，突出核心组合
import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
import os
from mlxtend.frequent_patterns import fpgrowth, association_rules

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.three_layer_architecture import TripleLayerPredictor, MODEL_FEATURES

def main():
    print("=" * 80)
    print("核心特征组合识别：规则蒸馏 + 频繁项集挖掘")
    print("=" * 80)
    
    # 1. 加载预处理后的数据
    print("\n[步骤1] 加载数据...")
    data_path = "data/processed/preprocessed_data.pkl"
    df = pd.read_pickle(data_path)
    print(f"数据加载完成，形状：{df.shape}")
    
    # 2. 初始化并训练三层预测器
    print("\n[步骤2] 训练三层模型...")
    predictor = TripleLayerPredictor()
    predictor.fit(df)
    
    # 3. 预测并获取风险等级
    print("\n[步骤3] 预测风险等级...")
    df_result = predictor.predict(df)
    
    # 4. 规则蒸馏：提取决策路径
    print("\n[步骤4] 规则蒸馏：提取决策路径...")
    extract_decision_rules(predictor, df_result)
    
    # 5. 特征离散化（在全体样本上进行）
    print("\n[步骤5] 特征离散化（三域离散化）...")
    discretized_df = discretize_features(df_result)
    
    # 6. 频繁项集挖掘（在全体样本上进行）
    print("\n[步骤6] 频繁项集挖掘（FP-Growth）...")
    frequent_itemsets = mine_frequent_itemsets(discretized_df)
    
    # 7. 频繁项集分析（在全体样本上计算置信度和提升度）
    print("\n[步骤7] 频繁项集分析...")
    rules = analyze_frequent_itemsets(frequent_itemsets, discretized_df, df_result)
    
    # 8. 筛选并输出核心特征组合
    print("\n[步骤8] 筛选并输出核心特征组合...")
    core_combinations = filter_core_combinations(rules)
    
    # 9. 保存结果
    output_path = "data/processed/core_feature_combinations.csv"
    core_combinations.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到：{output_path}")
    
    print("\n" + "=" * 80)
    print("核心特征组合识别完成！")
    print("=" * 80)

def extract_decision_rules(predictor, df_result):
    """提取高风险样本的决策路径"""
    # 获取 LightGBM 模型
    models = predictor.model_layer.models
    
    # 提取特征名称
    feature_names = MODEL_FEATURES
    
    # 筛选高风险样本
    high_risk_samples = df_result[df_result['最终风险等级'].isin(['临床确诊高风险', '高风险', '高风险(中医预警)'])]
    
    # 准备输入数据
    X = high_risk_samples[feature_names]
    
    # 提取每棵树的叶节点索引
    leaf_indices = []
    for model in models:
        leaf_indices.append(model.predict(X, pred_leaf=True))
    
    print(f"已提取 {len(models)} 个模型的决策路径，覆盖 {len(high_risk_samples)} 个高风险样本")

def discretize_features(df):
    """特征离散化（三域离散化）"""
    discretized = df.copy()
    
    # 1. 中医体质域：痰湿质、气虚质等
    discretized['痰湿质_离散'] = pd.cut(
        discretized['痰湿质'],
        bins=[-1, 30, 60, 100],
        labels=['痰湿质低', '痰湿质中', '痰湿质高']
    )
    
    if '气虚质' in discretized.columns:
        discretized['气虚质_离散'] = pd.cut(
            discretized['气虚质'],
            bins=[-1, 30, 60, 100],
            labels=['气虚质低', '气虚质中', '气虚质高']
        )
    
    # 2. 活动能力域：活动量表总分
    discretized['活动能力_离散'] = pd.cut(
        discretized['活动量表总分（ADL总分+IADL总分）'],
        bins=[-1, 39, 59, 100],
        labels=['活动能力低', '活动能力中', '活动能力高']
    )
    
    # 3. 血脂代谢域
    # 血脂异常项数
    discretized['血脂异常_离散'] = pd.cut(
        discretized['血脂异常项数'],
        bins=[-1, 0, 1, 10],
        labels=['血脂正常', '血脂异常1项', '血脂异常≥2项']
    )
    
    # 血尿酸
    discretized['血尿酸_离散'] = discretized['血尿酸'].apply(lambda x: '血尿酸高' if x > 420 else '血尿酸正常')
    
    # BMI
    discretized['BMI_离散'] = pd.cut(
        discretized['BMI'],
        bins=[-1, 18.5, 24, 28, 100],
        labels=['BMI偏瘦', 'BMI正常', 'BMI超重', 'BMI肥胖']
    )
    
    return discretized

def mine_frequent_itemsets(discretized_df):
    """频繁项集挖掘（FP-Growth）"""
    # 选择离散化后的特征
    discrete_features = [col for col in discretized_df.columns if col.endswith('_离散')]
    
    # 转换为 one-hot 编码
    one_hot = pd.get_dummies(discretized_df[discrete_features])
    
    # 使用 FP-Growth 挖掘频繁项集
    minsup = 0.02  # 降低支持度阈值，以便捕获更多组合
    frequent_itemsets = fpgrowth(one_hot, min_support=minsup, use_colnames=True)
    
    print(f"挖掘到 {len(frequent_itemsets)} 个频繁项集")
    return frequent_itemsets

def analyze_frequent_itemsets(frequent_itemsets, discretized_df, df_result):
    """直接分析频繁项集，在全体样本上计算置信度和提升度"""
    # 创建目标向量：1 表示高风险，0 表示其他
    y_high = df_result['最终风险等级'].isin(['临床确诊高风险', '高风险', '高风险(中医预警)']).astype(int)
    base_rate = y_high.mean()  # 整体高风险比例
    
    print(f"整体高风险比例：{base_rate:.3f}")
    
    # 选择离散化后的特征
    discrete_features = [col for col in discretized_df.columns if col.endswith('_离散')]
    one_hot = pd.get_dummies(discretized_df[discrete_features])
    
    # 计算每个频繁项集的统计指标
    results = []
    
    for _, row in frequent_itemsets.iterrows():
        items = set(row['itemsets'])
        
        # 计算项集在全体样本中的出现情况
        mask = pd.Series(True, index=one_hot.index)
        for item in items:
            mask = mask & (one_hot[item] == 1)
        
        # 在全体样本中的支持度
        support_all = mask.mean()
        
        # 在高风险样本中的支持度（即 P(组合 ∧ 高风险)）
        support_high = (mask & (y_high == 1)).mean()
        
        # 计算置信度和提升度
        if support_all > 0:
            confidence = support_high / support_all
        else:
            confidence = 0
        
        lift = confidence / base_rate if base_rate > 0 else 0
        
        # 计算项集大小
        item_size = len(items)
        
        results.append({
            'items': items,
            'item_size': item_size,
            'support_all': support_all,
            'support_high': support_high,
            'confidence': confidence,
            'lift': lift
        })
    
    # 转换为 DataFrame
    rules_df = pd.DataFrame(results)
    print(f"分析了 {len(rules_df)} 个频繁项集")
    return rules_df

def filter_core_combinations(rules):
    """筛选核心特征组合，特别关注血脂正常情况下的组合"""
    if rules.empty:
        print("没有生成规则")
        return pd.DataFrame(columns=['核心特征组合', '项集大小', '全体样本支持度', '高风险样本支持度', '置信度', '提升度'])
    
    # 筛选条件：置信度≥0.7，提升度≥1.1，项集大小≥2
    filtered_rules = rules[
        (rules['confidence'] >= 0.7) &
        (rules['lift'] >= 1.1) &
        (rules['item_size'] >= 2)
    ]
    
    # 优先突出赛题核心组合：包含痰湿质+活动能力+血脂代谢的组合
    # 特别关注血脂正常情况下的组合
    def has_core_pattern(items):
        item_str = str(items).lower()
        # 检查是否包含核心组合元素
        has_phlegm = '痰湿质高' in item_str or '痰湿质中' in item_str
        has_low_activity = '活动能力低' in item_str
        has_lipid = '血脂异常' in item_str
        has_normal_lipid = '血脂正常' in item_str
        has_high_bmi = '肥胖' in item_str or '超重' in item_str
        has_high_uric = '血尿酸高' in item_str
        has_phlegm_high = '痰湿质高' in item_str
        
        # 核心组合评分
        score = 0
        if has_phlegm_high:
            score += 4  # 痰湿质高是核心要素
        elif has_phlegm:
            score += 2
        if has_low_activity:
            score += 3
        if has_lipid:
            score += 2
        if has_normal_lipid:
            score += 3  # 血脂正常但仍然高风险的组合更有价值
        if has_high_bmi:
            score += 1
        if has_high_uric:
            score += 1
        
        # 情况1：血脂正常但具有痰湿质+活动能力低，这是核心组合
        if has_normal_lipid and has_phlegm_high and has_low_activity:
            return True
        # 情况2：痰湿质高 + 2个其他核心要素
        if has_phlegm_high and (has_low_activity + has_high_bmi + has_high_uric >= 2):
            return True
        # 情况3：包含3个及以上核心要素
        if has_phlegm + has_low_activity + has_lipid + has_high_bmi + has_high_uric >= 3:
            return True
        # 情况4：具有较高评分
        if score >= 5:
            return True
        return False
    
    filtered_rules['is_core'] = filtered_rules['items'].apply(has_core_pattern)
    
    # 添加特殊标记：血脂正常但仍然高风险的组合
    def has_normal_lipid(items):
        return '血脂正常' in str(items)
    
    filtered_rules['has_normal_lipid'] = filtered_rules['items'].apply(has_normal_lipid)
    
    # 排序：先按是否核心组合，再按是否血脂正常，再按提升度，再按置信度，最后按全体样本支持度
    filtered_rules = filtered_rules.sort_values(
        ['is_core', 'has_normal_lipid', 'lift', 'confidence', 'support_all'],
        ascending=[False, False, False, False, False]
    )
    
    # 转换为可读格式
    core_combinations = pd.DataFrame({
        '核心特征组合': filtered_rules['items'].apply(lambda x: ' + '.join(sorted(list(x)))),
        '项集大小': filtered_rules['item_size'],
        '全体样本支持度': filtered_rules['support_all'],
        '高风险样本支持度': filtered_rules['support_high'],
        '置信度': filtered_rules['confidence'],
        '提升度': filtered_rules['lift'],
        '是否核心组合': filtered_rules['is_core'],
        '是否血脂正常': filtered_rules['has_normal_lipid']
    })
    
    print(f"筛选出 {len(core_combinations)} 个核心特征组合")
    print(f"其中赛题核心组合：{len(core_combinations[core_combinations['是否核心组合']])} 个")
    print(f"其中血脂正常但高风险的组合：{len(core_combinations[core_combinations['是否血脂正常']])} 个")
    print("\nTop 20 核心特征组合：")
    print(core_combinations.head(20).to_string())
    
    return core_combinations

if __name__ == "__main__":
    main()
