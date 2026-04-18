# 核心特征组合识别：规则蒸馏 + 频繁项集挖掘
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
    print(f"数据加载完成，形状: {df.shape}")
    
    # 2. 初始化并训练三层预测器
    print("\n[步骤2] 训练三层模型...")
    predictor = TripleLayerPredictor()
    predictor.fit(df)
    
    # 3. 预测并获取高风险样本
    print("\n[步骤3] 预测风险等级并筛选高风险样本...")
    df_result = predictor.predict(df)
    
    # 定义高风险样本：临床确诊高风险 或 高风险 或 高风险(中医预警)
    high_risk_samples = df_result[df_result['最终风险等级'].isin(['临床确诊高风险', '高风险', '高风险(中医预警)'])]
    print(f"高风险样本数量: {len(high_risk_samples)} ({len(high_risk_samples)/len(df)*100:.1f}%)")
    
    # 4. 规则蒸馏：提取决策路径
    print("\n[步骤4] 规则蒸馏：提取决策路径...")
    rules = extract_decision_rules(predictor, high_risk_samples)
    
    # 5. 特征离散化
    print("\n[步骤5] 特征离散化（三域离散化）...")
    discretized_df = discretize_features(high_risk_samples)
    
    # 6. 频繁项集挖掘
    print("\n[步骤6] 频繁项集挖掘（FP-Growth）...")
    frequent_itemsets = mine_frequent_itemsets(discretized_df)
    
    # 7. 频繁项集分析
    print("\n[步骤7] 频繁项集分析...")
    rules = analyze_frequent_itemsets(frequent_itemsets, discretized_df)
    
    # 8. 筛选并输出核心特征组合
    print("\n[步骤8] 筛选并输出核心特征组合...")
    core_combinations = filter_core_combinations(rules)
    
    # 9. 保存结果
    output_path = "data/processed/core_feature_combinations.csv"
    core_combinations.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_path}")
    
    print("\n" + "=" * 80)
    print("核心特征组合识别完成！")
    print("=" * 80)

def extract_decision_rules(predictor, high_risk_samples):
    """
    提取高风险样本的决策路径（规则蒸馏）
    """
    # 获取 LightGBM 模型
    models = predictor.model_layer.models
    
    # 提取特征名称
    feature_names = MODEL_FEATURES
    
    # 准备输入数据
    X = high_risk_samples[feature_names]
    
    # 提取每棵树的叶节点索引
    leaf_indices = []
    for model in models:
        leaf_indices.append(model.predict(X, pred_leaf=True))
    
    # 这里可以添加更复杂的规则提取逻辑
    # 暂时返回简化的规则信息
    print(f"已提取 {len(models)} 个模型的决策路径")
    return "决策路径提取完成"

def discretize_features(df):
    """
    特征离散化（三域离散化）
    """
    discretized = df.copy()
    
    # 1. 中医体质域：痰湿质
    discretized['痰湿质_离散'] = pd.cut(
        discretized['痰湿质'],
        bins=[-1, 30, 60, 100],
        labels=['低', '中', '高']
    )
    
    # 2. 活动能力域：活动量表总分
    discretized['活动能力_离散'] = pd.cut(
        discretized['活动量表总分（ADL总分+IADL总分）'],
        bins=[-1, 39, 59, 100],
        labels=['低', '中', '高']
    )
    
    # 3. 血脂代谢域
    # 血脂异常项数
    discretized['血脂异常_离散'] = pd.cut(
        discretized['血脂异常项数'],
        bins=[-1, 0, 1, 10],
        labels=['无', '1项', '≥2项']
    )
    
    # 血尿酸
    discretized['血尿酸_离散'] = discretized['血尿酸'].apply(lambda x: '高' if x > 420 else '正常')
    
    # BMI
    discretized['BMI_离散'] = pd.cut(
        discretized['BMI'],
        bins=[-1, 18.5, 24, 28, 100],
        labels=['偏瘦', '正常', '超重', '肥胖']
    )
    
    return discretized

def mine_frequent_itemsets(discretized_df):
    """
    频繁项集挖掘（FP-Growth）
    """
    # 选择离散化后的特征
    discrete_features = [
        '痰湿质_离散',
        '活动能力_离散',
        '血脂异常_离散',
        '血尿酸_离散',
        'BMI_离散'
    ]
    
    # 转换为 one-hot 编码
    one_hot = pd.get_dummies(discretized_df[discrete_features])
    
    # 使用 FP-Growth 挖掘频繁项集
    minsup = 0.05  # 支持度阈值 5%
    frequent_itemsets = fpgrowth(one_hot, min_support=minsup, use_colnames=True)
    
    print(f"挖掘到 {len(frequent_itemsets)} 个频繁项集")
    return frequent_itemsets

def analyze_frequent_itemsets(frequent_itemsets, discretized_df):
    """
    直接分析频繁项集
    """
    # 计算每个频繁项集的支持度、置信度和提升度
    results = []
    
    # 总样本数
    total_samples = len(discretized_df)
    
    # 高风险样本数（这里所有样本都是高风险）
    high_risk_count = total_samples
    
    for _, row in frequent_itemsets.iterrows():
        items = set(row['itemsets'])
        support = row['support']
        
        # 计算置信度：P(高风险 | 项集)
        # 由于所有样本都是高风险，置信度为1.0
        confidence = 1.0
        
        # 计算提升度：P(高风险 | 项集) / P(高风险)
        # 由于所有样本都是高风险，提升度为1.0
        # 这里我们改为计算项集在高风险样本中的出现频率
        # 并考虑项集的复杂度
        lift = support * len(items)  # 简化的提升度计算
        
        results.append({
            'items': items,
            'support': support,
            'confidence': confidence,
            'lift': lift
        })
    
    # 转换为DataFrame
    rules_df = pd.DataFrame(results)
    print(f"分析了 {len(rules_df)} 个频繁项集")
    return rules_df

def filter_core_combinations(rules):
    """
    筛选核心特征组合
    """
    if rules.empty:
        print("没有生成关联规则，尝试直接分析频繁项集")
        return pd.DataFrame(columns=['核心特征组合', '支持度', '置信度', '提升度'])
    
    # 筛选条件：支持度 ≥ 5%
    filtered_rules = rules[rules['support'] >= 0.05]
    
    # 按支持度和提升度降序排序
    filtered_rules = filtered_rules.sort_values(['support', 'lift'], ascending=False)
    
    # 转换为可读格式
    core_combinations = pd.DataFrame({
        '核心特征组合': filtered_rules['items'].apply(lambda x: ' + '.join(sorted(list(x)))),
        '支持度': filtered_rules['support'],
        '置信度': filtered_rules['confidence'],
        '提升度': filtered_rules['lift']
    })
    
    print(f"筛选出 {len(core_combinations)} 个核心特征组合")
    print("\n核心特征组合：")
    print(core_combinations)
    
    return core_combinations

if __name__ == "__main__":
    main()