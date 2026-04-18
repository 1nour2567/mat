# 高风险核心特征组合挖掘
import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
import os
from mlxtend.frequent_patterns import fpgrowth, association_rules

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.three_layer_architecture import TripleLayerPredictor, MODEL_FEATURES

def main():
    print("=" * 80)
    print("高风险核心特征组合挖掘")
    print("=" * 80)
    
    # 1. 加载数据和模型
    print("\n[步骤1] 加载数据和训练模型...")
    data_path = "data/processed/preprocessed_data.pkl"
    df = pd.read_pickle(data_path)
    print(f"数据加载完成，形状: {df.shape}")
    
    # 初始化并训练三层预测器
    predictor = TripleLayerPredictor()
    predictor.fit(df)
    
    # 2. 预测风险等级
    print("\n[步骤2] 预测风险等级...")
    df_result = predictor.predict(df)
    
    # 3. 提取高风险样本
    print("\n[步骤3] 提取高风险样本...")
    high_risk_samples = extract_high_risk_samples(df_result)
    print(f"高风险样本数量: {len(high_risk_samples)}")
    
    # 4. 特征离散化
    print("\n[步骤4] 特征离散化...")
    discretized_df = discretize_features(high_risk_samples)
    
    # 5. 生成项集
    print("\n[步骤5] 生成项集...")
    itemset_df = generate_itemsets(discretized_df)
    
    # 6. 频繁项集挖掘
    print("\n[步骤6] 频繁项集挖掘...")
    frequent_itemsets = mine_frequent_itemsets(itemset_df)
    
    # 7. 筛选核心特征组合
    print("\n[步骤7] 筛选核心特征组合...")
    core_combinations = filter_core_combinations(frequent_itemsets, itemset_df)
    
    # 8. 输出结果
    print("\n[步骤8] 输出结果...")
    output_results(core_combinations)
    
    print("\n" + "=" * 80)
    print("高风险核心特征组合挖掘完成！")
    print("=" * 80)

def extract_high_risk_samples(df_result):
    """
    提取高风险样本
    """
    # 提取高风险样本（包括临床确诊高风险和模型判定的高风险）
    high_risk_conditions = [
        (df_result['最终风险等级'] == '临床确诊高风险'),
        (df_result['最终风险等级'] == '高风险'),
        (df_result['最终风险等级'] == '高风险(中医预警)')
    ]
    
    high_risk_mask = np.any(high_risk_conditions, axis=0)
    high_risk_samples = df_result[high_risk_mask].copy()
    
    return high_risk_samples

def discretize_features(df):
    """
    特征离散化（三域离散化）
    """
    discretized = df.copy()
    
    # 1. 中医体质域离散化
    # 痰湿质：按赛题附表离散为低/中/高
    discretized['痰湿质_离散'] = pd.cut(
        discretized['痰湿质'],
        bins=[-1, 40, 60, 100],
        labels=['痰湿_低', '痰湿_中', '痰湿_高']
    )
    
    # 气虚质离散化
    discretized['气虚质_离散'] = pd.cut(
        discretized['气虚质'],
        bins=[-1, 40, 60, 100],
        labels=['气虚_低', '气虚_中', '气虚_高']
    )
    
    # 2. 活动能力域离散化
    discretized['活动能力_离散'] = pd.cut(
        discretized['活动量表总分（ADL总分+IADL总分）'],
        bins=[-1, 40, 60, 100],
        labels=['活动_低', '活动_中', '活动_高']
    )
    
    # 3. 血脂代谢域离散化
    # 血脂异常项数
    discretized['血脂异常_离散'] = pd.cut(
        discretized['血脂异常项数'],
        bins=[-1, 0, 1, 10],
        labels=['血脂异常_0', '血脂异常_1', '血脂异常_≥2']
    )
    
    # 血尿酸
    discretized['血尿酸_离散'] = np.where(
        discretized['血尿酸'] > 420, '尿酸_高', '尿酸_正常'
    )
    
    # BMI
    discretized['BMI_离散'] = pd.cut(
        discretized['BMI'],
        bins=[-1, 18.5, 24, 28, 100],
        labels=['BMI_偏瘦', 'BMI_正常', 'BMI_超重', 'BMI_肥胖']
    )
    
    return discretized

def generate_itemsets(discretized_df):
    """
    生成项集
    """
    # 选择离散化的特征
    discrete_features = [
        '痰湿质_离散', '气虚质_离散', '活动能力_离散', 
        '血脂异常_离散', '血尿酸_离散', 'BMI_离散'
    ]
    
    # 创建项集
    itemsets = []
    for _, row in discretized_df.iterrows():
        itemset = []
        for feature in discrete_features:
            value = row[feature]
            if pd.notna(value):
                itemset.append(str(value))
        itemsets.append(itemset)
    
    # 转换为 one-hot 编码格式
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit_transform(itemsets)
    itemset_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    return itemset_df

def mine_frequent_itemsets(itemset_df, min_support=0.03):
    """
    使用 FP-Growth 挖掘频繁项集
    """
    frequent_itemsets = fpgrowth(itemset_df, min_support=min_support, use_colnames=True)
    return frequent_itemsets

def filter_core_combinations(frequent_itemsets, itemset_df):
    """
    筛选核心特征组合
    """
    # 计算关联规则
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    
    # 筛选提升度 ≥ 1.5 的规则
    core_rules = rules[rules['lift'] >= 1.5].copy()
    
    # 限制组合长度 ≤ 4
    core_rules = core_rules[core_rules['antecedents'].apply(len) <= 4]
    
    # 按提升度降序排序
    core_rules = core_rules.sort_values('lift', ascending=False)
    
    return core_rules

def output_results(core_combinations):
    """
    输出结果
    """
    print("\n=== 高风险核心特征组合 ===")
    print("按提升度降序排序")
    print("-" * 80)
    
    if core_combinations.empty:
        print("未找到符合条件的核心特征组合")
        return
    
    # 输出前10个核心组合
    for i, (_, row) in enumerate(core_combinations.head(10).iterrows()):
        antecedents = ', '.join(list(row['antecedents']))
        support = row['support'] * 100
        confidence = row['confidence'] * 100
        lift = row['lift']
        
        print(f"{i+1}. 特征组合: {antecedents}")
        print(f"   支持度: {support:.2f}% | 置信度: {confidence:.2f}% | 提升度: {lift:.2f}")
        print("-" * 80)
    
    # 保存结果
    output_path = "data/processed/high_risk_core_combinations.csv"
    core_combinations.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_path}")

if __name__ == "__main__":
    main()