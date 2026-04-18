
"""分析血脂正常但有潜在风险的样本"""
import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.three_layer_architecture import TripleLayerPredictor, ClinicalRuleLayer, MODEL_FEATURES

def main():
    print("=" * 80)
    print("血脂正常但有潜在风险的样本分析")
    print("=" * 80)
    
    # 1. 加载预处理后的数据
    data_path = "data/processed/preprocessed_data.pkl"
    df = pd.read_pickle(data_path)
    
    # 2. 初始化并训练三层预测器
    predictor = TripleLayerPredictor()
    predictor.fit(df)
    
    # 3. 预测风险等级
    df_result = predictor.predict(df)
    
    # 4. 筛选血脂正常的样本
    # 首先计算血脂异常项数
    df_result = ClinicalRuleLayer.apply_clinical_rules(df_result)[0]
    df_normal_lipid = df_result[df_result['血脂异常项数'] == 0].copy()
    
    print(f"血脂正常的样本数量: {len(df_normal_lipid)}")
    
    # 5. 分析这些样本的模型预测概率
    # 注意：由于血脂正常，这些样本不会通过临床规则被标记为高风险，但可能有模型预测概率
    # 让我们手动计算模型预测概率
    print("\n" + "=" * 80)
    print("血脂正常样本的模型预测概率分析")
    print("=" * 80)
    
    # 获取模型预测概率
    # 我们需要手动调用模型层的预测方法
    # 首先准备输入数据
    feature_names = MODEL_FEATURES
    X_normal_lipid = df_normal_lipid[feature_names]
    
    # 获取预测概率
    model_preds = []
    for model in predictor.model_layer.models:
        model_preds.append(model.predict_proba(X_normal_lipid)[:, 1])
    
    avg_pred = np.mean(model_preds, axis=0)
    df_normal_lipid['模型预测概率'] = avg_pred
    
    # 分析预测概率分布
    print(f"\n模型预测概率的描述性统计:")
    print(df_normal_lipid['模型预测概率'].describe())
    
    # 6. 分析特征分布
    print("\n" + "=" * 80)
    print("血脂正常样本的关键特征分析")
    print("=" * 80)
    
    # 分析痰湿质分布
    print("\n痰湿质分布:")
    print(df_normal_lipid['痰湿质'].describe())
    
    # 分析活动能力
    print("\n活动量表总分分布:")
    print(df_normal_lipid['活动量表总分（ADL总分+IADL总分）'].describe())
    
    # 分析BMI分布
    print("\nBMI分布:")
    print(df_normal_lipid['BMI'].describe())
    
    # 分析血尿酸分布
    print("\n血尿酸分布:")
    print(df_normal_lipid['血尿酸'].describe())
    
    # 7. 找出模型预测概率较高的样本
    print("\n" + "=" * 80)
    print("血脂正常但模型预测概率较高的样本分析")
    print("=" * 80)
    
    high_prob_samples = df_normal_lipid[df_normal_lipid['模型预测概率'] >= 0.25].sort_values('模型预测概率', ascending=False)
    print(f"\n模型预测概率>=0.25的样本数量: {len(high_prob_samples)}")
    
    if len(high_prob_samples) > 0:
        print("\nTop 10高概率样本:")
        print(high_prob_samples[['痰湿质', '活动量表总分（ADL总分+IADL总分）', 'BMI', '血尿酸', '模型预测概率']].head(10))
    
    # 8. 频繁特征组合分析
    print("\n" + "=" * 80)
    print("血脂正常样本中常见的特征组合分析")
    print("=" * 80)
    
    # 离散化特征
    df_normal_lipid['痰湿质_离散'] = pd.cut(df_normal_lipid['痰湿质'], bins=[-1, 30, 60, 100], labels=['低', '中', '高'])
    df_normal_lipid['活动能力_离散'] = pd.cut(df_normal_lipid['活动量表总分（ADL总分+IADL总分）'], bins=[-1, 39, 59, 100], labels=['低', '中', '高'])
    df_normal_lipid['BMI_离散'] = pd.cut(df_normal_lipid['BMI'], bins=[-1, 18.5, 24, 28, 100], labels=['偏瘦', '正常', '超重', '肥胖'])
    df_normal_lipid['血尿酸_离散'] = df_normal_lipid['血尿酸'].apply(lambda x: '高' if x > 420 else '正常')
    
    # 分析常见的组合
    combinations = []
    for _, row in df_normal_lipid.iterrows():
        combo = []
        if row['痰湿质_离散'] in ['中', '高']:
            combo.append(f"痰湿质{row['痰湿质_离散']}")
        if row['活动能力_离散'] == '低':
            combo.append("活动能力低")
        if row['BMI_离散'] in ['超重', '肥胖']:
            combo.append(f"BMI{row['BMI_离散']}")
        if row['血尿酸_离散'] == '高':
            combo.append("血尿酸高")
        combinations.append(", ".join(combo) if combo else "无高风险特征")
    
    df_normal_lipid['风险特征组合'] = combinations
    
    print("\n风险特征组合分布:")
    combo_counts = df_normal_lipid['风险特征组合'].value_counts()
    print(combo_counts)
    
    # 9. 针对高预测概率样本的分析
    print("\n" + "=" * 80)
    print("潜在高风险组合分析")
    print("=" * 80)
    
    if len(high_prob_samples) > 0:
        high_prob_samples['痰湿质_离散'] = pd.cut(high_prob_samples['痰湿质'], bins=[-1, 30, 60, 100], labels=['低', '中', '高'])
        high_prob_samples['活动能力_离散'] = pd.cut(high_prob_samples['活动量表总分（ADL总分+IADL总分）'], bins=[-1, 39, 59, 100], labels=['低', '中', '高'])
        high_prob_samples['BMI_离散'] = pd.cut(high_prob_samples['BMI'], bins=[-1, 18.5, 24, 28, 100], labels=['偏瘦', '正常', '超重', '肥胖'])
        high_prob_samples['血尿酸_离散'] = high_prob_samples['血尿酸'].apply(lambda x: '高' if x > 420 else '正常')
        
        combinations = []
        for _, row in high_prob_samples.iterrows():
            combo = []
            if row['痰湿质_离散'] in ['中', '高']:
                combo.append(f"痰湿质{row['痰湿质_离散']}")
            if row['活动能力_离散'] == '低':
                combo.append("活动能力低")
            if row['BMI_离散'] in ['超重', '肥胖']:
                combo.append(f"BMI{row['BMI_离散']}")
            if row['血尿酸_离散'] == '高':
                combo.append("血尿酸高")
            combinations.append(", ".join(combo) if combo else "无高风险特征")
        
        high_prob_samples['风险特征组合'] = combinations
        
        print("\n高预测概率样本的风险特征组合:")
        high_prob_combo_counts = high_prob_samples['风险特征组合'].value_counts()
        print(high_prob_combo_counts)
        
        # 10. 输出典型的潜在高风险组合
        print("\n" + "=" * 80)
        print("血脂正常但有潜在风险的核心特征组合")
        print("=" * 80)
        
        core_combinations = []
        
        # 痰湿质高 + 活动能力低
        combo1 = high_prob_samples[(high_prob_samples['痰湿质_离散'] == '高') & (high_prob_samples['活动能力_离散'] == '低')]
        if len(combo1) > 0:
            core_combinations.append({
                '组合': '痰湿质高 + 活动能力低',
                '样本数': len(combo1),
                '平均预测概率': combo1['模型预测概率'].mean(),
                '示例痰湿质': combo1['痰湿质'].mean(),
                '示例活动量表总分': combo1['活动量表总分（ADL总分+IADL总分）'].mean()
            })
        
        # 痰湿质高 + 超重/肥胖
        combo2 = high_prob_samples[(high_prob_samples['痰湿质_离散'] == '高') & (high_prob_samples['BMI_离散'].isin(['超重', '肥胖']))]
        if len(combo2) > 0:
            core_combinations.append({
                '组合': '痰湿质高 + BMI超重/肥胖',
                '样本数': len(combo2),
                '平均预测概率': combo2['模型预测概率'].mean(),
                '示例痰湿质': combo2['痰湿质'].mean(),
                '示例BMI': combo2['BMI'].mean()
            })
        
        # 痰湿质中/高 + 活动能力低 + 其他
        combo3 = high_prob_samples[high_prob_samples['痰湿质_离散'].isin(['中', '高']) & (high_prob_samples['活动能力_离散'] == '低')]
        if len(combo3) > 0:
            core_combinations.append({
                '组合': '痰湿质中/高 + 活动能力低',
                '样本数': len(combo3),
                '平均预测概率': combo3['模型预测概率'].mean(),
                '示例痰湿质': combo3['痰湿质'].mean(),
                '示例活动量表总分': combo3['活动量表总分（ADL总分+IADL总分）'].mean()
            })
        
        if core_combinations:
            core_combinations_df = pd.DataFrame(core_combinations)
            print(core_combinations_df.to_string(index=False))
            
            # 保存结果
            output_path = "data/processed/normal_lipid_risk_analysis.csv"
            core_combinations_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\n结果已保存到: {output_path}")
        else:
            print("\n在血脂正常样本中未发现典型的高风险组合")
    
    # 保存完整的血脂正常样本分析结果
    output_path_full = "data/processed/normal_lipid_samples_analysis.csv"
    df_normal_lipid.to_csv(output_path_full, index=False, encoding='utf-8-sig')
    print(f"\n完整血脂正常样本分析结果已保存到: {output_path_full}")
    
    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)

if __name__ == "__main__":
    main()
