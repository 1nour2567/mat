# 三层融合预警模型 - 主流程
import pandas as pd
import numpy as np
import os
import sys

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import preprocess_data
from src.three_layer_architecture import TripleLayerPredictor


def main():
    print("=" * 80)
    print("三层融合预警模型 - 完整流程")
    print("=" * 80)
    
    # 1. 数据预处理
    print("\n[步骤1] 数据预处理...")
    raw_data_path = "data/raw/附件1：样例数据.xlsx"
    processed_data_path = "data/processed/preprocessed_data.pkl"
    
    if not os.path.exists(processed_data_path):
        df_processed = preprocess_data(raw_data_path, processed_data_path)
    else:
        print(f"加载已预处理的数据: {processed_data_path}")
        df_processed = pd.read_pickle(processed_data_path)
    
    print(f"预处理完成，数据形状: {df_processed.shape}")
    
    # 2. 初始化三层预测器
    print("\n[步骤2] 初始化三层预测器...")
    predictor = TripleLayerPredictor()
    
    # 3. 训练模型
    print("\n[步骤3] 训练模型...")
    print("\n--- 开始第一层：统计模型层训练 ---")
    predictor.fit(df_processed, target_col='高血脂症二分类标签')
    print("--- 第一层训练完成 ---")
    
    # 4. 预测
    print("\n[步骤4] 完整预测流程...")
    df_result = predictor.predict(df_processed)
    
    # 5. 分析结果
    print("\n[步骤5] 结果分析...")
    analyze_results(df_result)
    
    # 6. 保存结果
    output_path = "data/processed/three_layer_result.pkl"
    df_result.to_pickle(output_path)
    print(f"\n结果已保存到: {output_path}")
    
    # 7. 打印样本结果
    print("\n" + "=" * 80)
    print("样本预测结果:")
    print("=" * 80)
    sample_cols = ['痰湿质', '活动量表总分（ADL总分+IADL总分）', 
                   '血脂异常项数', '临床确诊高风险', 
                   '模型预测概率', '最终风险等级', 
                   '高血脂症二分类标签']
    
    # 显示不同风险等级的样本
    risk_levels = df_result['最终风险等级'].unique()
    for level in risk_levels:
        print(f"\n--- {level} ---")
        samples = df_result[df_result['最终风险等级'] == level].head(2)
        if len(samples) > 0:
            print(samples[sample_cols].to_string())
    
    print("\n" + "=" * 80)
    print("三层融合预警模型流程完成！")
    print("=" * 80)
    

def analyze_results(df):
    """
    分析预测结果
    
    Args:
        df: 包含预测结果的数据框
    """
    print("\n--- 结果统计 ---")
    
    # 风险等级分布
    risk_dist = df['最终风险等级'].value_counts()
    print("\n最终风险等级分布:")
    for level, count in risk_dist.items():
        print(f"  {level}: {count} ({count/len(df)*100:.1f}%)")
    
    # 临床确诊高风险统计
    if '临床确诊高风险' in df.columns:
        clinical_count = df['临床确诊高风险'].sum()
        print(f"\n临床确诊高风险人数: {clinical_count} ({clinical_count/len(df)*100:.1f}%)")
    
    # 中医规则修正情况
    if '模型预测概率' in df.columns:
        # 统计被修正的样本
        uncertainty_mask = (df['模型预测概率'] >= 0.35) & (df['模型预测概率'] <= 0.65)
        uncertain_count = uncertainty_mask.sum()
        print(f"\n模型不确定区间样本数: {uncertain_count} ({uncertain_count/len(df)*100:.1f}%)")
        
        if uncertain_count > 0:
            uncertain_df = df[uncertain_mask]
            corrected_count = len(uncertain_df) - len(
                uncertain_df[uncertain_df['最终风险等级'] == uncertain_df['模型预测概率'].apply(
                    lambda p: '低风险' if p < 0.35 else ('中风险' if p < 0.65 else '高风险')
                )]
            )
            print(f"被中医规则修正的样本数: {corrected_count} ({corrected_count/uncertain_count*100:.1f}%)")
    
    # 准确率分析
    if '高血脂症二分类标签' in df.columns:
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        # 将最终风险等级转换为二分类
        y_true = df['高血脂症二分类标签'].values
        y_pred_risk = (df['最终风险等级'].isin(['高风险', '临床确诊高风险'])).astype(int)
        y_pred_prob = df['模型预测概率'].values
        
        accuracy = accuracy_score(y_true, y_pred_risk)
        auc = roc_auc_score(y_true, y_pred_prob)
        
        print(f"\n模型预测性能:")
        print(f"  准确率 (Accuracy): {accuracy:.4f}")
        print(f"  AUC-ROC: {auc:.4f}")
        

if __name__ == "__main__":
    main()
