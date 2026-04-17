# 主运行脚本
import os
import pandas as pd
from src._01_preprocessing import preprocess_data
from src._02_feature_engineering import feature_engineering
from src._03_risk_model import train_risk_model, ensemble_predict, classify_risk_level
from src._04_intervention_optimizer import optimize_interventions
from src._05_visualization import visualize_results

# 路径配置
RAW_DATA_PATH = 'data/raw/附件1：样例数据.xlsx'
PROCESSED_DATA_PATH = 'data/processed/preprocessed_data.pkl'
FEATURED_DATA_PATH = 'data/processed/featured_data.pkl'
MODEL_OUTPUT_PATH = 'data/processed/models.pkl'
FINAL_DATA_PATH = 'data/processed/final_data.pkl'

# 目标变量
TARGET = '高血脂症二分类标签'  # 使用实际数据中的二分类标签

# 预算配置
BUDGET = 10000  # 假设的预算值，需要根据实际情况调整

def main():
    """主运行函数"""
    print("=== 开始运行流程 ===")
    
    # 1. 数据预处理
    print("\n1. 数据预处理")
    preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    print("数据预处理完成")
    
    # 2. 特征工程
    print("\n2. 特征工程")
    df, selected_features = feature_engineering(PROCESSED_DATA_PATH, FEATURED_DATA_PATH, TARGET)
    print(f"特征工程完成，选中 {len(selected_features)} 个特征")
    
    # 3. 风险模型训练
    print("\n3. 风险模型训练")
    models, metrics = train_risk_model(FEATURED_DATA_PATH, MODEL_OUTPUT_PATH, TARGET)
    print("风险模型训练完成")
    
    # 4. 预测风险等级
    print("\n4. 预测风险等级")
    df = pd.read_pickle(FEATURED_DATA_PATH)
    features = [col for col in df.columns if col != TARGET]
    X = df[features]
    risk_probs = ensemble_predict(models, X)
    df['risk_probability'] = risk_probs
    df['risk_level'] = [classify_risk_level(p) for p in risk_probs]
    df.to_pickle(FINAL_DATA_PATH)
    print("风险等级预测完成")
    
    # 5. 干预优化
    print("\n5. 干预优化")
    df, optimal_strategy, total_reduction = optimize_interventions(FINAL_DATA_PATH, BUDGET)
    df.to_pickle(FINAL_DATA_PATH)
    print("干预优化完成")
    
    # 6. 可视化
    print("\n6. 可视化")
    visualize_results(FINAL_DATA_PATH, models)
    print("可视化完成")
    
    print("\n=== 流程运行结束 ===")

if __name__ == "__main__":
    main()