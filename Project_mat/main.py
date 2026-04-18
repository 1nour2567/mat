# 主运行脚本
import os
import pandas as pd
import sys

sys.path.append('/workspace/Project_mat')

from src.preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.risk_model import train_risk_model, ensemble_predict, classify_risk_level
from src.intervention_optimizer import optimize_interventions
from src.visualization import visualize_results
from config.constants import INTERVENTION_PARAMS

# 路径配置
BASE_DIR = '/workspace/Project_mat'
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data/raw/附件1：样例数据.xlsx')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data/processed/preprocessed_data.pkl')
FEATURED_DATA_PATH = os.path.join(BASE_DIR, 'data/processed/featured_data.pkl')
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, 'data/processed/models.pkl')
FINAL_DATA_PATH = os.path.join(BASE_DIR, 'data/processed/final_data.pkl')

# 目标变量
TARGET = '高血脂症二分类标签'

# 预算配置
BUDGET = INTERVENTION_PARAMS['constraints']['max_total_cost']

def main():
    """主运行函数"""
    print("=== 开始运行流程 ===")
    
    # 1. 数据预处理
    print("\n1. 数据预处理")
    preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    print("数据预处理完成")
    
    # 2. 特征工程
    print("\n2. 特征工程")
    print("构建三级候选特征集：")
    print("•基础特征层：九种体质积分+体质标签、TC/TG/LDL-C/HDL-C/GLU/UA/BMI、ADL/IADL总分及分项、人口统计学信息")
    print("•派生特征层：non-HDL-C、AIP、TC/HDL、LDL/HDL、TG/HDL、血脂异常项数、尿酸异常标志")
    print("•中西医交叉特征层：痰湿质得分×BMI、痰湿质得分×TG、痰湿质得分×AIP、痰湿质得分×LDL-C、痰湿质得分/HDL-C、气虚质得分×TC")
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
    
    # 计算风险等级（使用规则和概率）
    risk_levels = []
    for i, row in df.iterrows():
        lipid_abnormality_count = row.get('血脂异常项数', 0)
        phlegm_score = row.get('痰湿质得分', 0)
        activity_score = row.get('ADL总分', 0)
        prob = risk_probs[i]
        risk_level = classify_risk_level(prob, lipid_abnormality_count, phlegm_score, activity_score)
        risk_levels.append(risk_level)
    df['risk_level'] = risk_levels
    
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
