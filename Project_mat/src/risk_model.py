# 三级融合预警模型模块
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config.constants import THRESHOLDS, DEFAULT_TRAIN_TEST_SPLIT, RANDOM_SEED

def train_base_models(X_train, y_train):
    """训练基础模型"""
    # 模型1：随机森林
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf_model.fit(X_train, y_train)
    
    # 模型2：梯度提升树
    gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
    gbm_model.fit(X_train, y_train)
    
    # 模型3：逻辑回归
    lr_model = LogisticRegression(random_state=RANDOM_SEED)
    lr_model.fit(X_train, y_train)
    
    return rf_model, gbm_model, lr_model

def ensemble_predict(models, X):
    """融合预测"""
    # 获取各模型的预测概率
    rf_proba = models[0].predict_proba(X)[:, 1]
    gbm_proba = models[1].predict_proba(X)[:, 1]
    lr_proba = models[2].predict_proba(X)[:, 1]
    
    # 融合概率（简单平均）
    ensemble_proba = (rf_proba + gbm_proba + lr_proba) / 3
    
    return ensemble_proba

def classify_risk_level(prob):
    """根据概率划分风险等级"""
    if prob >= THRESHOLDS['risk_level_3']:
        return 3
    elif prob >= THRESHOLDS['risk_level_2']:
        return 2
    elif prob >= THRESHOLDS['risk_level_1']:
        return 1
    else:
        return 0

def train_risk_model(input_path, model_output_path, target):
    """完整风险模型训练流程"""
    # 加载特征工程后的数据
    df = pd.read_pickle(input_path)
    
    # 准备特征和目标
    features = [col for col in df.columns if col != target]
    X = df[features]
    y = df[target]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-DEFAULT_TRAIN_TEST_SPLIT, random_state=RANDOM_SEED)
    
    # 训练基础模型
    models = train_base_models(X_train, y_train)
    
    # 融合预测
    y_test_proba = ensemble_predict(models, X_test)
    
    # 评估模型
    y_test_pred = [1 if p >= THRESHOLDS['risk_level_1'] else 0 for p in y_test_proba]
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    print(f"模型评估指标：")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    # 保存模型
    joblib.dump(models, model_output_path)
    print(f"模型已保存至: {model_output_path}")
    
    return models, (accuracy, precision, recall, f1)