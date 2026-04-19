#!/usr/bin/env python3
# 模型训练：使用多种模型训练风险预警模型

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def load_data():
    """加载处理后的数据"""
    try:
        df = pd.read_pickle('/workspace/Project_mat/data/processed/featured_data.pkl')
        print(f"成功加载数据，样本数：{len(df)}")
        
        # 分离特征和标签
        X = df.drop('高血脂症二分类标签', axis=1)
        y = df['高血脂症二分类标签']
        
        print(f"特征数量：{X.shape[1]}")
        print(f"标签分布：{y.value_counts()}")
        
        return X, y
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None, None

def train_models(X, y):
    """训练多种模型"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"训练集大小：{X_train.shape[0]}")
    print(f"测试集大小：{X_test.shape[0]}")
    
    # 模型列表
    models = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    # 训练和评估模型
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # 保存结果
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        print(f"{name} 性能：")
        print(f"准确率：{accuracy:.4f}")
        print(f"精确率：{precision:.4f}")
        print(f"召回率：{recall:.4f}")
        print(f"F1值：{f1:.4f}")
        print(f"AUC：{auc:.4f}")
        
        # 保存模型
        joblib.dump(model, f'/workspace/Project_mat/models/{name}.joblib')
        print(f"{name} 模型已保存到 /workspace/Project_mat/models/{name}.joblib")
        
        # 选择最佳模型
        if auc > best_score:
            best_score = auc
            best_model = model
            best_model_name = name
    
    # 保存最佳模型
    if best_model is not None:
        joblib.dump(best_model, '/workspace/Project_mat/models/best_model.joblib')
        print(f"\n最佳模型：{best_model_name}")
        print(f"最佳模型AUC：{best_score:.4f}")
        print("最佳模型已保存到 /workspace/Project_mat/models/best_model.joblib")
    
    # 保存评估结果
    results_df = pd.DataFrame(results).T
    results_df.to_csv('/workspace/Project_mat/models/model_evaluation.csv')
    print("\n模型评估结果已保存到 /workspace/Project_mat/models/model_evaluation.csv")
    
    return results, best_model, X_test, y_test

def main():
    print("=== 模型训练 ===")
    
    # 创建模型目录
    import os
    os.makedirs('/workspace/Project_mat/models', exist_ok=True)
    
    X, y = load_data()
    if X is not None and y is not None:
        results, best_model, X_test, y_test = train_models(X, y)
        print("\n模型训练完成！")

if __name__ == "__main__":
    main()
