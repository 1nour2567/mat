# 特征池构建与筛选模块
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

def build_feature_pool(df):
    """构建特征池"""
    # 基础特征
    features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 类别特征编码
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_features:
        df = pd.get_dummies(df, columns=[col], drop_first=True)
    
    # 更新特征列表
    features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    return df, features

def select_features(df, features, target, k=20):
    """特征筛选"""
    X = df[features]
    y = df[target]
    
    # 特征选择
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    
    # 获取选中的特征
    selected_features = [features[i] for i in selector.get_support(indices=True)]
    
    return selected_features

def feature_engineering(input_path, output_path, target):
    """完整特征工程流程"""
    # 加载预处理后的数据
    df = pd.read_pickle(input_path)
    
    # 构建特征池
    df, features = build_feature_pool(df)
    
    # 特征筛选
    selected_features = select_features(df, features, target)
    
    # 标准化
    scaler = StandardScaler()
    df[selected_features] = scaler.fit_transform(df[selected_features])
    
    # 保存处理后的数据
    df.to_pickle(output_path)
    
    return df, selected_features