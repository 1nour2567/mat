#!/usr/bin/env python3
# 特征工程：提取衍生特征，选择关键特征

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data():
    """加载数据"""
    try:
        df = pd.read_pickle('/workspace/Project_mat/data/processed/preprocessed_data.pkl')
        print(f"成功加载数据，样本数：{len(df)}")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def feature_engineering(df):
    """特征工程"""
    # 重命名缩尾列，使其与用户提供的名称一致
    df = df.rename(columns={
        'non-HDL-C_缩尾': 'non-HDL-C_clip',
        'AIP_缩尾': 'AIP_clip',
        'TC/HDL比值_缩尾': 'TC/HDL比值_clip',
        'LDL/HDL比值_缩尾': 'LDL/HDL比值_clip',
        'TG/HDL比值_缩尾': 'TG/HDL比值_clip'
    })
    
    # 用户提供的前十个关键指标
    top_10_features = [
        'non-HDL-C_clip',
        'TC/HDL比值_clip',
        'TC（总胆固醇）',
        'TG/HDL比值_clip',
        'AIP_clip',
        'TG（甘油三酯）',
        '血尿酸',
        'LDL/HDL比值_clip',
        'HDL-C（高密度脂蛋白）',
        'ADL吃饭'
    ]
    
    # 检查这些特征是否存在
    missing_features = [feat for feat in top_10_features if feat not in df.columns]
    if missing_features:
        print(f"警告：以下特征不存在：{missing_features}")
    
    # 选择存在的特征
    available_features = [feat for feat in top_10_features if feat in df.columns]
    print(f"可用的特征：{available_features}")
    
    # 提取特征和标签
    X = df[available_features]
    y = df['高血脂症二分类标签']
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 转换为DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=available_features)
    
    print(f"\n特征工程完成：")
    print(f"特征数量：{X_scaled_df.shape[1]}")
    print(f"样本数量：{X_scaled_df.shape[0]}")
    print(f"\n特征描述：")
    print(X_scaled_df.describe())
    
    return X_scaled_df, y, available_features, scaler

def main():
    print("=== 特征工程 ===")
    df = load_data()
    if df is not None:
        X, y, features, scaler = feature_engineering(df)
        print("\n特征工程完成！")
        
        # 保存处理后的数据
        try:
            processed_data = pd.concat([X, y.reset_index(drop=True)], axis=1)
            processed_data.to_pickle('/workspace/Project_mat/data/processed/featured_data.pkl')
            print("处理后的数据已保存到 /workspace/Project_mat/data/processed/featured_data.pkl")
        except Exception as e:
            print(f"保存数据失败: {e}")

if __name__ == "__main__":
    main()
