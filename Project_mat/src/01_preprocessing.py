# 数据清洗与初始衍生模块
import pandas as pd
import numpy as np
from config.constants import AGE_CONSTRAINTS

def load_raw_data(file_path):
    """加载原始数据"""
    return pd.read_excel(file_path)

def clean_data(df):
    """数据清洗"""
    # 处理缺失值
    df = df.dropna()
    
    # 年龄约束过滤
    df = df[(df['age'] >= AGE_CONSTRAINTS['min_age']) & (df['age'] <= AGE_CONSTRAINTS['max_age'])]
    
    # 其他清洗逻辑...
    
    return df

def feature_derivation(df):
    """初始特征衍生"""
    # 衍生特征示例
    df['age_group'] = pd.cut(df['age'], bins=[18, 30, 45, 60, 65], labels=['18-30', '31-45', '46-60', '61-65'])
    
    # 其他衍生逻辑...
    
    return df

def preprocess_data(raw_data_path, output_path):
    """完整预处理流程"""
    # 加载数据
    df = load_raw_data(raw_data_path)
    
    # 清洗数据
    df = clean_data(df)
    
    # 特征衍生
    df = feature_derivation(df)
    
    # 保存处理后的数据
    df.to_pickle(output_path)
    
    return df