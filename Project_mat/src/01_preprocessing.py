# 数据清洗与初始衍生模块
import pandas as pd
import numpy as np
from config.constants import AGE_CONSTRAINTS, STRATIFICATION

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
    # 代谢派生指标
    if 'TC' in df.columns and 'HDL-C' in df.columns:
        df['non-HDL-C'] = df['TC'] - df['HDL-C']
    
    if 'TG' in df.columns and 'HDL-C' in df.columns:
        df['AIP'] = np.log10(df['TG'] / df['HDL-C'])
    
    if 'TC' in df.columns and 'HDL-C' in df.columns:
        df['TC/HDL比值'] = df['TC'] / df['HDL-C']
    
    if 'LDL-C' in df.columns and 'HDL-C' in df.columns:
        df['LDL/HDL比值'] = df['LDL-C'] / df['HDL-C']
    
    if 'TG' in df.columns and 'HDL-C' in df.columns:
        df['TG/HDL比值'] = df['TG'] / df['HDL-C']
    
    # 1%-99%缩尾版本
    for col in ['non-HDL-C', 'AIP', 'TC/HDL比值', 'LDL/HDL比值', 'TG/HDL比值']:
        if col in df.columns:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[f'{col}_缩尾'] = df[col].clip(lower=q1, upper=q99)
    
    # 临床异常标志
    if 'TC' in df.columns:
        df['TC异常'] = (df['TC'] > 5.2).astype(int)
    
    if 'TG' in df.columns:
        df['TG异常'] = (df['TG'] > 1.7).astype(int)
    
    if 'LDL-C' in df.columns:
        df['LDL-C异常'] = (df['LDL-C'] > 3.4).astype(int)
    
    if 'HDL-C' in df.columns:
        df['HDL-C异常'] = (df['HDL-C'] < 1.0).astype(int)
    
    # 血脂异常项数
    lipid_abnormalities = ['TC异常', 'TG异常', 'LDL-C异常', 'HDL-C异常']
    df['血脂异常项数'] = df[lipid_abnormalities].sum(axis=1)
    
    if 'UA' in df.columns:
        df['尿酸异常'] = (df['UA'] > 420).astype(int)
    
    # 赛题约束分层变量
    if 'ADL总分' in df.columns:
        df['活动能力分层'] = pd.cut(df['ADL总分'], bins=[-float('inf'), 39, 59, float('inf')], labels=['<40', '40-59', '≥60'])
    
    if '痰湿质得分' in df.columns:
        df['痰湿积分分层'] = pd.cut(df['痰湿质得分'], bins=[-float('inf'), 58, 61, float('inf')], labels=['≤58', '59-61', '≥62'])
    
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