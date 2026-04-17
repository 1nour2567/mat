# 数据清洗与初始衍生模块
import pandas as pd
import numpy as np
from config.constants import AGE_CONSTRAINTS, ACTIVITY_LEVELS, PHLEGM_DAMP_SCORES

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

def calculate_metabolic_indicators(df):
    """计算代谢派生指标"""
    # non-HDL-C = TC - HDL-C
    if 'TC' in df.columns and 'HDL-C' in df.columns:
        df['non-HDL-C'] = df['TC'] - df['HDL-C']
    
    # AIP = log10(TG/HDL-C)
    if 'TG' in df.columns and 'HDL-C' in df.columns:
        df['AIP'] = np.log10(df['TG'] / df['HDL-C'])
    
    # TC/HDL比值
    if 'TC' in df.columns and 'HDL-C' in df.columns:
        df['TC/HDL'] = df['TC'] / df['HDL-C']
    
    # LDL/HDL比值
    if 'LDL-C' in df.columns and 'HDL-C' in df.columns:
        df['LDL/HDL'] = df['LDL-C'] / df['HDL-C']
    
    # TG/HDL比值
    if 'TG' in df.columns and 'HDL-C' in df.columns:
        df['TG/HDL'] = df['TG'] / df['HDL-C']
    
    # 1%-99%缩尾版本
    for col in ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']:
        if col in df.columns:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[f'{col}_winsorized'] = df[col].clip(lower=q1, upper=q99)
    
    return df

def calculate_clinical_flags(df):
    """计算临床异常标志"""
    # TC异常（假设正常范围：3.1-5.2 mmol/L）
    if 'TC' in df.columns:
        df['TC_abnormal'] = ((df['TC'] < 3.1) | (df['TC'] > 5.2)).astype(int)
    
    # TG异常（假设正常范围：0.56-1.7 mmol/L）
    if 'TG' in df.columns:
        df['TG_abnormal'] = ((df['TG'] < 0.56) | (df['TG'] > 1.7)).astype(int)
    
    # LDL-C异常（假设正常范围：<3.4 mmol/L）
    if 'LDL-C' in df.columns:
        df['LDL_abnormal'] = (df['LDL-C'] >= 3.4).astype(int)
    
    # HDL-C异常（假设正常范围：1.04-1.55 mmol/L）
    if 'HDL-C' in df.columns:
        df['HDL_abnormal'] = ((df['HDL-C'] < 1.04) | (df['HDL-C'] > 1.55)).astype(int)
    
    # 血脂异常项数
    lipid_abnormal_cols = ['TC_abnormal', 'TG_abnormal', 'LDL_abnormal', 'HDL_abnormal']
    df['lipid_abnormal_count'] = df[lipid_abnormal_cols].sum(axis=1)
    
    # 尿酸异常（假设正常范围：150-420 μmol/L）
    if 'UA' in df.columns:
        df['UA_abnormal'] = ((df['UA'] < 150) | (df['UA'] > 420)).astype(int)
    
    return df

def add_stratification_variables(df):
    """添加赛题约束分层变量"""
    # 活动能力分层（<40/40-59/≥60）
    if 'activity_score' in df.columns:
        df['activity_level'] = pd.cut(df['activity_score'], bins=[-float('inf'), 39, 59, float('inf')], labels=['<40', '40-59', '≥60'])
    
    # 痰湿积分分层（≤58/59-61/≥62）
    if 'phlegm_damp_score' in df.columns:
        df['phlegm_damp_level'] = pd.cut(df['phlegm_damp_score'], bins=[-float('inf'), 58, 61, float('inf')], labels=['≤58', '59-61', '≥62'])
    
    return df

def feature_derivation(df):
    """初始特征衍生"""
    # 计算代谢派生指标
    df = calculate_metabolic_indicators(df)
    
    # 计算临床异常标志
    df = calculate_clinical_flags(df)
    
    # 添加分层变量
    df = add_stratification_variables(df)
    
    # 年龄分组
    df['age_group'] = pd.cut(df['age'], bins=[18, 30, 45, 60, 65], labels=['18-30', '31-45', '46-60', '61-65'])
    
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