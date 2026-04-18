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
    
    # 其他清洗逻辑...
    
    return df

def feature_derivation(df):
    """初始特征衍生"""
    # 代谢派生指标
    if 'TC（总胆固醇）' in df.columns and 'HDL-C（高密度脂蛋白）' in df.columns:
        df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    
    if 'TG（甘油三酯）' in df.columns and 'HDL-C（高密度脂蛋白）' in df.columns:
        df['AIP'] = np.log10(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    
    if 'TC（总胆固醇）' in df.columns and 'HDL-C（高密度脂蛋白）' in df.columns:
        df['TC/HDL比值'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    
    if 'LDL-C（低密度脂蛋白）' in df.columns and 'HDL-C（高密度脂蛋白）' in df.columns:
        df['LDL/HDL比值'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    
    if 'TG（甘油三酯）' in df.columns and 'HDL-C（高密度脂蛋白）' in df.columns:
        df['TG/HDL比值'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    # 1%-99%缩尾版本
    for col in ['non-HDL-C', 'AIP', 'TC/HDL比值', 'LDL/HDL比值', 'TG/HDL比值']:
        if col in df.columns:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[f'{col}_缩尾'] = df[col].clip(lower=q1, upper=q99)
    
    # 临床异常标志
    if 'TC（总胆固醇）' in df.columns:
        df['TC异常'] = (df['TC（总胆固醇）'] > 5.2).astype(int)
    
    if 'TG（甘油三酯）' in df.columns:
        df['TG异常'] = (df['TG（甘油三酯）'] > 1.7).astype(int)
    
    if 'LDL-C（低密度脂蛋白）' in df.columns:
        df['LDL-C异常'] = (df['LDL-C（低密度脂蛋白）'] > 3.4).astype(int)
    
    if 'HDL-C（高密度脂蛋白）' in df.columns:
        df['HDL-C异常'] = (df['HDL-C（高密度脂蛋白）'] < 1.0).astype(int)
    
    # 血脂异常项数
    lipid_abnormalities = ['TC异常', 'TG异常', 'LDL-C异常', 'HDL-C异常']
    # 只选择存在的列
    existing_lipid_abnormalities = [col for col in lipid_abnormalities if col in df.columns]
    if existing_lipid_abnormalities:
        df['血脂异常项数'] = df[existing_lipid_abnormalities].sum(axis=1)
    else:
        df['血脂异常项数'] = 0
    
    if '血尿酸' in df.columns:
        df['尿酸异常'] = (df['血尿酸'] > 420).astype(int)
    
    # 年龄组已经存在，不需要映射
    if '年龄组' not in df.columns:
        # 如果年龄组不存在，可以根据其他信息生成
        pass
    
    # 赛题约束分层变量
    if 'ADL总分' in df.columns:
        df['活动能力分层'] = pd.cut(df['ADL总分'], bins=[-float('inf'), 39, 59, float('inf')], labels=[0, 1, 2])
    
    if '痰湿质' in df.columns:
        df['痰湿积分分层'] = pd.cut(df['痰湿质'], bins=[-float('inf'), 58, 61, float('inf')], labels=[0, 1, 2])
    
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