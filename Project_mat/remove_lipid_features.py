import pandas as pd
import numpy as np
import os

# 要移除的特征列表
FEATURES_TO_REMOVE = [
    'HDL-C（高密度脂蛋白）',
    'LDL-C（低密度脂蛋白）',
    'TG（甘油三酯）',
    'TC（总胆固醇）',
    'AIP',
    'TC/HDL比值',
    'non-HDL-C'
]

def remove_lipid_features():
    """
    移除血脂相关特征并保存处理后的数据
    """
    # 加载原始数据
    input_path = 'data/processed/preprocessed_data.pkl'
    output_path = 'data/processed/preprocessed_data_no_lipid.pkl'
    
    print(f"加载数据: {input_path}")
    df = pd.read_pickle(input_path)
    
    print(f"原始数据形状: {df.shape}")
    print(f"原始列名: {list(df.columns)}")
    
    # 移除指定特征
    existing_features = [f for f in FEATURES_TO_REMOVE if f in df.columns]
    print(f"要移除的存在特征: {existing_features}")
    
    df_processed = df.drop(columns=existing_features, errors='ignore')
    
    print(f"处理后数据形状: {df_processed.shape}")
    print(f"处理后列名: {list(df_processed.columns)}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存处理后的数据
    df_processed.to_pickle(output_path)
    print(f"处理后数据已保存: {output_path}")
    
    return df_processed

if __name__ == "__main__":
    remove_lipid_features()
