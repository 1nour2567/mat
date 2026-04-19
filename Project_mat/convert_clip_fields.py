#!/usr/bin/env python3
# 将数据中的缩尾字段重命名为clip

import pandas as pd
import numpy as np

# 读取pkl文件
def read_pkl_file(file_path):
    """读取pkl文件并返回数据"""
    try:
        data = pd.read_pickle(file_path)
        print(f"成功读取文件: {file_path}")
        print(f"数据形状: {data.shape}")
        return data
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

# 替换字段
def replace_fields(data):
    """替换字段名，将缩尾改为clip"""
    # 需要替换的字段映射
    field_mapping = {
        'non-HDL-C_缩尾': 'non-HDL-C_clip',
        'AIP_缩尾': 'AIP_clip',
        'TC/HDL比值_缩尾': 'TC/HDL_clip',
        'LDL/HDL比值_缩尾': 'LDL/HDL_clip',
        'TG/HDL比值_缩尾': 'TG/HDL_clip'
    }
    
    # 重命名字段
    data = data.rename(columns=field_mapping)
    print("字段重命名完成:")
    for old_name, new_name in field_mapping.items():
        print(f"{old_name} → {new_name}")
    
    # 检查是否存在原始字段
    original_fields = ['non-HDL-C', 'AIP', 'TC/HDL比值', 'LDL/HDL比值', 'TG/HDL比值']
    for field in original_fields:
        if field in data.columns:
            print(f"注意: 原始字段 {field} 仍然存在")
    
    return data

# 保存文件
def save_file(data, output_path):
    """保存数据到新文件"""
    try:
        data.to_pickle(output_path)
        print(f"成功保存文件: {output_path}")
    except Exception as e:
        print(f"保存文件失败: {e}")

if __name__ == "__main__":
    input_path = '/workspace/Project_mat/data/processed/preprocessed_data.pkl'
    output_path = '/workspace/Project_mat/data/processed/preprocessed_data_clip.pkl'
    
    # 读取数据
    data = read_pkl_file(input_path)
    
    if data is not None:
        # 替换字段
        data = replace_fields(data)
        
        # 保存文件
        save_file(data, output_path)
        
        # 显示前几行数据
        print("\n处理后的数据前5行:")
        print(data.head())
        
        # 检查新字段是否存在
        new_fields = ['non-HDL-C_clip', 'AIP_clip', 'TC/HDL_clip', 'LDL/HDL_clip', 'TG/HDL_clip']
        print("\n新字段存在情况:")
        for field in new_fields:
            if field in data.columns:
                print(f"✓ {field}: 存在")
            else:
                print(f"✗ {field}: 不存在")