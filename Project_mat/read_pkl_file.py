#!/usr/bin/env python3
# 读取并分析preprocessed_data.pkl文件

import pandas as pd
import numpy as np

# 读取pkl文件
def read_pkl_file(file_path):
    """读取pkl文件并返回数据"""
    try:
        data = pd.read_pickle(file_path)
        print(f"成功读取文件: {file_path}")
        print(f"数据形状: {data.shape}")
        print(f"\n列名:")
        print(data.columns.tolist())
        print(f"\n数据类型:")
        print(data.dtypes)
        print(f"\n前5行数据:")
        print(data.head())
        print(f"\n数据统计信息:")
        print(data.describe())
        return data
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

if __name__ == "__main__":
    file_path = '/workspace/Project_mat/data/processed/preprocessed_data.pkl'
    data = read_pkl_file(file_path)
    
    # 如果数据读取成功，进一步分析
    if data is not None:
        # 检查是否包含关键指标
        key_columns = ['TC（总胆固醇）', 'TG（甘油三酯）', '血尿酸', 'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）', '痰湿质']
        print(f"\n关键指标存在情况:")
        for col in key_columns:
            if col in data.columns:
                print(f"✓ {col}: 存在")
            else:
                print(f"✗ {col}: 不存在")
        
        # 检查性别和年龄组分布
        if '性别' in data.columns:
            print(f"\n性别分布:")
            print(data['性别'].value_counts())
        
        if '年龄组' in data.columns:
            print(f"\n年龄组分布:")
            print(data['年龄组'].value_counts())