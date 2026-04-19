#!/usr/bin/env python3
# 检查数据结构

import pandas as pd
import numpy as np

def load_data():
    """加载数据"""
    try:
        # 加载preprocessed_data.pkl文件
        df = pd.read_pickle('/workspace/Project_mat/data/processed/preprocessed_data.pkl')
        print(f"成功加载数据，样本数：{len(df)}")
        print(f"数据列数：{len(df.columns)}")
        print("\n数据列名：")
        for col in df.columns:
            print(f"  - {col}")
        
        # 查看前几行数据
        print("\n前5行数据：")
        print(df.head())
        
        # 查看数据类型
        print("\n数据类型：")
        print(df.dtypes)
        
        # 查看年龄组分布
        if '年龄组' in df.columns:
            print("\n年龄组分布：")
            print(df['年龄组'].value_counts())
        
        # 查看性别分布
        if '性别' in df.columns:
            print("\n性别分布：")
            print(df['性别'].value_counts())
        
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def main():
    print("=== 检查数据结构 ===")
    df = load_data()
    if df is not None:
        print("\n数据结构检查完成！")

if __name__ == "__main__":
    main()
