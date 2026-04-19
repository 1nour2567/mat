#!/usr/bin/env python3
# 检查数据文件中的列

import pandas as pd

print("=== 检查数据文件中的列 ===")

# 加载数据
try:
    df = pd.read_pickle('/workspace/Project_mat/data/processed/preprocessed_data.pkl')
    print(f"成功加载数据，样本数：{len(df)}")
    print(f"列数：{len(df.columns)}")
    print("\n所有列名：")
    for i, col in enumerate(sorted(df.columns)):
        print(f"{i+1}. {col}")
        
    # 检查是否有缩尾相关的列
    print("\n缩尾相关的列：")
    for col in sorted(df.columns):
        if '缩尾' in col:
            print(f"- {col}")
            
    # 检查是否有AIP相关的列
    print("\nAIP相关的列：")
    for col in sorted(df.columns):
        if 'AIP' in col:
            print(f"- {col}")
            
    # 检查是否有TC/HDL、LDL/HDL、TG/HDL相关的列
    print("\n比值相关的列：")
    for col in sorted(df.columns):
        if any(x in col for x in ['TC/HDL', 'LDL/HDL', 'TG/HDL', 'non-HDL']):
            print(f"- {col}")
            
except Exception as e:
    print(f"加载数据失败: {e}")
