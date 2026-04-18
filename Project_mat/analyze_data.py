#!/usr/bin/env python3
# 数据分析脚本
import pandas as pd
import numpy as np

# 加载数据
df = pd.read_excel('data/raw/附件1：样例数据.xlsx')

# 查看数据基本信息
print("数据形状:", df.shape)
print("\n列名:", df.columns.tolist())
print("\n数据类型:")
print(df.dtypes)
print("\n前5行数据:")
print(df.head())
print("\n数据描述:")
print(df.describe())

# 保存数据以便后续分析
df.to_pickle('data/processed/raw_data.pkl')
print("\n数据已保存到 data/processed/raw_data.pkl")