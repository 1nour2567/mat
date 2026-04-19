import pandas as pd
import pickle

data = pickle.load(open('/workspace/Project_mat/data/processed/preprocessed_data.pkl', 'rb'))
print('All columns:')
for col in list(data.columns):
    print(f'  {col}')

print(f'\nShape: {data.shape}')

# 检查关键指标是否存在
print('\nChecking key columns:')
key_cols = ['痰湿质', 'TG', 'TC', 'HDL-C', 'LDL-C', '血尿酸', 'BMI', 
            'ADL总分', 'IADL总分', '活动量表总分', '高血脂症二分类标签']
for col in key_cols:
    print(f'  {col}: {"✓ Present" if col in data.columns else "✗ Missing"}')

