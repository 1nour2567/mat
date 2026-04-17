import pandas as pd
import os

# 定义文件路径
preprocessed_data_path = 'data/processed/preprocessed_data.pkl'
featured_data_path = 'data/processed/featured_data.pkl'
final_data_path = 'data/processed/final_data.pkl'

def read_and_display(file_path, description):
    print(f"\n=== 显示 {description} ===")
    if os.path.exists(file_path):
        df = pd.read_pickle(file_path)
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print("前5行数据:")
        print(df.head())
    else:
        print(f"文件不存在: {file_path}")

# 读取并显示每个文件
read_and_display(preprocessed_data_path, "预处理后的数据")
read_and_display(featured_data_path, "特征工程后的数据")
read_and_display(final_data_path, "最终处理后的数据")
