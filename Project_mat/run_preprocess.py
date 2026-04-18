#!/usr/bin/env python3
# 临时执行预处理脚本
from src.01_preprocessing import preprocess_data

if __name__ == "__main__":
    preprocess_data('data/raw/附件1：样例数据.xlsx', 'data/processed/preprocessed_data.pkl')
    print("预处理完成")