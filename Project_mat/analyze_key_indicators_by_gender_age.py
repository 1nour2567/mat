# 按性别和年龄组分析关键指标
import pandas as pd
import numpy as np
from src.feature_engineering import build_feature_pool, select_features
import warnings
warnings.filterwarnings('ignore')

def analyze_by_gender_age():
    """按性别和年龄组分析关键指标"""
    print("=== 按性别和年龄组分析关键指标 ===")
    
    # 加载预处理后的数据
    df = pd.read_pickle('data/processed/preprocessed_data.pkl')
    
    # 处理NaN值
    df = df.fillna(0)
    
    # 确保性别和年龄组列存在
    if '性别' not in df.columns or '年龄组' not in df.columns:
        print("错误：数据中缺少性别或年龄组列")
        return
    
    # 定义目标变量
    target = '高血脂症二分类标签'
    
    # 性别分组
    gender_groups = df['性别'].unique()
    gender_names = {1: '男性', 0: '女性'}
    
    # 年龄组分组
    age_groups = sorted(df['年龄组'].unique())
    age_group_names = {
        1: '40-49岁',
        2: '50-59岁',
        3: '60-69岁',
        4: '70-79岁',
        5: '80-89岁'
    }
    
    # 按性别分析
    print("\n=== 按性别分析 ===")
    for gender in gender_groups:
        gender_df = df[df['性别'] == gender]
        if len(gender_df) < 10:
            print(f"{gender_names.get(gender, f'性别{gender}')}样本量不足，跳过分析")
            continue
        
        print(f"\n{gender_names.get(gender, f'性别{gender}')} (n={len(gender_df)})")
        
        # 构建特征池
        gender_df, features = build_feature_pool(gender_df)
        
        # 筛选关键指标
        selected = select_features(gender_df, features, target, k=10)
        print(f"关键指标: {selected[:5]}")
    
    # 按年龄组分析
    print("\n=== 按年龄组分析 ===")
    for age_group in age_groups:
        age_df = df[df['年龄组'] == age_group]
        if len(age_df) < 10:
            print(f"{age_group_names.get(age_group, f'年龄组{age_group}')}样本量不足，跳过分析")
            continue
        
        print(f"\n{age_group_names.get(age_group, f'年龄组{age_group}')} (n={len(age_df)})")
        
        # 构建特征池
        age_df, features = build_feature_pool(age_df)
        
        # 筛选关键指标
        selected = select_features(age_df, features, target, k=10)
        print(f"关键指标: {selected[:5]}")
    
    # 按性别和年龄组交叉分析
    print("\n=== 按性别和年龄组交叉分析 ===")
    for gender in gender_groups:
        for age_group in age_groups:
            cross_df = df[(df['性别'] == gender) & (df['年龄组'] == age_group)]
            if len(cross_df) < 10:
                continue
            
            print(f"\n{gender_names.get(gender, f'性别{gender}')} - {age_group_names.get(age_group, f'年龄组{age_group}')} (n={len(cross_df)})")
            
            # 构建特征池
            cross_df, features = build_feature_pool(cross_df)
            
            # 筛选关键指标
            selected = select_features(cross_df, features, target, k=10)
            print(f"关键指标: {selected[:5]}")

if __name__ == "__main__":
    analyze_by_gender_age()