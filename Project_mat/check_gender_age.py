#!/usr/bin/env python3
# 检查性别和年龄处理是否正确
import pandas as pd
import numpy as np

def check_gender_age():
    """
    检查数据中的性别处理是否正确（0为女性，1为男性）
    以及年龄组的处理是否正确
    """
    print("=== 检查性别和年龄处理 ===")
    
    # 加载数据
    data_path = '/workspace/预处理后数据.csv'
    encodings = ['utf-8', 'gbk', 'latin1']
    for encoding in encodings:
        try:
            df = pd.read_csv(data_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码加载数据")
            break
        except:
            continue
    else:
        raise Exception("无法加载数据，请检查文件编码")
    
    # 检查性别列
    print("\n1. 性别处理检查：")
    if '性别' in df.columns and '性别名称' in df.columns:
        # 检查性别值范围
        gender_values = df['性别'].unique()
        print(f"性别值范围: {sorted(gender_values)}")
        
        # 检查性别与性别名称的对应关系
        gender_name_mapping = df.groupby('性别')['性别名称'].unique().to_dict()
        print("性别与性别名称的对应关系：")
        for gender, names in gender_name_mapping.items():
            print(f"  性别 {gender}: {names}")
        
        # 检查是否符合 0=女，1=男 的规则
        if 0 in gender_name_mapping and 1 in gender_name_mapping:
            if '女' in gender_name_mapping[0] and '男' in gender_name_mapping[1]:
                print("✓ 性别处理正确：0=女，1=男")
            else:
                print("✗ 性别处理错误：不符合 0=女，1=男 的规则")
        else:
            print("✗ 性别处理错误：缺少必要的性别值")
    else:
        print("✗ 性别处理错误：缺少性别列")
    
    # 检查年龄组
    print("\n2. 年龄组处理检查：")
    if '年龄组' in df.columns and '年龄组名称' in df.columns:
        # 检查年龄组值范围
        age_group_values = df['年龄组'].unique()
        print(f"年龄组值范围: {sorted(age_group_values)}")
        
        # 检查年龄组与年龄组名称的对应关系
        age_group_name_mapping = df.groupby('年龄组')['年龄组名称'].unique().to_dict()
        print("年龄组与年龄组名称的对应关系：")
        for age_group, names in age_group_name_mapping.items():
            print(f"  年龄组 {age_group}: {names}")
        
        # 检查是否符合预期的年龄组划分
        expected_age_groups = {1: '40-49岁', 2: '50-59岁', 3: '60-69岁', 4: '70-79岁', 5: '80-89岁'}
        match = True
        for age_group, expected_name in expected_age_groups.items():
            if age_group in age_group_name_mapping:
                if expected_name in age_group_name_mapping[age_group]:
                    print(f"✓ 年龄组 {age_group} 对应正确: {expected_name}")
                else:
                    print(f"✗ 年龄组 {age_group} 对应错误: 期望 {expected_name}, 实际 {age_group_name_mapping[age_group]}")
                    match = False
            else:
                print(f"✗ 年龄组 {age_group} 缺失")
                match = False
        
        if match:
            print("✓ 年龄组处理正确")
        else:
            print("✗ 年龄组处理错误")
    else:
        print("✗ 年龄组处理错误：缺少年龄组列")
    
    # 检查数据量
    print(f"\n3. 数据量检查：")
    print(f"总样本数: {len(df)}")
    if '性别' in df.columns:
        gender_counts = df['性别'].value_counts()
        print("性别分布：")
        for gender, count in gender_counts.items():
            print(f"  性别 {gender}: {count} ({count/len(df)*100:.1f}%)")
    
    if '年龄组' in df.columns:
        age_group_counts = df['年龄组'].value_counts().sort_index()
        print("年龄组分布：")
        for age_group, count in age_group_counts.items():
            print(f"  年龄组 {age_group}: {count} ({count/len(df)*100:.1f}%)")

if __name__ == "__main__":
    check_gender_age()
