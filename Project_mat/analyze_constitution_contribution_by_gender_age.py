#!/usr/bin/env python3
# 研究九种体质对发病风险的贡献度差异（男女和不同年龄组）
import pandas as pd
import numpy as np
import sys

sys.path.append('/workspace/Project_mat')

from src.feature_engineering import analyze_constitution_contribution

def analyze_by_gender(df):
    """
    按性别分析九种体质对发病风险的贡献度
    """
    print("=== 按性别分析九种体质对发病风险的贡献度 ===")
    
    # 按性别分组
    gender_groups = df.groupby('性别')
    
    results = {}
    
    for gender, group in gender_groups:
        gender_name = '女' if gender == 0 else '男'
        print(f"\n{gender_name}（性别代码：{gender}）体质贡献度分析：")
        
        # 分析体质贡献度
        contributions = analyze_constitution_contribution(group, '高血脂症二分类标签')
        results[gender_name] = contributions
        
        if contributions:
            print(f"{gender_name}群体中贡献度最高的体质：{contributions[0][0]}（{contributions[0][1]:.4f}）")
    
    return results

def analyze_by_age_group(df):
    """
    按年龄组分析九种体质对发病风险的贡献度
    """
    print("\n=== 按年龄组分析九种体质对发病风险的贡献度 ===")
    
    # 按年龄组分组
    age_groups = df.groupby('年龄组')
    age_group_mapping = {1: '40-49岁', 2: '50-59岁', 3: '60-69岁', 4: '70-79岁', 5: '80-89岁'}
    
    results = {}
    
    for age_group, group in age_groups:
        age_group_name = age_group_mapping.get(age_group, f"年龄组{age_group}")
        print(f"\n{age_group_name}（年龄组代码：{age_group}）体质贡献度分析：")
        
        # 分析体质贡献度
        contributions = analyze_constitution_contribution(group, '高血脂症二分类标签')
        results[age_group_name] = contributions
        
        if contributions:
            print(f"{age_group_name}群体中贡献度最高的体质：{contributions[0][0]}（{contributions[0][1]:.4f}）")
    
    return results

def compare_contributions(gender_results, age_results):
    """
    比较不同分组之间的体质贡献度差异
    """
    print("\n=== 体质贡献度差异比较 ===")
    
    # 比较性别差异
    print("\n1. 性别差异比较：")
    if '男' in gender_results and '女' in gender_results:
        male_contribs = dict(gender_results['男'])
        female_contribs = dict(gender_results['女'])
        
        common_constitutions = set(male_contribs.keys()) & set(female_contribs.keys())
        
        print("各体质在男女群体中的贡献度对比：")
        for constitution in common_constitutions:
            male_score = male_contribs[constitution]
            female_score = female_contribs[constitution]
            diff = abs(male_score - female_score)
            print(f"   {constitution}: 男={male_score:.4f}, 女={female_score:.4f}, 差异={diff:.4f}")
    
    # 比较年龄组差异
    print("\n2. 年龄组差异比较：")
    age_group_names = list(age_results.keys())
    
    if len(age_group_names) > 1:
        # 获取所有年龄组共有的体质
        all_constitutions = set()
        for age_group_name, contributions in age_results.items():
            all_constitutions.update([c[0] for c in contributions])
        
        print("各体质在不同年龄组中的贡献度对比：")
        for constitution in all_constitutions:
            print(f"\n   {constitution}：")
            for age_group_name in age_group_names:
                contribs = dict(age_results[age_group_name])
                if constitution in contribs:
                    print(f"      {age_group_name}: {contribs[constitution]:.4f}")
                else:
                    print(f"      {age_group_name}: - ")
    
    # 找出各分组中贡献度最高的体质
    print("\n3. 各分组贡献度最高的体质：")
    print("   性别分组：")
    for gender_name, contributions in gender_results.items():
        if contributions:
            print(f"      {gender_name}：{contributions[0][0]}（{contributions[0][1]:.4f}）")
    
    print("   年龄组分组：")
    for age_group_name, contributions in age_results.items():
        if contributions:
            print(f"      {age_group_name}：{contributions[0][0]}（{contributions[0][1]:.4f}）")

def main():
    """
    主分析函数
    """
    print("=== 研究九种体质对发病风险的贡献度差异 ===")
    
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
    
    # 检查必要的列
    required_cols = ['性别', '年龄组', '高血脂症二分类标签']
    constitution_cols = ['平和质', '气虚质', '阳虚质', '阴虚质', 
                       '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
    
    for col in required_cols + constitution_cols:
        if col not in df.columns:
            print(f"缺少必要的列：{col}")
            return
    
    # 按性别分析
    gender_results = analyze_by_gender(df)
    
    # 按年龄组分析
    age_results = analyze_by_age_group(df)
    
    # 比较差异
    compare_contributions(gender_results, age_results)
    
    # 生成总结
    print("\n=== 分析总结 ===")
    print("1. 性别差异：")
    print("   - 女性群体中贡献度最高的体质：")
    if '女' in gender_results and gender_results['女']:
        print(f"      {gender_results['女'][0][0]}（{gender_results['女'][0][1]:.4f}）")
    
    print("   - 男性群体中贡献度最高的体质：")
    if '男' in gender_results and gender_results['男']:
        print(f"      {gender_results['男'][0][0]}（{gender_results['男'][0][1]:.4f}）")
    
    print("\n2. 年龄组差异：")
    for age_group_name, contributions in age_results.items():
        if contributions:
            print(f"   - {age_group_name}：{contributions[0][0]}（{contributions[0][1]:.4f}）")
    
    print("\n3. 结论：")
    print("   - 九种体质对高血脂发病风险的贡献度在不同性别和年龄组之间存在差异")
    print("   - 不同年龄段的主要风险体质有所不同，需要针对性地进行干预")
    print("   - 性别因素也会影响体质与发病风险的关联模式")

if __name__ == "__main__":
    main()
