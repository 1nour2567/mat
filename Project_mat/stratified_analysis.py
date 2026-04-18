import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]  # 优先黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号 - 变方块

# 加载数据
df = pd.read_excel('data/raw/附件1：样例数据.xlsx')

# 加载常量
from config.constants import AGE_CONSTRAINTS, STRATIFICATION

print(f"年龄约束：{AGE_CONSTRAINTS}")
print(f"分层变量：{STRATIFICATION}")

# 目标变量
target = '高血脂症二分类标签'
constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']

# 1. 按年龄分层分析
print("\n=== 按年龄分层分析 ===")
age_groups = AGE_CONSTRAINTS['age_groups']

age_stratified_results = []
for age_group, age_range in age_groups.items():
    subset = df[df['年龄组'] == age_group]
    if len(subset) > 0:
        # 计算各体质标签的分布
        contingency_table = pd.crosstab(subset['体质标签'], subset[target])
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            prevalence = subset[target].mean()
            age_stratified_results.append({
                'age_group': age_range,
                'sample_size': len(subset),
                'prevalence': prevalence,
                'chi2': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
            print(f"年龄组 {age_range}：")
            print(f"  样本数: {len(subset)}")
            print(f"  高血脂患病率: {prevalence:.4f}")
            print(f"  卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
            print(f"  {'显著关联' if p_value < 0.05 else '无显著关联'}")

# 2. 按性别分层分析
print("\n=== 按性别分层分析 ===")
gender_stratified_results = []
for gender in [0, 1]:  # 假设0为女性，1为男性
    subset = df[df['性别'] == gender]
    if len(subset) > 0:
        contingency_table = pd.crosstab(subset['体质标签'], subset[target])
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            prevalence = subset[target].mean()
            gender_stratified_results.append({
                'gender': '男' if gender == 1 else '女',
                'sample_size': len(subset),
                'prevalence': prevalence,
                'chi2': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
            print(f"性别 {'男' if gender == 1 else '女'}：")
            print(f"  样本数: {len(subset)}")
            print(f"  高血脂患病率: {prevalence:.4f}")
            print(f"  卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
            print(f"  {'显著关联' if p_value < 0.05 else '无显著关联'}")

# 3. 按活动能力分层分析
print("\n=== 按活动能力分层分析 ===")
activity_levels = STRATIFICATION['activity_level']

# 计算活动总分
df['活动总分'] = df['ADL总分'] + df['IADL总分']

activity_stratified_results = []
# 低活动能力
subset_low = df[df['活动总分'] < activity_levels['low']]
if len(subset_low) > 0:
    contingency_table = pd.crosstab(subset_low['体质标签'], subset_low[target])
    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        prevalence = subset_low[target].mean()
        activity_stratified_results.append({
            'activity_level': '低活动能力',
            'sample_size': len(subset_low),
            'prevalence': prevalence,
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
        print("低活动能力 (<40)：")
        print(f"  样本数: {len(subset_low)}")
        print(f"  高血脂患病率: {prevalence:.4f}")
        print(f"  卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
        print(f"  {'显著关联' if p_value < 0.05 else '无显著关联'}")

# 中等活动能力
subset_medium = df[(df['活动总分'] >= activity_levels['low']) & (df['活动总分'] <= activity_levels['medium'])]
if len(subset_medium) > 0:
    contingency_table = pd.crosstab(subset_medium['体质标签'], subset_medium[target])
    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        prevalence = subset_medium[target].mean()
        activity_stratified_results.append({
            'activity_level': '中等活动能力',
            'sample_size': len(subset_medium),
            'prevalence': prevalence,
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
        print("中等活动能力 (40-59)：")
        print(f"  样本数: {len(subset_medium)}")
        print(f"  高血脂患病率: {prevalence:.4f}")
        print(f"  卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
        print(f"  {'显著关联' if p_value < 0.05 else '无显著关联'}")

# 高活动能力
subset_high = df[df['活动总分'] >= activity_levels['high']]
if len(subset_high) > 0:
    contingency_table = pd.crosstab(subset_high['体质标签'], subset_high[target])
    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        prevalence = subset_high[target].mean()
        activity_stratified_results.append({
            'activity_level': '高活动能力',
            'sample_size': len(subset_high),
            'prevalence': prevalence,
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
        print("高活动能力 (≥60)：")
        print(f"  样本数: {len(subset_high)}")
        print(f"  高血脂患病率: {prevalence:.4f}")
        print(f"  卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
        print(f"  {'显著关联' if p_value < 0.05 else '无显著关联'}")

# 4. 按痰湿积分分层分析
print("\n=== 按痰湿积分分层分析 ===")
phlegm_levels = STRATIFICATION['phlegm_dampness']

phlegm_stratified_results = []
# 低痰湿积分
subset_low = df[df['痰湿质'] <= phlegm_levels['low']]
if len(subset_low) > 0:
    contingency_table = pd.crosstab(subset_low['体质标签'], subset_low[target])
    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        prevalence = subset_low[target].mean()
        phlegm_stratified_results.append({
            'phlegm_level': '低痰湿积分',
            'sample_size': len(subset_low),
            'prevalence': prevalence,
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
        print("低痰湿积分 (≤58)：")
        print(f"  样本数: {len(subset_low)}")
        print(f"  高血脂患病率: {prevalence:.4f}")
        print(f"  卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
        print(f"  {'显著关联' if p_value < 0.05 else '无显著关联'}")

# 中等痰湿积分
subset_medium = df[(df['痰湿质'] > phlegm_levels['low']) & (df['痰湿质'] <= phlegm_levels['medium'])]
if len(subset_medium) > 0:
    contingency_table = pd.crosstab(subset_medium['体质标签'], subset_medium[target])
    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        prevalence = subset_medium[target].mean()
        phlegm_stratified_results.append({
            'phlegm_level': '中等痰湿积分',
            'sample_size': len(subset_medium),
            'prevalence': prevalence,
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
        print("中等痰湿积分 (59-61)：")
        print(f"  样本数: {len(subset_medium)}")
        print(f"  高血脂患病率: {prevalence:.4f}")
        print(f"  卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
        print(f"  {'显著关联' if p_value < 0.05 else '无显著关联'}")

# 高痰湿积分
subset_high = df[df['痰湿质'] >= phlegm_levels['high']]
if len(subset_high) > 0:
    contingency_table = pd.crosstab(subset_high['体质标签'], subset_high[target])
    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        prevalence = subset_high[target].mean()
        phlegm_stratified_results.append({
            'phlegm_level': '高痰湿积分',
            'sample_size': len(subset_high),
            'prevalence': prevalence,
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
        print("高痰湿积分 (≥62)：")
        print(f"  样本数: {len(subset_high)}")
        print(f"  高血脂患病率: {prevalence:.4f}")
        print(f"  卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
        print(f"  {'显著关联' if p_value < 0.05 else '无显著关联'}")

# 生成图表

# 图1：年龄分层卡方检验结果
print("\n生成图1：年龄分层卡方检验结果")
plt.figure(figsize=(12, 8))

age_data = pd.DataFrame(age_stratified_results)
if not age_data.empty:
    # 创建子图
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 患病率
    axes[0].bar(age_data['age_group'], age_data['prevalence'], color='#E76F51')
    axes[0].set_title('各年龄组高血脂患病率')
    axes[0].set_xlabel('年龄组')
    axes[0].set_ylabel('患病率')
    axes[0].set_ylim(0, 1)
    
    # p值
    axes[1].bar(age_data['age_group'], age_data['p_value'], color='#2A9D8F')
    axes[1].axhline(y=0.05, color='red', linestyle='--', linewidth=1)
    axes[1].set_title('各年龄组卡方检验p值')
    axes[1].set_xlabel('年龄组')
    axes[1].set_ylabel('p值')
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('figure5_年龄分层卡方检验.png', dpi=300, bbox_inches='tight')
    plt.close()

# 图2：性别分层卡方检验结果
print("生成图2：性别分层卡方检验结果")
gender_data = pd.DataFrame(gender_stratified_results)
if not gender_data.empty:
    plt.figure(figsize=(10, 6))
    
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 患病率
    axes[0].bar(gender_data['gender'], gender_data['prevalence'], color='#E76F51')
    axes[0].set_title('不同性别高血脂患病率')
    axes[0].set_ylabel('患病率')
    axes[0].set_ylim(0, 1)
    
    # p值
    axes[1].bar(gender_data['gender'], gender_data['p_value'], color='#2A9D8F')
    axes[1].axhline(y=0.05, color='red', linestyle='--', linewidth=1)
    axes[1].set_title('不同性别卡方检验p值')
    axes[1].set_ylabel('p值')
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('figure6_性别分层卡方检验.png', dpi=300, bbox_inches='tight')
    plt.close()

# 图3：活动能力分层卡方检验结果
print("生成图3：活动能力分层卡方检验结果")
activity_data = pd.DataFrame(activity_stratified_results)
if not activity_data.empty:
    plt.figure(figsize=(12, 8))
    
    # 创建子图
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 患病率
    axes[0].bar(activity_data['activity_level'], activity_data['prevalence'], color='#E76F51')
    axes[0].set_title('不同活动能力高血脂患病率')
    axes[0].set_xlabel('活动能力')
    axes[0].set_ylabel('患病率')
    axes[0].set_ylim(0, 1)
    
    # p值
    axes[1].bar(activity_data['activity_level'], activity_data['p_value'], color='#2A9D8F')
    axes[1].axhline(y=0.05, color='red', linestyle='--', linewidth=1)
    axes[1].set_title('不同活动能力卡方检验p值')
    axes[1].set_xlabel('活动能力')
    axes[1].set_ylabel('p值')
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('figure7_活动能力分层卡方检验.png', dpi=300, bbox_inches='tight')
    plt.close()

# 图4：痰湿积分分层卡方检验结果
print("生成图4：痰湿积分分层卡方检验结果")
phlegm_data = pd.DataFrame(phlegm_stratified_results)
if not phlegm_data.empty:
    plt.figure(figsize=(12, 8))
    
    # 创建子图
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 患病率
    axes[0].bar(phlegm_data['phlegm_level'], phlegm_data['prevalence'], color='#E76F51')
    axes[0].set_title('不同痰湿积分高血脂患病率')
    axes[0].set_xlabel('痰湿积分水平')
    axes[0].set_ylabel('患病率')
    axes[0].set_ylim(0, 1)
    
    # p值
    axes[1].bar(phlegm_data['phlegm_level'], phlegm_data['p_value'], color='#2A9D8F')
    axes[1].axhline(y=0.05, color='red', linestyle='--', linewidth=1)
    axes[1].set_title('不同痰湿积分卡方检验p值')
    axes[1].set_xlabel('痰湿积分水平')
    axes[1].set_ylabel('p值')
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('figure8_痰湿积分分层卡方检验.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\n所有图表生成完成！")
print("已保存的图表：")
print("1. figure5_年龄分层卡方检验.png")
print("2. figure6_性别分层卡方检验.png")
print("3. figure7_活动能力分层卡方检验.png")
print("4. figure8_痰湿积分分层卡方检验.png")

# 生成综合分析报告
print("\n=== 分层分析综合报告 ===")
print("1. 年龄分层分析：")
for result in age_stratified_results:
    print(f"   {result['age_group']}: 患病率={result['prevalence']:.4f}, p值={result['p_value']:.4f}, {'显著' if result['significant'] else '不显著'}")

print("\n2. 性别分层分析：")
for result in gender_stratified_results:
    print(f"   {result['gender']}: 患病率={result['prevalence']:.4f}, p值={result['p_value']:.4f}, {'显著' if result['significant'] else '不显著'}")

print("\n3. 活动能力分层分析：")
for result in activity_stratified_results:
    print(f"   {result['activity_level']}: 患病率={result['prevalence']:.4f}, p值={result['p_value']:.4f}, {'显著' if result['significant'] else '不显著'}")

print("\n4. 痰湿积分分层分析：")
for result in phlegm_stratified_results:
    print(f"   {result['phlegm_level']}: 患病率={result['prevalence']:.4f}, p值={result['p_value']:.4f}, {'显著' if result['significant'] else '不显著'}")
