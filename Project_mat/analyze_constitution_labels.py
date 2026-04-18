import pandas as pd
import numpy as np

# 加载数据
try:
    df = pd.read_csv('/workspace/Project_mat/预处理后数据.txt', sep='\t', encoding='gbk')
except:
    df = pd.read_csv('/workspace/Project_mat/预处理后数据.txt', sep='\t', encoding='gb18030')

print("="*80)
print("体质标签与体质分数分析")
print("="*80)

# 体质名称映射
constitution_names = {
    1: '平和质',
    2: '气虚质',
    3: '阳虚质',
    4: '阴虚质',
    5: '痰湿质',
    6: '湿热质',
    7: '血瘀质',
    8: '气郁质',
    9: '特禀质'
}

constitution_list = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']

print("\n1. 体质标签分布：")
label_dist = df['体质标签'].value_counts().sort_index()
for label, count in label_dist.items():
    print(f"   {constitution_names[label]}: {count}人 ({count/len(df)*100:.1f}%)")

print("\n2. 按体质标签分组的体质分数统计：")
for label in sorted(df['体质标签'].unique()):
    name = constitution_names[label]
    subset = df[df['体质标签'] == label]
    print(f"\n   --- {name} ---")
    for const in constitution_list:
        scores = subset[const]
        print(f"     {const}: {scores.mean():.1f} ± {scores.std():.1f} (min={scores.min():.0f}, max={scores.max():.0f})")

print("\n3. 验证阈值判定规则：")
print("\n   偏颇体质判定规则（≥40分成立，30-39分倾向，<30分不成立）")
print("   平和质判定规则（平和分≥60，且其他失衡体质分数足够低）")

print("\n4. 检查每个样本的体质标签与最高分的关系：")
max_constitution = df[constitution_list].idxmax(axis=1)
label_match = df['体质名称'] == max_constitution
print(f"\n   体质标签 == 最高分体质的比例: {label_match.mean()*100:.1f}%")
print(f"   体质标签 != 最高分体质的数量: {len(df) - label_match.sum()}")

# 查看不匹配的样本
if len(df) - label_match.sum() > 0:
    print("\n   不匹配的样本示例：")
    mismatch = df[~label_match].head(5)
    for idx, row in mismatch.iterrows():
        print(f"\n     样本ID: {row['样本ID']}")
        print(f"     体质标签: {row['体质名称']} (标签值={row['体质标签']})")
        print(f"     最高分体质: {max_constitution[idx]}")
        print(f"     各体质分数:")
        for const in constitution_list:
            score = row[const]
            marker = " *" if const == row['体质名称'] else ""
            marker2 = " ↑" if const == max_constitution[idx] else ""
            print(f"       {const}: {score:.0f}{marker}{marker2}")

print("\n" + "="*80)
print("5. 基于阈值规则重新判定体质：")
print("="*80)

def determine_constitution(row):
    """基于阈值规则判定体质"""
    # 偏颇体质检查
    biased_constitutions = ['气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
    established = []
    tending = []
    
    for const in biased_constitutions:
        score = row[const]
        if score >= 40:
            established.append(const)
        elif 30 <= score < 40:
            tending.append(const)
    
    # 检查平和质
    pinghe_score = row['平和质']
    other_scores = [row[const] for const in biased_constitutions]
    max_other_score = max(other_scores)
    
    pinghe_qualified = (pinghe_score >= 60) and (max_other_score < 30)
    
    return {
        '样本ID': row['样本ID'],
        '原标签': row['体质名称'],
        '平和质分数': pinghe_score,
        '其他体质最高分': max_other_score,
        '成立的偏颇体质': established,
        '倾向的偏颇体质': tending,
        '平和质合格': pinghe_qualified
    }

# 对部分样本进行判定
results = []
for idx, row in df.head(20).iterrows():
    results.append(determine_constitution(row))

result_df = pd.DataFrame(results)
print("\n阈值判定结果（前20个样本）：")
print(result_df.to_string(index=False))

print("\n" + "="*80)
