#!/usr/bin/env python3
# 创建综合评分瀑布图

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]  # 优先黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号 - 变方块

# 加载数据
df = pd.read_excel('data/raw/附件1：样例数据.xlsx')

# 候选特征池构建
basic_features = ['平和质', '气虚质', '阳虚质', '阴虚质', '湿热质', '血瘀质', '气郁质', '特禀质',
                 'TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI',
                 'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）',
                 'ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡',
                 'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药',
                 '年龄组', '性别', '吸烟史', '饮酒史']

# 派生特征层
df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']

def count_dyslipidemia(row):
    count = 0
    if row['TC（总胆固醇）'] > 5.2: count += 1
    if row['TG（甘油三酯）'] > 1.7: count += 1
    if row['LDL-C（低密度脂蛋白）'] > 3.4: count += 1
    if row['HDL-C（高密度脂蛋白）'] < 1.0: count += 1
    return count
df['血脂异常项数'] = df.apply(count_dyslipidemia, axis=1)

def uric_acid_abnormal(row):
    if row['性别'] == 1:
        return 1 if row['血尿酸'] > 420 else 0
    else:
        return 1 if row['血尿酸'] > 360 else 0
df['尿酸异常标志'] = df.apply(uric_acid_abnormal, axis=1)

derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL', '尿酸异常标志']

# 中西医交叉特征层
df['痰湿质得分×BMI'] = df['痰湿质'] * df['BMI']
df['痰湿质得分×TG'] = df['痰湿质'] * df['TG（甘油三酯）']
df['痰湿质得分×AIP'] = df['痰湿质'] * df['AIP']
df['痰湿质得分×LDL-C'] = df['痰湿质'] * df['LDL-C（低密度脂蛋白）']
df['痰湿质得分/HDL-C'] = df['痰湿质'] / df['HDL-C（高密度脂蛋白）']
df['气虚质得分×TC'] = df['气虚质'] * df['TC（总胆固醇）']

cross_features = ['痰湿质得分×BMI', '痰湿质得分×TG', '痰湿质得分×AIP', '痰湿质得分×LDL-C', '痰湿质得分/HDL-C', '气虚质得分×TC']

all_features = basic_features + derived_features + cross_features

# 目标变量
target = '高血脂症二分类标签'
target_phlegm = '痰湿质'

# 计算评分
X = df[all_features]
y_phlegm = df[target_phlegm]
y_risk = df[target]

# 1. Spearman相关系数
spearman_scores = []
for feature in all_features:
    corr, _ = spearmanr(df[feature], y_phlegm)
    spearman_scores.append(abs(corr))
spearman_scores = np.array(spearman_scores)

# 2. 互信息
mutual_info_scores = mutual_info_classif(X, y_risk, random_state=42)

# 3. PLS联合结构载荷
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_combined = np.column_stack((y_phlegm, y_risk))
pls = PLSRegression(n_components=2, scale=False)
pls.fit(X_scaled, y_combined)

pls_loadings = np.zeros(len(all_features))
for i in range(len(all_features)):
    corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
    corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
    pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2

# 准备特征类型信息
feature_types = []
for feature in all_features:
    if feature in basic_features:
        if feature in ['平和质', '气虚质', '阳虚质', '阴虚质', '湿热质', '血瘀质', '气郁质', '特禀质']:
            feature_types.append('基础中医')
        elif feature in ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI']:
            feature_types.append('基础西医')
        elif feature in ['ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）',
                        'ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡',
                        'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药']:
            feature_types.append('活动量表')
        else:
            feature_types.append('基础特征')
    elif feature in derived_features:
        feature_types.append('派生特征')
    elif feature in cross_features:
        feature_types.append('交叉特征')
    else:
        feature_types.append('其他')

# 熵权法计算权重
n = len(all_features)  # 特征数量
m = 3  # 3个维度
X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings))

# 矩阵标准化（极差标准化）
def normalize_matrix(X):
    X_norm = np.zeros_like(X)
    for j in range(X.shape[1]):
        min_val = np.min(X[:, j])
        max_val = np.max(X[:, j])
        if max_val - min_val == 0:
            X_norm[:, j] = 0
        else:
            X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
    # 添加极小值避免后续对数计算出现 ln(0)
    X_norm = X_norm + 0.0001
    return X_norm

X_norm = normalize_matrix(X_matrix)

# 计算特征在各维度下的比重 p_ij
P = np.zeros_like(X_norm)
for j in range(m):
    col_sum = np.sum(X_norm[:, j])
    if col_sum == 0:
        P[:, j] = 0
    else:
        P[:, j] = X_norm[:, j] / col_sum

# 计算各维度的信息熵 e_j
e = np.zeros(m)
k = 1 / np.log(n)  # 保证 0 ≤ e_j ≤ 1
for j in range(m):
    entropy = 0
    for i in range(n):
        if P[i, j] > 0:
            entropy += P[i, j] * np.log(P[i, j])
    e[j] = -k * entropy

# 计算信息冗余度（差异性系数）d_j
d = 1 - e

# 计算最终权重 w_j
w = d / np.sum(d)

# 计算各维度的贡献
weighted_contributions = X_norm * w

# 计算综合评分
total_scores = np.sum(weighted_contributions, axis=1)

# 构建数据框
feature_scores = pd.DataFrame({
    'feature': all_features,
    'feature_type': feature_types,
    'spearman_score': spearman_scores,
    'mutual_info_score': mutual_info_scores,
    'pls_loading': pls_loadings,
    'spearman_contribution': weighted_contributions[:, 0],
    'mutual_info_contribution': weighted_contributions[:, 1],
    'pls_contribution': weighted_contributions[:, 2],
    'total_score': total_scores
})

# 按综合评分降序排序
feature_scores = feature_scores.sort_values('total_score', ascending=False)

# 取前15个指标
top15 = feature_scores.head(15).copy()

# 重新排序，使综合评分降序排列
top15 = top15.sort_values('total_score', ascending=False)

# 准备数据用于堆叠柱状图
features = top15['feature'].tolist()
feature_types = top15['feature_type'].tolist()
spearman_contrib = top15['spearman_contribution'].tolist()
mutual_info_contrib = top15['mutual_info_contribution'].tolist()
pls_contrib = top15['pls_contribution'].tolist()
total_scores = top15['total_score'].tolist()

# 生成综合评分瀑布图
plt.figure(figsize=(16, 10))

# 设置柱状图宽度
bar_width = 0.6
x = np.arange(len(features))

# 绘制堆叠柱状图
bars1 = plt.bar(x, spearman_contrib, width=bar_width, label='Spearman贡献', color='#264653', alpha=0.8)
bars2 = plt.bar(x, mutual_info_contrib, width=bar_width, label='互信息贡献', color='#2A9D8F', alpha=0.8, bottom=spearman_contrib)
bars3 = plt.bar(x, pls_contrib, width=bar_width, label='PLS贡献', color='#8AC926', alpha=0.8, bottom=np.array(spearman_contrib) + np.array(mutual_info_contrib))

# 设置横轴标签（指标名称 + 类型）
x_labels = []
for i in range(len(features)):
    x_labels.append(f"{features[i]}\n({feature_types[i]})")

plt.xticks(x, x_labels, rotation=45, ha='right', fontsize=10)

# 设置纵轴
plt.ylabel('综合评分', fontsize=12)
plt.ylim(0, max(total_scores) * 1.2)

# 添加标题
plt.title('综合评分瀑布图 - Top 15关键指标', fontsize=14, pad=20)

# 添加图例
plt.legend(loc='upper right', fontsize=10)

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.3)

# 添加数值标注
def add_labels(bars, values):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{value:.2f}', ha='center', va='bottom', fontsize=8)

# 标注总评分
for i, score in enumerate(total_scores):
    plt.text(x[i], score + 0.01, f'{score:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 关键标注
# 1. 在痰湿质得分柱子上标注"单一目标高分，双目标失效"
if '痰湿质' in top15['feature'].tolist():
    idx = top15[top15['feature'] == '痰湿质'].index[0]
    pos = top15.index.get_loc(idx)
    plt.annotate('单一目标高分，双目标失效',
                 xy=(x[pos], total_scores[pos]),
                 xytext=(x[pos], total_scores[pos] + 0.05),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=9, color='red', fontweight='bold')

# 2. 在交叉特征柱子上标注"中西医协同效应区"
for i, feature_type in enumerate(feature_types):
    if feature_type == '交叉特征':
        plt.annotate('中西医协同效应区',
                     xy=(x[i], total_scores[i]),
                     xytext=(x[i], total_scores[i] + 0.05),
                     arrowprops=dict(facecolor='blue', shrink=0.05),
                     fontsize=9, color='blue', fontweight='bold')

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('figure5_综合评分瀑布图.png', dpi=300, bbox_inches='tight')
plt.close()

print("综合评分瀑布图已生成并保存为 figure5_综合评分瀑布图.png")
print("\nTop 15关键指标及其综合评分：")
for i, row in top15.iterrows():
    print(f"{i+1}. {row['feature']} ({row['feature_type']}): {row['total_score']:.3f}")
