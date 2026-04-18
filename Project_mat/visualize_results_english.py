import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# 设置字体为英文
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号 - 变方块

# 加载数据
df = pd.read_excel('data/raw/附件1：样例数据.xlsx')

# 候选特征池构建
basic_features = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质', '体质标签',
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

derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL', '血脂异常项数', '尿酸异常标志']

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
        feature_types.append('Basic Feature')
    elif feature in derived_features:
        feature_types.append('Derived Feature')
    elif feature in cross_features:
        feature_types.append('Integrated Feature')
    else:
        feature_types.append('Other')

# === 熵权法计算权重 ===
print("=== Entropy Weight Method (EWM) ===")

# 构建原始矩阵 X = (x_ij)_{n×m}
n = len(all_features)  # 47 features
m = 3  # 3 dimensions
X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings))

# 第一步：矩阵标准化（极差标准化）
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

# 第二步：计算特征在各维度下的比重 p_ij
P = np.zeros_like(X_norm)
for j in range(m):
    col_sum = np.sum(X_norm[:, j])
    if col_sum == 0:
        P[:, j] = 0
    else:
        P[:, j] = X_norm[:, j] / col_sum

# 第三步：计算各维度的信息熵 e_j
e = np.zeros(m)
k = 1 / np.log(n)  # 保证 0 ≤ e_j ≤ 1
for j in range(m):
    # 计算 p_ij * ln(p_ij)，注意处理 p_ij=0 的情况
    entropy = 0
    for i in range(n):
        if P[i, j] > 0:
            entropy += P[i, j] * np.log(P[i, j])
    e[j] = -k * entropy

# 第四步：计算信息冗余度（差异性系数）d_j
d = 1 - e

# 第五步：计算最终权重 w_j
w = d / np.sum(d)

print(f"Dimension Weights:")
print(f"Spearman R (Phlegm): {w[0]:.4f}")
print(f"Mutual Info MI (Risk): {w[1]:.4f}")
print(f"PLS Loading (Joint): {w[2]:.4f}")
print(f"Total Weight: {np.sum(w):.4f}")

# 计算综合评分
total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]

# 构建数据框
feature_scores = pd.DataFrame({
    'feature': all_features,
    'feature_type': feature_types,
    'spearman_score': spearman_scores,
    'mutual_info_score': mutual_info_scores,
    'pls_loading': pls_loadings,
    'spearman_normalized': X_norm[:, 0],
    'mutual_info_normalized': X_norm[:, 1],
    'pls_normalized': X_norm[:, 2],
    'total_score': total_scores
})

feature_scores = feature_scores.sort_values('total_score', ascending=False)

print("\nTop 10 Key Indicators:")
print(feature_scores.head(10)[['feature', 'total_score']])

# === 生成图表 ===

# 图 1：双目标联合筛选综合评分排序图（使用熵权法权重）
print("\nGenerating Figure 1: Comprehensive Score Ranking (EWM)")
plt.figure(figsize=(12, 10))

# 取前15个指标
top15 = feature_scores.head(15)

# 颜色映射
color_map = {
    'Basic Feature': '#2A9D8F',
    'Derived Feature': '#E9C46A',
    'Integrated Feature': '#F4A261'
}

colors = [color_map[t] for t in top15['feature_type']]

# 水平条形图
y_pos = np.arange(len(top15))
bars = plt.barh(y_pos, top15['total_score'], color=colors, height=0.7)

# 反转Y轴，使评分最高的在最上面
plt.gca().invert_yaxis()

# 设置Y轴标签
plt.yticks(y_pos, top15['feature'], fontsize=11)

# 设置X轴
plt.xlabel('Comprehensive Score (F_score)', fontsize=12)
plt.xlim(0, 1)

# 添加数值标注
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', va='center', fontsize=9)

# 添加标题和图例
plt.title('Figure 1: Comprehensive Score Ranking (EWM)\nF = {:.4f}×R\' + {:.4f}×M\' + {:.4f}×L\''.format(w[0], w[1], w[2]), 
          fontsize=14, pad=20)

# 创建图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=color_map['Basic Feature'], label='Basic Feature'),
    Patch(facecolor=color_map['Derived Feature'], label='Derived Feature'),
    Patch(facecolor=color_map['Integrated Feature'], label='Integrated Feature')
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=10)

# 添加网格线
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_comprehensive_ranking_ewm.png', dpi=300, bbox_inches='tight')
plt.close()

# 图 2：三维评分分解图（使用熵权法权重）
print("Generating Figure 2: Three-Dimensional Score Decomposition (EWM)")
plt.figure(figsize=(14, 8))

top10 = feature_scores.head(10)

# 设置柱状图宽度
bar_width = 0.25
x = np.arange(len(top10))

# 绘制三组柱状图
bars1 = plt.bar(x - bar_width, top10['spearman_normalized'], width=bar_width, 
                label=f'Spearman R (Phlegm) w={w[0]:.3f}', color='#264653', alpha=0.8)
bars2 = plt.bar(x, top10['mutual_info_normalized'], width=bar_width, 
                label=f'Mutual Info MI (Risk) w={w[1]:.3f}', color='#2A9D8F', alpha=0.8)
bars3 = plt.bar(x + bar_width, top10['pls_normalized'], width=bar_width, 
                label=f'PLS Loading (Joint) w={w[2]:.3f}', color='#8AC926', alpha=0.8)

# 设置标签和标题
plt.xlabel('Key Indicators', fontsize=12)
plt.ylabel('Normalized Score (0~1)', fontsize=12)
plt.title('Figure 2: Three-Dimensional Score Decomposition (EWM)', fontsize=14, pad=20)
plt.xticks(x, top10['feature'], rotation=45, ha='right')
plt.legend(loc='upper right')
plt.ylim(0, 1.1)

# 添加数值标注
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=8)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.tight_layout()
plt.savefig('figure2_3d_decomposition_ewm.png', dpi=300, bbox_inches='tight')
plt.close()

# 图 3：九种体质风险贡献双面板图
print("Generating Figure 3: Nine Constitutions Risk Contribution")
constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
constitution_types_eng = ['Balanced', 'Qi Deficiency', 'Yang Deficiency', 'Yin Deficiency', 
                         'Phlegm-Dampness', 'Damp-Heat', 'Blood Stasis', 'Qi Stagnation', 'Special Endowment']

# 计算相对风险
total_prevalence = df[target].mean()
relative_risks = []
for i in range(1, 10):
    subset = df[df['体质标签'] == i]
    if len(subset) > 0:
        prevalence = subset[target].mean()
        rr = prevalence / total_prevalence
    else:
        rr = 1.0
    relative_risks.append(rr)

# 卡方检验
contingency_table = pd.crosstab(df['体质标签'], df[target])
chi2, p_value, _, _ = chi2_contingency(contingency_table)

# Logistic回归分析
control_vars = ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI',
                '活动量表总分（ADL总分+IADL总分）', '年龄组', '性别', '吸烟史', '饮酒史']
X_logistic = df[constitution_types + control_vars]
y_logistic = df[target]

scaler_logistic = StandardScaler()
X_logistic_scaled = scaler_logistic.fit_transform(X_logistic)

logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_logistic_scaled, y_logistic)

coefficients = logistic_model.coef_[0]
feature_names = constitution_types + control_vars

# 提取体质相关的系数
constitution_coefficients = []
for feature in constitution_types:
    idx = feature_names.index(feature)
    constitution_coefficients.append(coefficients[idx])

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 左面板：相对风险
y_pos_const = np.arange(len(constitution_types))
# 按相对风险降序排序
rr_sorted_idx = np.argsort(relative_risks)[::-1]
rr_sorted = [relative_risks[i] for i in rr_sorted_idx]
constitutions_sorted_rr = [constitution_types_eng[i] for i in rr_sorted_idx]

bars_rr = ax1.barh(y_pos_const, rr_sorted, color='#E76F51', height=0.6)
ax1.invert_yaxis()
ax1.set_yticks(y_pos_const)
ax1.set_yticklabels(constitutions_sorted_rr, fontsize=11)
ax1.set_xlabel('Relative Risk (RR)', fontsize=12)
ax1.set_title(f'Left Panel: Relative Risk Analysis\nOverall Prevalence: {total_prevalence:.3f}, Chi-square p = {p_value:.4f}', 
              fontsize=12, pad=15)
ax1.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5, label='Overall Risk Baseline')
ax1.legend(loc='lower right')

# 添加数值标注
for i, bar in enumerate(bars_rr):
    width = bar.get_width()
    ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', va='center', fontsize=9)

# 右面板：标准化回归系数
# 使用与左面板相同的体质顺序
constitutions_sorted = constitutions_sorted_rr
coef_sorted = []
for i in rr_sorted_idx:
    coef_sorted.append(constitution_coefficients[i])

# 颜色映射
def get_coef_color(coef):
    if coef > 0.05:
        return '#E76F51'  # 红色，风险因子
    elif coef < -0.05:
        return '#2A9D8F'  # 蓝色，保护因子
    else:
        return '#8D99AE'  # 灰色，中性

colors_coef = [get_coef_color(c) for c in coef_sorted]

bars_coef = ax2.barh(y_pos_const, coef_sorted, color=colors_coef, height=0.6)
ax2.invert_yaxis()
ax2.set_yticks(y_pos_const)
ax2.set_yticklabels(constitutions_sorted, fontsize=11)
ax2.set_xlabel('Standardized Regression Coefficient β', fontsize=12)
ax2.set_title('Right Panel: Logistic Regression Net Contribution\nControlled Variables: Lipids, Activity, Demographics', 
              fontsize=12, pad=15)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)

# 添加数值标注
for i, bar in enumerate(bars_coef):
    width = bar.get_width()
    if width >= 0:
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', va='center', fontsize=9)
    else:
        ax2.text(width - 0.05, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', va='center', fontsize=9)

# 调整布局
plt.tight_layout()
plt.savefig('figure3_constitution_risk_contribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 图 4：关键指标与双目标的相关性热力图
print("Generating Figure 4: Correlation Heatmap (EWM)")
top10_features = feature_scores.head(10)['feature'].tolist()
heatmap_features = top10_features + ['痰湿质', '高血脂症二分类标签']

# 计算Spearman相关系数
corr_matrix = df[heatmap_features].corr(method='spearman')

# 创建热力图
plt.figure(figsize=(14, 12))

# 绘制热力图
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, square=True, linewidths=1, cbar_kws={'shrink': 0.8})

# 添加标题
plt.title('Figure 4: Correlation Heatmap (EWM)\n(Spearman Correlation Coefficient)', 
          fontsize=14, pad=20)

# 添加黑色粗边框框出最后两列
ax = plt.gca()
# 获取热力图的位置
n_rows, n_cols = corr_matrix.shape
# 框出最后两列（痰湿质和高血脂标签）
for i in range(n_rows):
    for j in [n_cols - 2, n_cols - 1]:
        rect = plt.Rectangle((j, i), 1, 1, fill=False, color='black', linewidth=2)
        ax.add_patch(rect)

plt.tight_layout()
plt.savefig('figure4_correlation_heatmap_ewm.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAll figures generated successfully!")
print("Saved figures:")
print("1. figure1_comprehensive_ranking_ewm.png")
print("2. figure2_3d_decomposition_ewm.png")
print("3. figure3_constitution_risk_contribution.png")
print("4. figure4_correlation_heatmap_ewm.png")
