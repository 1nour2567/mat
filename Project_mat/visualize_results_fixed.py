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
from matplotlib import font_manager
import os

# 中文字体映射 - 用于英文标签替代方案
feature_name_map = {
    '痰湿质': 'Phlegm-Dampness',
    '痰湿质得分×TG': 'Phlegm × TG',
    '痰湿质得分×BMI': 'Phlegm × BMI',
    '痰湿质得分/HDL-C': 'Phlegm / HDL-C',
    '痰湿质得分×LDL-C': 'Phlegm × LDL-C',
    '血脂异常项数': 'Lipid Abnormal Count',
    '痰湿质得分×AIP': 'Phlegm × AIP',
    'TG/HDL': 'TG/HDL',
    'TG（甘油三酯）': 'TG (Triglycerides)',
    'AIP': 'AIP',
    'non-HDL-C': 'non-HDL-C',
    'TC/HDL': 'TC/HDL',
    'LDL/HDL': 'LDL/HDL',
    'TC（总胆固醇）': 'TC (Total Cholesterol)',
    '血尿酸': 'Uric Acid',
    'HDL-C（高密度脂蛋白）': 'HDL-C',
    '尿酸异常标志': 'Uric Acid Abnormal',
    '平和质': 'Balanced',
    'LDL-C（低密度脂蛋白）': 'LDL-C',
    '体质标签': 'Constitution Label',
    '气虚质': 'Qi Deficiency',
    '阳虚质': 'Yang Deficiency',
    '阴虚质': 'Yin Deficiency',
    '湿热质': 'Damp-Heat',
    '血瘀质': 'Blood Stasis',
    '气郁质': 'Qi Stagnation',
    '特禀质': 'Special',
    '气虚质得分×TC': 'Qi Deficiency × TC',
    'ADL总分': 'ADL Total',
    'IADL总分': 'IADL Total',
    '活动量表总分（ADL总分+IADL总分）': 'Activity Total',
    'BMI': 'BMI',
    '空腹血糖': 'Fasting Glucose',
    '年龄组': 'Age Group',
    '性别': 'Gender',
    '吸烟史': 'Smoking',
    '饮酒史': 'Drinking',
    'ADL用厕': 'ADL Toilet',
    'ADL吃饭': 'ADL Eating',
    'ADL步行': 'ADL Walking',
    'ADL穿衣': 'ADL Dressing',
    'ADL洗澡': 'ADL Bathing',
    'IADL购物': 'IADL Shopping',
    'IADL做饭': 'IADL Cooking',
    'IADL理财': 'IADL Finance',
    'IADL交通': 'IADL Transport',
    'IADL服药': 'IADL Medicine',
    '高血脂症二分类标签': 'Hyperlipidemia'
}

constitution_name_map = {
    '平和质': 'Balanced',
    '气虚质': 'Qi Deficiency',
    '阳虚质': 'Yang Deficiency',
    '阴虚质': 'Yin Deficiency',
    '痰湿质': 'Phlegm-Dampness',
    '湿热质': 'Damp-Heat',
    '血瘀质': 'Blood Stasis',
    '气郁质': 'Qi Stagnation',
    '特禀质': 'Special'
}

def translate_feature_name(name):
    """将中文特征名转换为英文，避免乱码问题"""
    return feature_name_map.get(name, name)

def translate_constitution_name(name):
    """将中文体质名转换为英文，避免乱码问题"""
    return constitution_name_map.get(name, name)

# 设置字体 - 健壮的中文字体设置
def setup_chinese_font():
    """自动检测并设置可用的中文字体"""
    # 常见的中文字体列表
    chinese_fonts = [
        'SimHei',
        'WenQuanYi Micro Hei',
        'Heiti TC',
        'Microsoft YaHei',
        'STHeiti',
        'STHeiti TC',
        'Noto Sans CJK SC',
        'Noto Sans CJK TC',
        'Source Han Sans SC',
        'Source Han Sans TC',
        'PingFang SC',
        'PingFang TC'
    ]
    
    # 检查系统中可用的字体
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    
    # 打印一些可用的字体信息（用于调试）
    print("系统可用的字体（前20个）：")
    for i, font in enumerate(available_fonts[:20]):
        print(f"  {i+1}. {font}")
    
    # 查找第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            print(f"\n找到可用的中文字体: {font}")
            break
    
    # 如果没有找到特定的中文字体，尝试使用系统默认字体
    if selected_font is None:
        print("\n未找到指定的中文字体，尝试使用系统默认字体")
        # 使用font_manager中的字体
        for f in font_manager.fontManager.ttflist:
            if any(keyword in f.name.lower() for keyword in ['chinese', 'cjk', 'jp', 'cn']):
                selected_font = f.name
                print(f"使用字体: {selected_font}")
                break
    
    # 设置matplotlib参数
    use_chinese = False
    if selected_font:
        plt.rcParams["font.family"] = selected_font
        use_chinese = True
    else:
        print("\n未找到中文字体，将使用英文标签避免乱码")
        plt.rcParams["font.family"] = 'DejaVu Sans'
    
    # 解决负号显示问题
    plt.rcParams["axes.unicode_minus"] = False
    
    return use_chinese

# 调用字体设置函数
use_chinese_fonts = setup_chinese_font()

# 加载数据
df = pd.read_excel('data/raw/附件1：样例数据.xlsx')

# 候选特征池构建（重新构建，确保数据完整）
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
    # 假设参考值：TC>5.2，TG>1.7，LDL-C>3.4，HDL-C<1.0
    if row['TC（总胆固醇）'] > 5.2: count += 1
    if row['TG（甘油三酯）'] > 1.7: count += 1
    if row['LDL-C（低密度脂蛋白）'] > 3.4: count += 1
    if row['HDL-C（高密度脂蛋白）'] < 1.0: count += 1
    return count
df['血脂异常项数'] = df.apply(count_dyslipidemia, axis=1)

def uric_acid_abnormal(row):
    if row['性别'] == 1:  # 男性
        return 1 if row['血尿酸'] > 420 else 0
    else:  # 女性
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

# 4. 综合评分
def normalize(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score == 0:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)

spearman_normalized = normalize(spearman_scores)
mutual_info_normalized = normalize(mutual_info_scores)
pls_normalized = normalize(pls_loadings)

# 基于数据驱动的熵权法：使用变异系数确定权重
print("=== 基于数据驱动的熵权法计算权重 ===")

# 计算三个维度的归一化得分矩阵
dimensions = np.column_stack((spearman_normalized, mutual_info_normalized, pls_normalized))

# 计算每个维度的变异系数（CV = 标准差 / 均值）
cv_values = []
for i in range(dimensions.shape[1]):
    dim_data = dimensions[:, i]
    mean_val = np.mean(dim_data)
    std_val = np.std(dim_data, ddof=1)
    cv = std_val / mean_val if mean_val != 0 else 0
    cv_values.append(cv)

# 归一化变异系数得到权重
cv_array = np.array(cv_values)
weights = cv_array / cv_array.sum()

print(f"各维度变异系数:")
print(f"  Spearman R' (痰湿): {cv_values[0]:.4f}")
print(f"  互信息 M' (风险): {cv_values[1]:.4f}")
print(f"  PLS载荷 L' (联合): {cv_values[2]:.4f}")
print(f"\n基于数据驱动的权重:")
print(f"  w1 (Spearman): {weights[0]:.4f}")
print(f"  w2 (互信息): {weights[1]:.4f}")
print(f"  w3 (PLS): {weights[2]:.4f}")

# 计算综合评分
total_scores = weights[0] * spearman_normalized + weights[1] * mutual_info_normalized + weights[2] * pls_normalized

# 准备特征类型信息
feature_types = []
for feature in all_features:
    if feature in basic_features:
        feature_types.append('基础特征')
    elif feature in derived_features:
        feature_types.append('派生特征')
    elif feature in cross_features:
        feature_types.append('中西医交叉特征')
    else:
        feature_types.append('其他')

# 构建数据框
feature_scores = pd.DataFrame({
    'feature': all_features,
    'feature_type': feature_types,
    'spearman_score': spearman_scores,
    'mutual_info_score': mutual_info_scores,
    'pls_loading': pls_loadings,
    'spearman_normalized': spearman_normalized,
    'mutual_info_normalized': mutual_info_normalized,
    'pls_normalized': pls_normalized,
    'total_score': total_scores
})

feature_scores = feature_scores.sort_values('total_score', ascending=False)

# === 图 1：双目标联合筛选综合评分排序图 ===
print("\n生成图 1：双目标联合筛选综合评分排序图")
plt.figure(figsize=(12, 10))

# 取前15个指标
top15 = feature_scores.head(15)

# 颜色映射
color_map = {
    '基础特征': '#2A9D8F',
    '派生特征': '#E9C46A',
    '中西医交叉特征': '#F4A261'
}

colors = [color_map[t] for t in top15['feature_type']]

# 准备标签
if use_chinese_fonts:
    y_labels = top15['feature']
    weight_formula = f'F = {weights[0]:.2f}×R\' + {weights[1]:.2f}×M\' + {weights[2]:.2f}×L\''
    title = '图 1：双目标联合筛选综合评分排序图\n' + weight_formula
    x_label = '综合评分 F_score'
    legend_labels = ['基础特征', '派生特征', '中西医交叉特征']
else:
    y_labels = [translate_feature_name(f) for f in top15['feature']]
    weight_formula = f'F = {weights[0]:.2f}×R\' + {weights[1]:.2f}×M\' + {weights[2]:.2f}×L\''
    title = 'Figure 1: Dual-Objective Screening Score\n' + weight_formula
    x_label = 'Combined Score F_score'
    legend_labels = ['Basic Features', 'Derived Features', 'TCM-WM Cross Features']

# 水平条形图
y_pos = np.arange(len(top15))
bars = plt.barh(y_pos, top15['total_score'], color=colors, height=0.7)

# 反转Y轴，使评分最高的在最上面
plt.gca().invert_yaxis()

# 设置Y轴标签
plt.yticks(y_pos, y_labels, fontsize=11)

# 设置X轴
plt.xlabel(x_label, fontsize=12)
plt.xlim(0, 1)

# 添加数值标注
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', va='center', fontsize=9)

# 添加标题和图例
plt.title(title, fontsize=14, pad=20)

# 创建图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=color_map['基础特征'], label=legend_labels[0]),
    Patch(facecolor=color_map['派生特征'], label=legend_labels[1]),
    Patch(facecolor=color_map['中西医交叉特征'], label=legend_labels[2])
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=10)

# 添加网格线
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_综合评分排序.png', dpi=300, bbox_inches='tight')
plt.close()

# === 图 2：三维评分分解图（分组柱状图）===
print("生成图 2：三维评分分解图")
plt.figure(figsize=(14, 8))

top10 = feature_scores.head(10)

# 设置柱状图宽度
bar_width = 0.25
x = np.arange(len(top10))

# 准备标签
if use_chinese_fonts:
    x_labels = top10['feature']
    title = '图 2：三维评分分解图'
    y_label = '归一化得分 (0~1)'
    legend_labels = ['Spearman R (痰湿)', '互信息 MI (风险)', 'PLS 载荷 (联合)']
else:
    x_labels = [translate_feature_name(f) for f in top10['feature']]
    title = 'Figure 2: Three-Dimensional Score Decomposition'
    y_label = 'Normalized Score (0~1)'
    legend_labels = ['Spearman R (Phlegm)', 'Mutual Info MI (Risk)', 'PLS Loading (Combined)']

# 绘制三组柱状图
bars1 = plt.bar(x - bar_width, top10['spearman_normalized'], width=bar_width, 
                label=legend_labels[0], color='#264653', alpha=0.8)
bars2 = plt.bar(x, top10['mutual_info_normalized'], width=bar_width, 
                label=legend_labels[1], color='#2A9D8F', alpha=0.8)
bars3 = plt.bar(x + bar_width, top10['pls_normalized'], width=bar_width, 
                label=legend_labels[2], color='#8AC926', alpha=0.8)

# 设置标签和标题
plt.xlabel('Key Indicators' if not use_chinese_fonts else '关键指标', fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.title(title, fontsize=14, pad=20)
plt.xticks(x, x_labels, rotation=45, ha='right')
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
plt.savefig('figure2_三维评分分解.png', dpi=300, bbox_inches='tight')
plt.close()

# === 图 3：九种体质风险贡献双面板图 ===
print("生成图 3：九种体质风险贡献双面板图")
constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']

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
constitutions_sorted_rr = [constitution_types[i] for i in rr_sorted_idx]

# 准备标签
if use_chinese_fonts:
    y_labels_left = constitutions_sorted_rr
    title_left = f'左面板：相对风险分析\n总体患病率: {total_prevalence:.3f}, 卡方检验 p = {p_value:.4f}'
    x_label_left = '相对风险 (RR)'
    legend_label_left = '总体风险基准'
    title_right = '右面板：Logistic 回归净贡献\n控制变量：血脂、活动能力、人口学因素'
    x_label_right = '标准化回归系数 β'
else:
    y_labels_left = [translate_constitution_name(c) for c in constitutions_sorted_rr]
    title_left = f'Left Panel: Relative Risk Analysis\nPrevalence: {total_prevalence:.3f}, Chi-square p = {p_value:.4f}'
    x_label_left = 'Relative Risk (RR)'
    legend_label_left = 'Population Risk Baseline'
    title_right = 'Right Panel: Logistic Regression Net Contribution\nControlled: Lipids, Activity, Demographics'
    x_label_right = 'Standardized Coefficient β'

bars_rr = ax1.barh(y_pos_const, rr_sorted, color='#E76F51', height=0.6)
ax1.invert_yaxis()
ax1.set_yticks(y_pos_const)
ax1.set_yticklabels(y_labels_left, fontsize=11)
ax1.set_xlabel(x_label_left, fontsize=12)
ax1.set_title(title_left, fontsize=12, pad=15)
ax1.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5, label=legend_label_left)
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
for const in constitutions_sorted:
    idx = constitution_types.index(const)
    coef_sorted.append(constitution_coefficients[idx])

# 颜色映射
def get_coef_color(coef):
    if coef > 0.05:
        return '#E76F51'  # 红色，风险因子
    elif coef < -0.05:
        return '#2A9D8F'  # 蓝色，保护因子
    else:
        return '#8D99AE'  # 灰色，中性

colors_coef = [get_coef_color(c) for c in coef_sorted]

# 准备标签
if use_chinese_fonts:
    y_labels_right = constitutions_sorted
else:
    y_labels_right = [translate_constitution_name(c) for c in constitutions_sorted]

bars_coef = ax2.barh(y_pos_const, coef_sorted, color=colors_coef, height=0.6)
ax2.invert_yaxis()
ax2.set_yticks(y_pos_const)
ax2.set_yticklabels(y_labels_right, fontsize=11)
ax2.set_xlabel(x_label_right, fontsize=12)
ax2.set_title(title_right, fontsize=12, pad=15)
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
plt.savefig('figure3_体质风险贡献.png', dpi=300, bbox_inches='tight')
plt.close()

# === 图 4：关键指标与双目标的相关性热力图 ===
print("生成图 4：关键指标与双目标的相关性热力图")
top10_features = feature_scores.head(10)['feature'].tolist()
heatmap_features = top10_features + ['痰湿质', '高血脂症二分类标签']

# 计算Spearman相关系数
corr_matrix = df[heatmap_features].corr(method='spearman')

# 创建热力图
plt.figure(figsize=(14, 12))

# 准备标签
if use_chinese_fonts:
    heatmap_labels = heatmap_features
    title = '图 4：关键指标与双目标的相关性热力图\n(Spearman 相关系数)'
else:
    heatmap_labels = [translate_feature_name(f) for f in heatmap_features]
    title = 'Figure 4: Correlation Heatmap of Key Indicators\n(Spearman Correlation)'

# 绘制热力图
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, square=True, linewidths=1, cbar_kws={'shrink': 0.8},
            xticklabels=heatmap_labels, yticklabels=heatmap_labels)

# 添加标题
plt.title(title, fontsize=14, pad=20)

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
plt.savefig('figure4_相关性热力图.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n所有图表生成完成！")
print("已保存的图表：")
print("1. figure1_综合评分排序.png")
print("2. figure2_三维评分分解.png")
print("3. figure3_体质风险贡献.png")
print("4. figure4_相关性热力图.png")

if not use_chinese_fonts:
    print("\n注意：由于系统未找到中文字体，图表中使用了英文标签。")
    print("如需显示中文，请安装中文字体，如 SimHei、WenQuanYi Micro Hei 等。")
