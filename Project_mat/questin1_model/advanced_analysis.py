import pandas as pd
import numpy as np
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 加载数据
df = pd.read_excel('data/raw/附件1：样例数据.xlsx')

# 5.1.2 候选特征池构建
print("=== 5.1.2 候选特征池构建 ===")

# 基础特征层
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

# 血脂异常项数
def count_dyslipidemia(row):
    count = 0
    # 假设参考值：TC>5.2，TG>1.7，LDL-C>3.4，HDL-C<1.0
    if row['TC（总胆固醇）'] > 5.2: count += 1
    if row['TG（甘油三酯）'] > 1.7: count += 1
    if row['LDL-C（低密度脂蛋白）'] > 3.4: count += 1
    if row['HDL-C（高密度脂蛋白）'] < 1.0: count += 1
    return count
df['血脂异常项数'] = df.apply(count_dyslipidemia, axis=1)

# 尿酸异常标志（假设参考值：男性>420，女性>360）
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

# 所有特征
all_features = basic_features + derived_features + cross_features
print(f"\n基础特征层: {len(basic_features)} 个特征")
print(f"派生特征层: {len(derived_features)} 个特征")
print(f"中西医交叉特征层: {len(cross_features)} 个特征")
print(f"总特征数: {len(all_features)} 个特征")

# 5.1.3 基于Spearman-互信息-PLS的双目标联合筛选模型
print("\n=== 5.1.3 基于Spearman-互信息-PLS的双目标联合筛选模型 ===")

# 目标变量
target = '高血脂症二分类标签'
target_phlegm = '痰湿质'

# 准备数据
X = df[all_features]
y_phlegm = df[target_phlegm]
y_risk = df[target]

# 1. Spearman相关系数（痰湿表征能力）
spearman_scores = []
for feature in all_features:
    corr, _ = spearmanr(df[feature], y_phlegm)
    spearman_scores.append(abs(corr))
spearman_scores = np.array(spearman_scores)

# 2. 互信息（风险预警能力）
mutual_info_scores = mutual_info_classif(X, y_risk, random_state=42)

# 3. PLS联合结构载荷（双目标整合能力）
# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 双响应矩阵
y_combined = np.column_stack((y_phlegm, y_risk))

# PLS回归
pls = PLSRegression(n_components=2, scale=False)
pls.fit(X_scaled, y_combined)

# 计算联合结构载荷
# 结构载荷是X与潜变量的相关系数
pls_loadings = np.zeros(len(all_features))
for i in range(len(all_features)):
    # 计算每个特征与前两个潜变量的相关系数，并取平均值
    corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
    corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
    pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2

# 4. 综合评分函数
# 归一化
def normalize(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score == 0:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)

spearman_normalized = normalize(spearman_scores)
mutual_info_normalized = normalize(mutual_info_scores)
pls_normalized = normalize(pls_loadings)

# 综合评分
weights = [0.35, 0.35, 0.30]
total_scores = weights[0] * spearman_normalized + weights[1] * mutual_info_normalized + weights[2] * pls_normalized

# 排序
feature_scores = pd.DataFrame({
    'feature': all_features,
    'spearman_score': spearman_scores,
    'mutual_info_score': mutual_info_scores,
    'pls_loading': pls_loadings,
    'total_score': total_scores
})

feature_scores = feature_scores.sort_values('total_score', ascending=False)

print("\n关键指标排序（前20）：")
print(feature_scores.head(20)[['feature', 'total_score']])

# 5.1.4 九种体质风险贡献度分析
print("\n=== 5.1.4 九种体质风险贡献度分析 ===")

constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']

# 1. 基于主标签的相对风险分析
print("\n1. 基于主标签的相对风险分析：")
total_prevalence = df[target].mean()
print(f"总体高血脂患病率: {total_prevalence:.4f}")

relative_risks = {}
for i in range(1, 10):  # 体质标签1-9
    subset = df[df['体质标签'] == i]
    if len(subset) > 0:
        prevalence = subset[target].mean()
        relative_risk = prevalence / total_prevalence
        relative_risks[i] = relative_risk
        print(f"体质标签 {i} 的患病率: {prevalence:.4f}, 相对风险: {relative_risk:.4f}")

# 卡方独立性检验
contingency_table = pd.crosstab(df['体质标签'], df[target])
chi2, p_value, _, _ = chi2_contingency(contingency_table)
print(f"\n卡方检验结果: chi2={chi2:.4f}, p-value={p_value:.4f}")
if p_value < 0.05:
    print("体质标签与高血脂发病风险存在统计学显著关联")
else:
    print("体质标签与高血脂发病风险无统计学显著关联")

# 2. 基于积分的多因素贡献分析
print("\n2. 基于积分的多因素贡献分析：")

# 控制变量
control_vars = ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI',
                '活动量表总分（ADL总分+IADL总分）', '年龄组', '性别', '吸烟史', '饮酒史']

# 构建Logistic回归模型
X_logistic = df[constitution_types + control_vars]
y_logistic = df[target]

# 标准化
scaler_logistic = StandardScaler()
X_logistic_scaled = scaler_logistic.fit_transform(X_logistic)

# 拟合模型
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_logistic_scaled, y_logistic)

# 标准化回归系数
coefficients = logistic_model.coef_[0]
feature_names = constitution_types + control_vars

# 体质相关的系数
constitution_coefficients = {}
for i, feature in enumerate(feature_names):
    if feature in constitution_types:
        constitution_coefficients[feature] = abs(coefficients[i])

# 排序
constitution_coefficients_sorted = sorted(constitution_coefficients.items(), key=lambda x: x[1], reverse=True)

print("\n九种体质的标准化回归系数（绝对值）：")
for feature, coef in constitution_coefficients_sorted:
    print(f"{feature}: {coef:.4f}")

# 5.1.5 求解结果
print("\n=== 5.1.5 求解结果 ===")
print("\n关键指标排序（前10）：")
print(feature_scores.head(10)['feature'].tolist())

print("\n体质贡献结论：")
print("根据标准化回归系数，体质贡献度从高到低：")
for feature, coef in constitution_coefficients_sorted:
    print(f"{feature}: {coef:.4f}")
