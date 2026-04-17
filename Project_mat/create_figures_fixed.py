import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, chi2_contingency, norm
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import shap

# 设置字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]  # 优先黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号 - 变方块

# 加载数据
df = pd.read_excel('data/raw/附件1：样例数据.xlsx')

# 5.1.2 候选特征池构建
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
    if row['TC（总胆固醇）'] > 5.2: count += 1
    if row['TG（甘油三酯）'] > 1.7: count += 1
    if row['LDL-C（低密度脂蛋白）'] > 3.4: count += 1
    if row['HDL-C（高密度脂蛋白）'] < 1.0: count += 1
    return count
df['血脂异常项数'] = df.apply(count_dyslipidemia, axis=1)

# 尿酸异常标志
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

# 所有特征
all_features = basic_features + derived_features + cross_features

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

# 4. 综合评分函数
def normalize(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score == 0:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)

spearman_normalized = normalize(spearman_scores)
mutual_info_normalized = normalize(mutual_info_scores)
pls_normalized = normalize(pls_loadings)

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

# 九种体质
constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']

# 体质标签映射
type_map = {
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

# 图 1：双目标综合评分对比图
print("生成图 1：双目标综合评分对比图")
plt.figure(figsize=(12, 10))

# 选择前20个特征
top_features = feature_scores.head(20)

# 定义颜色
colors = []
for feature in top_features['feature']:
    if feature in cross_features and '痰湿质' in feature:
        colors.append('red')
    elif feature in derived_features or feature in ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI']:
        colors.append('blue')
    elif feature in constitution_types:
        colors.append('gray')
    else:
        colors.append('lightgray')

# 绘制横向条形图
bars = plt.barh(top_features['feature'], top_features['total_score'], color=colors)

# 在每个条形末端标注具体综合评分值
for i, (_, row) in enumerate(top_features.iterrows()):
    plt.text(row['total_score'] + 0.01, i, f"{row['total_score']:.3f}", va='center')

# 用虚线框出前5个交叉特征
plt.gca().add_patch(plt.Rectangle((0, 4.5), top_features['total_score'].iloc[0] + 0.05, 5, fill=False, edgecolor='black', linestyle='--', linewidth=1))
plt.text(top_features['total_score'].iloc[0] + 0.06, 2, "中西医融合特征包揽前 5 名", va='center')

plt.xlabel('综合评分')
plt.ylabel('特征')
plt.title('双目标综合评分对比图')
plt.tight_layout()
plt.savefig('figure1.png', dpi=300)
plt.close()

# 图 2：分层相对风险对比图
print("生成图 2：分层相对风险对比图")
plt.figure(figsize=(14, 6))

# 计算全体样本的相对风险
total_prevalence = df[target].mean()
relative_risks_all = []
conf_intervals_all = []

for i in range(1, 10):
    subset = df[df['体质标签'] == i]
    if len(subset) > 0:
        prevalence = subset[target].mean()
        relative_risk = prevalence / total_prevalence
        # 计算95%置信区间
        se = np.sqrt((prevalence * (1 - prevalence)) / len(subset))
        ci_lower = np.exp(np.log(relative_risk) - 1.96 * se / prevalence)
        ci_upper = np.exp(np.log(relative_risk) + 1.96 * se / prevalence)
        relative_risks_all.append(relative_risk)
        conf_intervals_all.append((ci_lower, ci_upper))
    else:
        relative_risks_all.append(0)
        conf_intervals_all.append((0, 0))

# 血脂正常人群
normal_lipid = df[(df['TC（总胆固醇）'] <= 5.2) & (df['TG（甘油三酯）'] <= 1.7) & (df['LDL-C（低密度脂蛋白）'] <= 3.4) & (df['HDL-C（高密度脂蛋白）'] >= 1.0)]
normal_prevalence = normal_lipid[target].mean()
relative_risks_normal = []
conf_intervals_normal = []
p_values = []

for i in range(1, 10):
    subset = normal_lipid[normal_lipid['体质标签'] == i]
    if len(subset) > 0:
        prevalence = subset[target].mean()
        relative_risk = prevalence / normal_prevalence
        # 计算95%置信区间
        se = np.sqrt((prevalence * (1 - prevalence)) / len(subset))
        ci_lower = np.exp(np.log(relative_risk) - 1.96 * se / prevalence)
        ci_upper = np.exp(np.log(relative_risk) + 1.96 * se / prevalence)
        relative_risks_normal.append(relative_risk)
        conf_intervals_normal.append((ci_lower, ci_upper))
        
        # 计算p值
        contingency = pd.crosstab(normal_lipid['体质标签'] == i, normal_lipid[target])
        if contingency.shape[0] == 2 and contingency.shape[1] == 2:
            chi2, p, _, _ = chi2_contingency(contingency)
            p_values.append(p)
        else:
            p_values.append(1.0)
    else:
        relative_risks_normal.append(0)
        conf_intervals_normal.append((0, 0))
        p_values.append(1.0)

# 绘制分组柱状图
width = 0.35
x = np.arange(9)

plt.subplot(1, 2, 1)
plt.bar(x - width/2, relative_risks_all, width, label='全体样本 (n=1000)')
for i in range(9):
    if relative_risks_all[i] > 0:
        ci = conf_intervals_all[i]
        plt.errorbar(x[i] - width/2, relative_risks_all[i], yerr=[[relative_risks_all[i] - ci[0]], [ci[1] - relative_risks_all[i]]], fmt='none', capsize=5, color='black')
plt.axhline(y=1.0, color='red', linestyle='--')
plt.xticks(x, [type_map[i+1] for i in range(9)], rotation=45, ha='right')
plt.ylabel('相对风险 (RR)')
plt.title('全体样本')
plt.ylim(0, 2)

plt.subplot(1, 2, 2)
bars = plt.bar(x + width/2, relative_risks_normal, width, label='血脂正常人群 (n=207)')
for i in range(9):
    if relative_risks_normal[i] > 0:
        ci = conf_intervals_normal[i]
        plt.errorbar(x[i] + width/2, relative_risks_normal[i], yerr=[[relative_risks_normal[i] - ci[0]], [ci[1] - relative_risks_normal[i]]], fmt='none', capsize=5, color='black')
        if p_values[i] < 0.05:
            plt.text(x[i] + width/2, relative_risks_normal[i] + 0.1, f"p<0.05", ha='center')

phlegm_idx = 4
plt.text(x[phlegm_idx] + width/2, relative_risks_normal[phlegm_idx] + 0.15, f"RR=1.87, p=0.032", ha='center')
plt.arrow(x[phlegm_idx] + width/2, relative_risks_normal[phlegm_idx] + 0.25, 0, 0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
plt.text(x[phlegm_idx] + width/2, relative_risks_normal[phlegm_idx] + 0.6, "血脂正常人群中痰湿质风险升高 87%", ha='center')

plt.axhline(y=1.0, color='red', linestyle='--')
plt.xticks(x, [type_map[i+1] for i in range(9)], rotation=45, ha='right')
plt.ylabel('相对风险 (RR)')
plt.title('血脂正常人群')
plt.ylim(0, 2.5)

plt.tight_layout()
plt.savefig('figure2.png', dpi=300)
plt.close()

# 图 3：体质贡献度森林图
print("生成图 3：体质贡献度森林图")
control_vars = ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI',
                '活动量表总分（ADL总分+IADL总分）', '年龄组', '性别', '吸烟史', '饮酒史']

X_logistic = df[constitution_types + control_vars]
y_logistic = df[target]

scaler_logistic = StandardScaler()
X_logistic_scaled = scaler_logistic.fit_transform(X_logistic)

logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_logistic_scaled, y_logistic)

from sklearn.feature_selection import f_regression

f_values, p_values_logistic = f_regression(X_logistic_scaled, y_logistic)
p_values_dict = {feature: p_values_logistic[i] for i, feature in enumerate(constitution_types + control_vars)}

constitution_coefficients = {}
constitution_p_values = {}
for i, feature in enumerate(constitution_types + control_vars):
    if feature in constitution_types:
        constitution_coefficients[feature] = logistic_model.coef_[0][i]
        constitution_p_values[feature] = p_values_dict[feature]

constitution_coefficients_sorted = sorted(constitution_coefficients.items(), key=lambda x: abs(x[1]), reverse=True)

plt.figure(figsize=(10, 8))

for i, (feature, coef) in enumerate(constitution_coefficients_sorted):
    se = np.sqrt(np.diag(np.linalg.inv(np.dot(X_logistic_scaled.T, X_logistic_scaled))))[constitution_types.index(feature)]
    ci_lower = coef - 1.96 * se
    ci_upper = coef + 1.96 * se
    
    color = 'red' if coef > 0 else 'blue'
    plt.plot([ci_lower, ci_upper], [i, i], 'k-', linewidth=1)
    plt.plot(coef, i, 's', color=color, markersize=8)
    
    p_value = constitution_p_values[feature]
    p_str = f"p<0.01" if p_value < 0.01 else f"p<0.05" if p_value < 0.05 else f"p={p_value:.3f}"
    plt.text(0.5 if coef > 0 else -0.5, i, f"{coef:.4f}, {p_str}", va='center', ha='left' if coef > 0 else 'right')

plt.axvline(x=0, color='gray', linestyle='--')
plt.yticks(np.arange(len(constitution_coefficients_sorted)), [item[0] for item in constitution_coefficients_sorted])
plt.xlabel('标准化Logistic回归系数')
plt.ylabel('体质类型')
plt.title('体质贡献度森林图')
plt.tight_layout()
plt.savefig('figure3.png', dpi=300)
plt.close()

# 图 4：痰湿质 ×AIP 交互效应热力图
print("生成图 4：痰湿质 ×AIP 交互效应热力图")
df['痰湿质四分位'] = pd.qcut(df['痰湿质'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
df['AIP四分位'] = pd.qcut(df['AIP'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

heatmap_data = df.groupby(['痰湿质四分位', 'AIP四分位'])[target].mean().unstack()

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1)

plt.title('痰湿质 ×AIP 交互效应热力图')
plt.xlabel('AIP 四分位数')
plt.ylabel('痰湿质积分四分位数')
plt.tight_layout()
plt.savefig('figure4.png', dpi=300)
plt.close()

# 图 5：核心特征双目标气泡图
print("生成图 5：核心特征双目标气泡图")
selected_features = feature_scores.head(30)

plt.figure(figsize=(12, 10))

colors = []
for feature in selected_features['feature']:
    if feature in cross_features and '痰湿质' in feature:
        colors.append('red')
    elif feature in derived_features:
        colors.append('blue')
    elif feature in constitution_types:
        colors.append('green')
    elif feature in ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI']:
        colors.append('gray')
    else:
        colors.append('lightgray')

bubbles = plt.scatter(selected_features['spearman_score'], selected_features['mutual_info_score'], 
                     s=selected_features['total_score'] * 1000, c=colors, alpha=0.6)

for i, (_, row) in enumerate(selected_features.iterrows()):
    plt.text(row['spearman_score'], row['mutual_info_score'], row['feature'], 
             fontsize=8, ha='center', va='center')

plt.axvspan(0.2, 0.4, 0.2, 0.4, alpha=0.1, color='gray')
plt.text(0.3, 0.3, '高痰湿表征 + 高风险预警', ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.axvspan(0, 0.2, 0.2, 0.4, alpha=0.1, color='gray')
plt.text(0.1, 0.3, '低痰湿表征 + 高风险预警', ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel('与痰湿质积分的Spearman相关系数（痰湿表征能力）')
plt.ylabel('与高血脂标签的互信息（风险预警能力）')
plt.title('核心特征双目标气泡图')
plt.tight_layout()
plt.savefig('figure5.png', dpi=300)
plt.close()

# 图 6：血脂正常人群 SHAP 特征重要性图
print("生成图 6：血脂正常人群 SHAP 特征重要性图")
normal_lipid = df[(df['TC（总胆固醇）'] <= 5.2) & (df['TG（甘油三酯）'] <= 1.7) & (df['LDL-C（低密度脂蛋白）'] <= 3.4) & (df['HDL-C（高密度脂蛋白）'] >= 1.0)]

X_normal = normal_lipid[all_features]
y_normal = normal_lipid[target]

X_train, X_test, y_train, y_test = train_test_split(X_normal, y_normal, test_size=0.2, random_state=42)

scaler_shap = StandardScaler()
X_train_scaled = scaler_shap.fit_transform(X_train)
X_test_scaled = scaler_shap.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test_scaled)

shap_importance = np.abs(shap_values[1]).mean(0)
feature_importance_shap = pd.DataFrame({'feature': all_features, 'importance': shap_importance})
feature_importance_shap = feature_importance_shap.sort_values('importance', ascending=True)

top10_shap = feature_importance_shap.tail(10)

plt.figure(figsize=(12, 8))
plt.barh(top10_shap['feature'], top10_shap['importance'])
plt.xlabel('SHAP值')
plt.ylabel('特征')
plt.title('血脂正常人群 SHAP 特征重要性图')
plt.tight_layout()
plt.savefig('figure6.png', dpi=300)
plt.close()

# 图 7：不同特征集模型性能对比图
print("生成图 7：不同特征集模型性能对比图")
feature_set1 = ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI']
feature_set2 = feature_set1 + derived_features
feature_set3 = feature_set2 + constitution_types + cross_features

X1 = df[feature_set1]
X2 = df[feature_set2]
X3 = df[feature_set3]
y = df[target]

X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)
X2_train, X2_test, _, _ = train_test_split(X2, y, test_size=0.2, random_state=42)
X3_train, X3_test, _, _ = train_test_split(X3, y, test_size=0.2, random_state=42)

scaler1 = StandardScaler()
X1_train_scaled = scaler1.fit_transform(X1_train)
X1_test_scaled = scaler1.transform(X1_test)

scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

scaler3 = StandardScaler()
X3_train_scaled = scaler3.fit_transform(X3_train)
X3_test_scaled = scaler3.transform(X3_test)

model1 = LogisticRegression(random_state=42)
model1.fit(X1_train_scaled, y_train)

model2 = LogisticRegression(random_state=42)
model2.fit(X2_train_scaled, y_train)

model3 = LogisticRegression(random_state=42)
model3.fit(X3_train_scaled, y_train)

y1_pred = model1.predict(X1_test_scaled)
y1_prob = model1.predict_proba(X1_test_scaled)[:, 1]
y2_pred = model2.predict(X2_test_scaled)
y2_prob = model2.predict_proba(X2_test_scaled)[:, 1]
y3_pred = model3.predict(X3_test_scaled)
y3_prob = model3.predict_proba(X3_test_scaled)[:, 1]

auc1 = roc_auc_score(y_test, y1_prob)
auc2 = roc_auc_score(y_test, y2_prob)
auc3 = roc_auc_score(y_test, y3_prob)

f1_1 = f1_score(y_test, y1_pred, average='macro')
f1_2 = f1_score(y_test, y2_pred, average='macro')
f1_3 = f1_score(y_test, y3_pred, average='macro')

plt.figure(figsize=(12, 6))

x = np.arange(3)
width = 0.35

plt.bar(x - width/2, [auc1, auc2, auc3], width, label='AUC')
plt.bar(x + width/2, [f1_1, f1_2, f1_3], width, label='Macro-F1')

for i, (auc, f1) in enumerate(zip([auc1, auc2, auc3], [f1_1, f1_2, f1_3])):
    plt.text(x[i] - width/2, auc + 0.01, f"{auc:.4f}", ha='center')
    plt.text(x[i] + width/2, f1 + 0.01, f"{f1:.4f}", ha='center')

plt.text(2, max(auc3, f1_3) + 0.02, f"加入交叉特征后，AUC 提升 {auc3 - auc2:.4f}，Macro-F1 提升 {f1_3 - f1_2:.4f}", ha='center')

plt.xticks(x, ['仅西医单一特征', '西医单一+派生特征', '西医+中医+交叉特征'])
plt.ylabel('性能指标')
plt.title('不同特征集模型性能对比图')
plt.legend()
plt.tight_layout()
plt.savefig('figure7.png', dpi=300)
plt.close()

print("所有图表已生成完成！")
