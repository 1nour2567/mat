import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

# 加载数据
df = pd.read_excel('data/raw/附件1：样例数据.xlsx')

# 定义特征和目标变量
# 血常规体检指标
blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']

# 中老年人活动量表评分
activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分', 'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分', '活动量表总分（ADL总分+IADL总分）']

# 九种体质
constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']

# 目标变量
target = '高血脂症二分类标签'

print("=== 分析一：痰湿体质与血常规指标和活动量表评分的相关性 ===")
# 计算痰湿体质与血常规指标和活动量表评分的相关性
corr_with_phlegm = df[blood_indices + activity_indices + ['痰湿质']].corr()['痰湿质']
print("\n痰湿体质与血常规指标的相关性：")
print(corr_with_phlegm[blood_indices].sort_values(ascending=False))
print("\n痰湿体质与活动量表评分的相关性：")
print(corr_with_phlegm[activity_indices].sort_values(ascending=False))

print("\n=== 分析二：筛选能预警高血脂发病风险的关键指标 ===")
# 准备特征和目标
features = blood_indices + activity_indices + constitution_types
X = df[features]
y = df[target]

# 特征选择
selector = SelectKBest(f_classif, k=15)
X_new = selector.fit_transform(X, y)
selected_features = [features[i] for i in selector.get_support(indices=True)]

print("\n通过SelectKBest筛选的关键指标：")
print(selected_features)

# 使用随机森林计算特征重要性
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

feature_importance = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\n通过随机森林计算的特征重要性：")
print(feature_importance)

print("\n=== 分析三：九种体质对发病风险的贡献度差异 ===")
# 计算九种体质与高血脂症的相关性
constitution_corr = df[constitution_types + [target]].corr()[target]
constitution_corr = constitution_corr[constitution_types].sort_values(ascending=False)
print("\n九种体质与高血脂症的相关性：")
print(constitution_corr)

# 计算九种体质在高血脂患者和非高血脂患者中的平均得分差异
high_risk = df[df[target] == 1]
low_risk = df[df[target] == 0]
constitution_diff = high_risk[constitution_types].mean() - low_risk[constitution_types].mean()
constitution_diff = constitution_diff.sort_values(ascending=False)
print("\n高血脂患者与非高血脂患者的体质得分差异：")
print(constitution_diff)

print("\n=== 分析四：综合分析 ===")
# 找出同时与痰湿体质高度相关且能有效预警高血脂的指标
phlegm_corr_threshold = 0.3
importance_threshold = 0.05

# 筛选与痰湿体质高度相关的指标
phlegm_related = corr_with_phlegm[abs(corr_with_phlegm) > phlegm_corr_threshold].index.tolist()

# 筛选重要性高的指标
important_features = feature_importance[feature_importance['importance'] > importance_threshold]['feature'].tolist()

# 找出两者的交集
key_indicators = list(set(phlegm_related) & set(important_features))
print("\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
print(key_indicators)

# 分析九种体质对发病风险的贡献度排名
print("\n九种体质对高血脂发病风险的贡献度排名（从高到低）：")
print(constitution_corr.index.tolist())
