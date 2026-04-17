# 特征池构建与筛选模块
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression

def build_feature_pool(df):
    """构建特征池"""
    # 基础特征
    features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 类别特征编码
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_features:
        df = pd.get_dummies(df, columns=[col], drop_first=True)
    
    # 中西医交叉特征
    if '痰湿质得分' in df.columns and 'BMI' in df.columns:
        df['痰湿质得分×BMI'] = df['痰湿质得分'] * df['BMI']
    
    if '痰湿质得分' in df.columns and 'TG' in df.columns:
        df['痰湿质得分×TG'] = df['痰湿质得分'] * df['TG']
    
    if '痰湿质得分' in df.columns and 'AIP' in df.columns:
        df['痰湿质得分×AIP'] = df['痰湿质得分'] * df['AIP']
    
    if '痰湿质得分' in df.columns and 'LDL-C' in df.columns:
        df['痰湿质得分×LDL-C'] = df['痰湿质得分'] * df['LDL-C']
    
    if '痰湿质得分' in df.columns and 'HDL-C' in df.columns:
        df['痰湿质得分/HDL-C'] = df['痰湿质得分'] / df['HDL-C']
    
    if '气虚质得分' in df.columns and 'TC' in df.columns:
        df['气虚质得分×TC'] = df['气虚质得分'] * df['TC']
    
    # 更新特征列表
    features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    return df, features

def handle_class_imbalance(X, y):
    """处理类别不平衡"""
    # 使用SMOTE-Tomek算法
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)
    return X_res, y_res

def calculate_spearman_correlation(df, features, phlegm_score_col):
    """计算Spearman相关系数"""
    correlations = {}
    for feature in features:
        if feature != phlegm_score_col:
            corr, _ = spearmanr(df[feature], df[phlegm_score_col])
            correlations[feature] = abs(corr)
    return correlations

def calculate_mutual_info(df, features, target):
    """计算互信息"""
    X = df[features]
    y = df[target]
    mi = mutual_info_classif(X, y, random_state=42)
    mi_dict = {features[i]: mi[i] for i in range(len(features))}
    return mi_dict

def calculate_pls_loadings(df, features, phlegm_score_col, target):
    """计算PLS联合结构载荷"""
    X = df[features]
    Y = df[[phlegm_score_col, target]]
    
    # 标准化
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)
    
    # PLS回归
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, Y_scaled)
    
    # 计算载荷
    loadings = pls.x_loadings_
    # 取前两个潜变量的载荷绝对值之和
    loading_scores = {features[i]: np.sum(np.abs(loadings[i, :])) for i in range(len(features))}
    return loading_scores

def normalize_scores(scores):
    """归一化评分"""
    max_score = max(scores.values())
    min_score = min(scores.values())
    if max_score == min_score:
        return {k: 1.0 for k in scores}
    return {k: (v - min_score) / (max_score - min_score) for k, v in scores.items()}

def select_features(df, features, target, phlegm_score_col='痰湿质得分', k=20):
    """基于Spearman-互信息-PLS的双目标联合筛选"""
    # 计算各指标
    spearman_scores = calculate_spearman_correlation(df, features, phlegm_score_col)
    mi_scores = calculate_mutual_info(df, features, target)
    pls_scores = calculate_pls_loadings(df, features, phlegm_score_col, target)
    
    # 归一化
    spearman_normalized = normalize_scores(spearman_scores)
    mi_normalized = normalize_scores(mi_scores)
    pls_normalized = normalize_scores(pls_scores)
    
    # 综合评分
    weights = {'spearman': 0.35, 'mi': 0.35, 'pls': 0.30}
    combined_scores = {}
    for feature in features:
        if feature in spearman_normalized and feature in mi_normalized and feature in pls_normalized:
            score = (weights['spearman'] * spearman_normalized[feature] +
                    weights['mi'] * mi_normalized[feature] +
                    weights['pls'] * pls_normalized[feature])
            combined_scores[feature] = score
    
    # 排序并选择前k个特征
    sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    selected_features = [f[0] for f in sorted_features[:k]]
    
    # 打印前10个关键指标
    print("前10个关键指标：")
    for i, (feature, score) in enumerate(sorted_features[:10]):
        print(f"{i+1}. {feature}: {score:.4f}")
    
    return selected_features

def analyze_constitution_contribution(df, target):
    """九种体质风险贡献度分析"""
    # 体质积分列
    constitution_scores = ['平和质得分', '气虚质得分', '阳虚质得分', '阴虚质得分', '痰湿质得分', 
                         '湿热质得分', '血瘀质得分', '气郁质得分', '特禀质得分']
    
    # 基于主标签的相对风险分析
    if '体质标签' in df.columns:
        overall_prevalence = df[target].mean()
        print("\n基于主标签的相对风险分析：")
        for constitution in df['体质标签'].unique():
            subset = df[df['体质标签'] == constitution]
            if len(subset) > 0:
                prevalence = subset[target].mean()
                relative_risk = prevalence / overall_prevalence if overall_prevalence > 0 else 0
                print(f"{constitution}: 患病率={prevalence:.4f}, 相对风险={relative_risk:.4f}")
    
    # 基于积分的多因素贡献分析
    print("\n基于积分的多因素贡献分析：")
    # 这里可以添加Logistic回归分析
    
    return True

def feature_engineering(input_path, output_path, target):
    """完整特征工程流程"""
    # 加载预处理后的数据
    df = pd.read_pickle(input_path)
    
    # 构建特征池
    df, features = build_feature_pool(df)
    
    # 特征筛选
    selected_features = select_features(df, features, target)
    
    # 体质贡献度分析
    analyze_constitution_contribution(df, target)
    
    # 标准化
    scaler = StandardScaler()
    df[selected_features] = scaler.fit_transform(df[selected_features])
    
    # 保存处理后的数据
    df.to_pickle(output_path)
    
    return df, selected_features