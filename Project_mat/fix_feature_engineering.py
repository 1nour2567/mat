# 修复特征工程计算问题
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

# 修复后的熵权法
def entropy_weight_method_fixed(spearman_scores, mi_scores, pls_scores):
    """修复后的熵权法计算"""
    # 获取共同特征
    common_features = list(set(spearman_scores.keys()) & set(mi_scores.keys()) & set(pls_scores.keys()))
    
    if len(common_features) < 2:
        return {'spearman': 0.35, 'mi': 0.35, 'pls': 0.30}
    
    n = len(common_features)
    m = 3
    
    # 构建原始矩阵 X = (x_ij)_{n × m}
    X = np.zeros((n, m))
    for i, feature in enumerate(common_features):
        X[i, 0] = spearman_scores.get(feature, 0)
        X[i, 1] = mi_scores.get(feature, 0)
        X[i, 2] = pls_scores.get(feature, 0)
    
    # 检查矩阵是否有有效值
    if np.all(X == 0):
        return {'spearman': 0.35, 'mi': 0.35, 'pls': 0.30}
    
    # 第一步：矩阵标准化（极差标准化）
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    
    Z = (X - X_min) / X_range
    
    # 注：为了避免后续对数计算出现ln(0)，加一个极小值
    Z = Z + 0.0001
    
    # 第二步：计算特征在各维度下的比重 p_ij
    P = Z / np.sum(Z, axis=0, keepdims=True)
    
    # 第三步：计算各维度的信息熵 e_j
    k = 1 / np.log(n)
    e = -k * np.sum(P * np.log(P), axis=0)
    
    # 第四步：计算信息冗余度（差异性系数）d_j
    d = 1 - e
    
    # 第五步：计算最终权重 w_j
    w = d / np.sum(d)
    
    # 处理NaN值
    if np.any(np.isnan(w)):
        return {'spearman': 0.35, 'mi': 0.35, 'pls': 0.30}
    
    weights = {
        'spearman': w[0],
        'mi': w[1],
        'pls': w[2]
    }
    
    return weights

# 修复后的特征选择
def select_features_fixed(df, features, target, phlegm_score_col='痰湿质', k=20):
    """修复后的特征选择函数"""
    print(f"\n分析特征数: {len(features)}")
    
    # 1. 计算各指标
    # Spearman相关系数
    spearman_scores = {}
    for feature in features:
        if feature != phlegm_score_col and feature in df.columns:
            valid_data = df[[feature, phlegm_score_col]].dropna()
            if len(valid_data) > 2:
                try:
                    corr, _ = spearmanr(valid_data[feature], valid_data[phlegm_score_col])
                    spearman_scores[feature] = abs(corr)
                except:
                    spearman_scores[feature] = 0.0
            else:
                spearman_scores[feature] = 0.0
    
    # 互信息
    valid_features = [f for f in features if f in df.columns]
    if len(valid_features) > 0:
        X = df[valid_features].fillna(0)
        y = df[target].fillna(0)
        try:
            mi = mutual_info_classif(X, y, random_state=42)
            mi_scores = {valid_features[i]: mi[i] for i in range(len(valid_features))}
        except:
            mi_scores = {f: 0.0 for f in valid_features}
    else:
        mi_scores = {}
    
    # PLS联合结构载荷
    valid_features_pls = [f for f in features if f in df.columns and f != phlegm_score_col]
    if len(valid_features_pls) >= 2:
        X = df[valid_features_pls].fillna(0)
        Y = df[[phlegm_score_col, target]].fillna(0)
        
        try:
            # 标准化
            scaler_X = StandardScaler()
            scaler_Y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            Y_scaled = scaler_Y.fit_transform(Y)
            
            # PLS回归
            n_components = min(2, X_scaled.shape[1])
            pls = PLSRegression(n_components=n_components, scale=False)
            pls.fit(X_scaled, Y_scaled)
            
            # 计算载荷：取前两个潜变量的载荷绝对值之和
            loadings = pls.x_loadings_
            loading_scores = {valid_features_pls[i]: np.sum(np.abs(loadings[i, :])) for i in range(len(valid_features_pls))}
        except:
            loading_scores = {f: 0.0 for f in valid_features_pls}
    else:
        loading_scores = {}
    
    # 2. 熵权法计算权重
    weights = entropy_weight_method_fixed(spearman_scores, mi_scores, loading_scores)
    
    # 3. 综合评分
    common_features = list(set(spearman_scores.keys()) & set(mi_scores.keys()) & set(loading_scores.keys()))
    combined_scores = {}
    
    for feature in common_features:
        score = (weights['spearman'] * spearman_scores[feature] +
                weights['mi'] * mi_scores[feature] +
                weights['pls'] * loading_scores[feature])
        combined_scores[feature] = score
    
    # 排序并选择前k个特征
    sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    selected_features = [f[0] for f in sorted_features[:k]]
    
    # 打印前10个关键指标
    print("\n前10个关键指标：")
    for i, (feature, score) in enumerate(sorted_features[:10]):
        print(f"{i+1}. {feature}: {score:.4f}")
    
    return selected_features

# 测试修复后的函数
def test_fixed_functions():
    """测试修复后的函数"""
    print("=== 测试修复后的特征选择函数 ===")
    
    # 加载数据
    df = pd.read_pickle('data/processed/preprocessed_data.pkl')
    df = df.fillna(0)
    
    # 构建特征池
    from src.feature_engineering import build_feature_pool
    df, features = build_feature_pool(df)
    
    # 测试特征选择
    target = '高血脂症二分类标签'
    selected = select_features_fixed(df, features, target, k=10)
    print(f"\n筛选出的关键指标: {selected[:5]}")

if __name__ == "__main__":
    test_fixed_functions()