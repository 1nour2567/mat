# 综合分析脚本：筛选能表征痰湿体质严重程度且预警高血脂风险的关键指标
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

# 设置中文字体
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

def load_data(file_path):
    """加载数据"""
    try:
        return pd.read_excel(file_path)
    except UnicodeDecodeError:
        return pd.read_excel(file_path, encoding='gbk')

def select_relevant_features(df):
    """选择相关特征"""
    # 血常规体检指标
    blood_test_features = [
        'TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）',
        '空腹血糖', '血尿酸', 'BMI'
    ]
    
    # 中老年人活动量表评分
    activity_features = [
        'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）',
        'ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡',
        'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药'
    ]
    
    # 目标变量
    target_features = ['痰湿质', '高血脂症二分类标签']
    
    # 合并所有特征
    all_features = blood_test_features + activity_features + target_features
    
    # 筛选存在的特征
    existing_features = [f for f in all_features if f in df.columns]
    
    return df[existing_features]

def calculate_correlations(df, feature_cols, target_cols):
    """计算特征与目标变量的相关性"""
    correlations = {}
    
    for feature in feature_cols:
        feature_corr = {}
        for target in target_cols:
            corr, _ = spearmanr(df[feature], df[target])
            feature_corr[target] = abs(corr)
        correlations[feature] = feature_corr
    
    return correlations

def calculate_mutual_info(df, feature_cols, target_col):
    """计算互信息"""
    X = df[feature_cols]
    y = df[target_col]
    
    # 处理缺失值
    X = X.fillna(X.mean())
    
    mutual_info = mutual_info_classif(X, y, random_state=42)
    return dict(zip(feature_cols, mutual_info))

def calculate_pls_loadings(df, feature_cols, target_cols):
    """计算PLS联合结构载荷"""
    X = df[feature_cols]
    y = df[target_cols]
    
    # 处理缺失值
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PLS回归
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y)
    
    # 计算载荷
    loadings = np.zeros(len(feature_cols))
    for i in range(len(feature_cols)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    return dict(zip(feature_cols, loadings))

def entropy_weight_method(scores):
    """熵权法计算权重"""
    # 标准化
    def normalize(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001  # 避免对数为0
        return X_norm
    
    # 构建矩阵
    n = len(scores)
    m = 3  # 三个维度：Spearman相关、互信息、PLS载荷
    X_matrix = np.zeros((n, m))
    
    for i, (feature, score_dict) in enumerate(scores.items()):
        X_matrix[i, 0] = score_dict['spearman_痰湿质']
        X_matrix[i, 1] = score_dict['mutual_info_高血脂']
        X_matrix[i, 2] = score_dict['pls_loadings']
    
    # 标准化
    X_norm = normalize(X_matrix)
    
    # 计算比重
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    # 计算信息熵
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    # 计算权重
    d = 1 - e
    w = d / np.sum(d)
    
    return w

def main():
    print("=== 综合分析：筛选能表征痰湿体质严重程度且预警高血脂风险的关键指标 ===")
    
    # 加载数据
    file_path = 'data/raw/附件1：样例数据.xlsx'
    df = load_data(file_path)
    print(f"成功加载数据，样本数：{len(df)}")
    
    # 选择相关特征
    df_selected = select_relevant_features(df)
    print(f"筛选后特征数：{len(df_selected.columns)}")
    
    # 分离特征和目标变量
    feature_cols = [col for col in df_selected.columns if col not in ['痰湿质', '高血脂症二分类标签']]
    target_cols = ['痰湿质', '高血脂症二分类标签']
    
    print(f"特征列数：{len(feature_cols)}")
    print(f"目标变量：{target_cols}")
    
    # 计算Spearman相关系数
    print("\n1. 计算Spearman相关系数...")
    correlations = calculate_correlations(df_selected, feature_cols, target_cols)
    
    # 计算互信息（针对高血脂症）
    print("2. 计算互信息...")
    mutual_info = calculate_mutual_info(df_selected, feature_cols, '高血脂症二分类标签')
    
    # 计算PLS联合结构载荷
    print("3. 计算PLS联合结构载荷...")
    pls_loadings = calculate_pls_loadings(df_selected, feature_cols, target_cols)
    
    # 整合评分
    scores = {}
    for feature in feature_cols:
        scores[feature] = {
            'spearman_痰湿质': correlations[feature]['痰湿质'],
            'mutual_info_高血脂': mutual_info[feature],
            'pls_loadings': pls_loadings[feature]
        }
    
    # 使用熵权法计算权重
    print("4. 使用熵权法计算权重...")
    weights = entropy_weight_method(scores)
    print(f"各维度权重：")
    print(f"Spearman相关（痰湿质）: {weights[0]:.4f}")
    print(f"互信息（高血脂）: {weights[1]:.4f}")
    print(f"PLS载荷（联合）: {weights[2]:.4f}")
    
    # 计算综合评分
    print("5. 计算综合评分...")
    comprehensive_scores = {}
    for feature, score_dict in scores.items():
        total_score = (
            weights[0] * score_dict['spearman_痰湿质'] +
            weights[1] * score_dict['mutual_info_高血脂'] +
            weights[2] * score_dict['pls_loadings']
        )
        comprehensive_scores[feature] = total_score
    
    # 排序并输出结果
    print("\n6. 关键指标排序...")
    sorted_features = sorted(comprehensive_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=== 综合评分结果（前20个关键指标）===")
    print("排名\t指标名称\t\t\t综合评分")
    print("-" * 60)
    for i, (feature, score) in enumerate(sorted_features[:20], 1):
        print(f"{i}\t{feature}\t\t{score:.4f}")
    
    # 分析指标类型
    print("\n=== 指标类型分析 ===")
    blood_test_features = [
        'TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）',
        '空腹血糖', '血尿酸', 'BMI'
    ]
    
    activity_features = [
        'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）',
        'ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡',
        'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药'
    ]
    
    blood_test_ranking = []
    activity_ranking = []
    
    for i, (feature, score) in enumerate(sorted_features, 1):
        if feature in blood_test_features:
            blood_test_ranking.append((i, feature, score))
        elif feature in activity_features:
            activity_ranking.append((i, feature, score))
    
    print("\n血常规体检指标排名：")
    print("排名\t指标名称\t\t\t综合评分")
    print("-" * 60)
    for rank, feature, score in blood_test_ranking:
        print(f"{rank}\t{feature}\t\t{score:.4f}")
    
    print("\n活动量表评分排名：")
    print("排名\t指标名称\t\t\t综合评分")
    print("-" * 60)
    for rank, feature, score in activity_ranking:
        print(f"{rank}\t{feature}\t\t{score:.4f}")
    
    # 输出结论
    print("\n=== 分析结论 ===")
    print("1. 能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    print(f"   第1名：{sorted_features[0][0]}（{sorted_features[0][1]:.4f}")
    print(f"   第2名：{sorted_features[1][0]}（{sorted_features[1][1]:.4f}")
    print(f"   第3名：{sorted_features[2][0]}（{sorted_features[2][1]:.4f}")
    
    print("\n2. 血常规体检指标与活动量表评分的对比：")
    print(f"   血常规指标在前10名中的数量：{sum(1 for rank, _, _ in blood_test_ranking if rank <= 10)}")
    print(f"   活动量表评分在前10名中的数量：{sum(1 for rank, _, _ in activity_ranking if rank <= 10)}")
    
    print("\n3. 综合评估：")
    if blood_test_ranking[0][0] < activity_ranking[0][0]:
        print("   血常规体检指标在表征痰湿体质和预警高血脂风险方面表现更优")
    else:
        print("   活动量表评分在表征痰湿体质和预警高血脂风险方面表现更优")

if __name__ == "__main__":
    main()