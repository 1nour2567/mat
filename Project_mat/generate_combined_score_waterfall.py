#!/usr/bin/env python3
# 生成综合评分瀑布图

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 颜色设置
colors = {
    'spearman': '#1f77b4',  # 蓝色 - Spearman贡献
    'mutual_info': '#ff7f0e',  # 橙色 - 互信息贡献
    'pls': '#2ca02c'  # 绿色 - PLS贡献
}

# 指标类型映射
feature_types = {
    '痰湿质得分': '基础中医',
    '痰湿质得分×BMI': '交叉特征',
    '痰湿质得分×TG': '交叉特征',
    '痰湿质得分/HDL-C': '交叉特征',
    '痰湿质得分×LDL-C': '交叉特征',
    '血脂异常项数': '基础西医',
    '痰湿质得分×AIP': '交叉特征',
    'TG/HDL比值': '基础西医',
    'TG': '基础西医',
    'AIP': '基础西医',
    'TC': '基础西医',
    '血尿酸': '基础西医',
    'ADL总分': '活动量表',
    'HDL-C': '基础西医',
    '活动量表总分': '活动量表'
}

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
    print("=== 生成综合评分瀑布图 ===")
    
    # 加载数据
    file_path = 'data/raw/附件1：样例数据.xlsx'
    df = load_data(file_path)
    print(f"成功加载数据，样本数：{len(df)}")
    
    # 选择相关特征
    df_selected = select_relevant_features(df)
    
    # 分离特征和目标变量
    feature_cols = [col for col in df_selected.columns if col not in ['痰湿质', '高血脂症二分类标签']]
    target_cols = ['痰湿质', '高血脂症二分类标签']
    
    # 计算Spearman相关系数
    print("1. 计算Spearman相关系数...")
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
    
    # 计算综合评分和各维度贡献
    print("5. 计算综合评分和各维度贡献...")
    comprehensive_scores = {}
    contributions = {}
    for feature, score_dict in scores.items():
        spearman_contrib = weights[0] * score_dict['spearman_痰湿质']
        mutual_info_contrib = weights[1] * score_dict['mutual_info_高血脂']
        pls_contrib = weights[2] * score_dict['pls_loadings']
        total_score = spearman_contrib + mutual_info_contrib + pls_contrib
        
        comprehensive_scores[feature] = total_score
        contributions[feature] = {
            'spearman': spearman_contrib,
            'mutual_info': mutual_info_contrib,
            'pls': pls_contrib
        }
    
    # 排序并选择Top 15
    print("6. 选择Top 15指标...")
    sorted_features = sorted(comprehensive_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    top_features = [f[0] for f in sorted_features]
    top_scores = [f[1] for f in sorted_features]
    
    # 准备堆叠数据
    spearman_contribs = []
    mutual_info_contribs = []
    pls_contribs = []
    feature_type_labels = []
    
    for feature in top_features:
        contrib = contributions[feature]
        spearman_contribs.append(contrib['spearman'])
        mutual_info_contribs.append(contrib['mutual_info'])
        pls_contribs.append(contrib['pls'])
        feature_type_labels.append(feature_types.get(feature, '其他'))
    
    # 创建瀑布图
    print("7. 创建综合评分瀑布图...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制堆叠柱状图
    x = np.arange(len(top_features))
    width = 0.8
    
    # 底部位置
    bottom = np.zeros(len(top_features))
    
    # 绘制各维度贡献
    spearman_bars = ax.bar(x, spearman_contribs, width, label='Spearman贡献', color=colors['spearman'], bottom=bottom)
    bottom += spearman_contribs
    
    mutual_info_bars = ax.bar(x, mutual_info_contribs, width, label='互信息贡献', color=colors['mutual_info'], bottom=bottom)
    bottom += mutual_info_contribs
    
    pls_bars = ax.bar(x, pls_contribs, width, label='PLS贡献', color=colors['pls'], bottom=bottom)
    
    # 设置图表属性
    ax.set_xlabel('指标名称（类型）', fontsize=12)
    ax.set_ylabel('综合评分', fontsize=12)
    ax.set_title('综合评分瀑布图（Top 15指标）', fontsize=16)
    
    # 设置x轴标签，包含指标类型
    x_labels = [f'{feature}\n({feature_type})' for feature, feature_type in zip(top_features, feature_type_labels)]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    
    # 添加图例
    ax.legend()
    
    # 添加关键标注
    # 注意：这里我们假设痰湿质和交叉特征在Top 15中
    # 由于实际数据中可能没有这些，我们先检查
    for i, feature in enumerate(top_features):
        if feature == '痰湿质':
            ax.text(i, top_scores[i] + 0.005, '单一目标高分，\n双目标失效', 
                    ha='center', va='bottom', fontsize=9, color='red')
        # 这里可以添加交叉特征的标注
        # 由于实际数据中可能没有交叉特征，暂时注释
        # if '交叉' in feature or '协同' in feature:
        #     ax.text(i, top_scores[i] + 0.005, '中西医协同效应区', 
        #             ha='center', va='bottom', fontsize=9, color='purple')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_path = '综合评分瀑布图.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到：{output_path}")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()
