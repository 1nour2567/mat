# 特征池构建与筛选模块
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from scipy.stats import spearmanr, chi2_contingency
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

def build_feature_pool(df):
    """
    5.1.2 候选特征池构建
    融合原始指标与派生指标，构建三级候选特征集：
    •基础特征层：九种体质积分+体质标签、TC/TG/LDL-C/HDL-C/GLU/UA/BMI、ADL/IADL总分及分项、人口统计学信息
    •派生特征层：non-HDL-C、AIP、TC/HDL、LDL/HDL、TG/HDL、血脂异常项数、尿酸异常标志
    •中西医交叉特征层：痰湿质得分×BMI、痰湿质得分×TG、痰湿质得分×AIP、痰湿质得分×LDL-C、痰湿质得分/HDL-C、气虚质得分×TC
    """
    print("\n=== 5.1.2 构建三级候选特征池 ===")
    
    # 确保数据中已有基础特征
    base_features = []
    
    # 九种体质积分
    constitution_scores = ['平和质得分', '气虚质得分', '阳虚质得分', '阴虚质得分', 
                         '痰湿质得分', '湿热质得分', '血瘀质得分', '气郁质得分', '特禀质得分']
    for col in constitution_scores:
        if col in df.columns:
            base_features.append(col)
    
    # 体质标签（如果存在）
    if '体质标签' in df.columns:
        base_features.append('体质标签')
    
    # 血脂、血糖、尿酸、BMI指标
    lab_features = ['TC', 'TG', 'LDL-C', 'HDL-C', 'GLU', 'UA', 'BMI']
    for col in lab_features:
        if col in df.columns:
            base_features.append(col)
    
    # ADL/IADL总分及分项
    adl_features = ['ADL总分', 'IADL总分']
    for col in adl_features:
        if col in df.columns:
            base_features.append(col)
    
    # 人口统计学信息
    demo_features = ['age', 'gender']
    for col in demo_features:
        if col in df.columns:
            base_features.append(col)
    
    # 派生特征层（确保已在01_preprocessing.py中生成，这里只添加到特征列表）
    derived_features = []
    
    # 派生特征：non-HDL-C、AIP、TC/HDL、LDL/HDL、TG/HDL
    derived_names = ['non-HDL-C', 'AIP', 'TC/HDL比值', 'LDL/HDL比值', 'TG/HDL比值']
    for col in derived_names:
        if col in df.columns:
            derived_features.append(col)
    
    # 血脂异常项数、尿酸异常标志
    if '血脂异常项数' in df.columns:
        derived_features.append('血脂异常项数')
    if '尿酸异常' in df.columns:
        derived_features.append('尿酸异常')
    
    # 中西医交叉特征层
    cross_features = []
    
    # 痰湿质得分×BMI
    if '痰湿质得分' in df.columns and 'BMI' in df.columns:
        df['痰湿质得分×BMI'] = df['痰湿质得分'] * df['BMI']
        cross_features.append('痰湿质得分×BMI')
    
    # 痰湿质得分×TG
    if '痰湿质得分' in df.columns and 'TG' in df.columns:
        df['痰湿质得分×TG'] = df['痰湿质得分'] * df['TG']
        cross_features.append('痰湿质得分×TG')
    
    # 痰湿质得分×AIP
    if '痰湿质得分' in df.columns and 'AIP' in df.columns:
        df['痰湿质得分×AIP'] = df['痰湿质得分'] * df['AIP']
        cross_features.append('痰湿质得分×AIP')
    
    # 痰湿质得分×LDL-C
    if '痰湿质得分' in df.columns and 'LDL-C' in df.columns:
        df['痰湿质得分×LDL-C'] = df['痰湿质得分'] * df['LDL-C']
        cross_features.append('痰湿质得分×LDL-C')
    
    # 痰湿质得分/HDL-C
    if '痰湿质得分' in df.columns and 'HDL-C' in df.columns:
        df['痰湿质得分/HDL-C'] = df['痰湿质得分'] / df['HDL-C']
        cross_features.append('痰湿质得分/HDL-C')
    
    # 气虚质得分×TC
    if '气虚质得分' in df.columns and 'TC' in df.columns:
        df['气虚质得分×TC'] = df['气虚质得分'] * df['TC']
        cross_features.append('气虚质得分×TC')
    
    # 合并所有特征
    all_features = base_features + derived_features + cross_features
    
    # 确保只保留存在的数值型特征
    numeric_features = []
    for feature in all_features:
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
            numeric_features.append(feature)
    
    print(f"基础特征层: {len(base_features)}个")
    print(f"派生特征层: {len(derived_features)}个")
    print(f"中西医交叉特征层: {len(cross_features)}个")
    print(f"总候选特征数: {len(numeric_features)}个")
    
    return df, numeric_features

def handle_class_imbalance(X, y):
    """处理类别不平衡"""
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)
    return X_res, y_res

def calculate_spearman_correlation(df, features, phlegm_score_col='痰湿质得分'):
    """
    5.1.3 - 1. Spearman相关系数（痰湿表征能力）
    采用Spearman等级相关系数衡量各指标与痰湿质积分的关联强度，取绝对值，值越大表征能力越强。
    """
    correlations = {}
    for feature in features:
        if feature != phlegm_score_col and feature in df.columns:
            valid_data = df[[feature, phlegm_score_col]].dropna()
            if len(valid_data) > 2:
                corr, _ = spearmanr(valid_data[feature], valid_data[phlegm_score_col])
                correlations[feature] = abs(corr)
            else:
                correlations[feature] = 0.0
    return correlations

def calculate_mutual_info(df, features, target):
    """
    5.1.3 - 2. 互信息（风险预警能力）
    采用互信息衡量各指标包含的高血脂发病信息量，值越大预警能力越强。
    """
    valid_features = [f for f in features if f in df.columns]
    X = df[valid_features].fillna(0)
    y = df[target]
    
    mi = mutual_info_classif(X, y, random_state=42)
    mi_dict = {valid_features[i]: mi[i] for i in range(len(valid_features))}
    return mi_dict

def calculate_pls_loadings(df, features, phlegm_score_col='痰湿质得分', target=None):
    """
    5.1.3 - 3. PLS联合结构载荷（双目标整合能力）
    以"痰湿质积分+高血脂标签"为双响应矩阵进行偏最小二乘回归，
    计算各变量在前两个潜变量上的联合结构载荷，反映指标在双目标空间的综合解释力。
    """
    valid_features = [f for f in features if f in df.columns and f != phlegm_score_col]
    
    if len(valid_features) < 2:
        return {f: 0.0 for f in valid_features}
    
    X = df[valid_features].fillna(0)
    Y = df[[phlegm_score_col, target]].fillna(0)
    
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
    loading_scores = {valid_features[i]: np.sum(np.abs(loadings[i, :])) for i in range(len(valid_features))}
    
    return loading_scores

def entropy_weight_method(spearman_scores, mi_scores, pls_scores):
    """
    5.1.3 - 4. 熵权法（EWM）计算三维度权重
    按照严格的数学步骤：
    第一步：矩阵标准化（极差标准化）
    第二步：计算特征在各维度下的比重 p_ij
    第三步：计算各维度的信息熵 e_j
    第四步：计算信息冗余度（差异性系数）d_j
    第五步：计算最终权重 w_j
    """
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
    
    weights = {
        'spearman': w[0],
        'mi': w[1],
        'pls': w[2]
    }
    
    print("\n=== 熵权法（EWM）计算结果 ===")
    print(f"Spearman相关系数权重: {weights['spearman']:.4f}")
    print(f"互信息权重: {weights['mi']:.4f}")
    print(f"PLS联合结构载荷权重: {weights['pls']:.4f}")
    
    return weights

def select_features(df, features, target, phlegm_score_col='痰湿质得分', k=20):
    """
    5.1.3 基于Spearman-互信息-PLS的双目标联合筛选模型
    针对"痰湿表征+风险预警"双目标要求，构建三维综合评分体系
    """
    print("\n=== 5.1.3 基于Spearman-互信息-PLS的双目标联合筛选 ===")
    
    # 1. 计算各指标
    spearman_scores = calculate_spearman_correlation(df, features, phlegm_score_col)
    mi_scores = calculate_mutual_info(df, features, target)
    pls_scores = calculate_pls_loadings(df, features, phlegm_score_col, target)
    
    # 2. 熵权法计算权重
    weights = entropy_weight_method(spearman_scores, mi_scores, pls_scores)
    
    # 3. 综合评分
    common_features = list(set(spearman_scores.keys()) & set(mi_scores.keys()) & set(pls_scores.keys()))
    combined_scores = {}
    
    for feature in common_features:
        score = (weights['spearman'] * spearman_scores[feature] +
                weights['mi'] * mi_scores[feature] +
                weights['pls'] * pls_scores[feature])
        combined_scores[feature] = score
    
    # 排序并选择前k个特征
    sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    selected_features = [f[0] for f in sorted_features[:k]]
    
    # 打印前10个关键指标
    print("\n前10个关键指标：")
    for i, (feature, score) in enumerate(sorted_features[:10]):
        print(f"{i+1}. {feature}: {score:.4f}")
    
    return selected_features

def analyze_constitution_contribution(df, target):
    """
    5.1.4 九种体质风险贡献度分析
    从离散标签和连续积分两个维度展开：
    1.基于主标签的相对风险分析：计算各类体质的高血脂确诊率与总体患病率的比值，
      并通过卡方独立性检验验证体质标签与发病风险的统计学关联。
    2.基于积分的多因素贡献分析：构建控制血脂、活动能力、人口统计学因素的Logistic回归模型，
      通过标准化回归系数的绝对值衡量不同体质维度的净贡献度。
    """
    print("\n=== 5.1.4 九种体质风险贡献度分析 ===")
    
    constitution_scores = ['平和质得分', '气虚质得分', '阳虚质得分', '阴虚质得分', 
                         '痰湿质得分', '湿热质得分', '血瘀质得分', '气郁质得分', '特禀质得分']
    
    # 1. 基于主标签的相对风险分析
    print("\n1. 基于主标签的相对风险分析：")
    if '体质标签' in df.columns:
        overall_prevalence = df[target].mean()
        print(f"总体患病率: {overall_prevalence:.4f}")
        
        # 计算各类体质的患病率和相对风险
        contingency_table = []
        constitutions = []
        
        for constitution in df['体质标签'].unique():
            subset = df[df['体质标签'] == constitution]
            if len(subset) > 0:
                prevalence = subset[target].mean()
                relative_risk = prevalence / overall_prevalence if overall_prevalence > 0 else 0
                
                cases = subset[target].sum()
                non_cases = len(subset) - cases
                contingency_table.append([cases, non_cases])
                constitutions.append(constitution)
                
                print(f"{constitution}: 患病率={prevalence:.4f}, 相对风险={relative_risk:.4f}, n={len(subset)}")
        
        # 卡方独立性检验
        if len(contingency_table) >= 2:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            print(f"\n卡方独立性检验: chi2={chi2:.4f}, p-value={p_value:.4f}, dof={dof}")
            if p_value < 0.05:
                print("结论: 体质标签与发病风险存在统计学显著关联 (p<0.05)")
            else:
                print("结论: 体质标签与发病风险无统计学显著关联")
    
    # 2. 基于积分的多因素贡献分析
    print("\n2. 基于积分的多因素贡献分析：")
    
    # 构建控制变量：血脂、活动能力、人口统计学因素
    control_vars = []
    
    # 血脂指标
    lipid_vars = ['TC', 'TG', 'LDL-C', 'HDL-C']
    for var in lipid_vars:
        if var in df.columns:
            control_vars.append(var)
    
    # 活动能力
    if 'ADL总分' in df.columns:
        control_vars.append('ADL总分')
    
    # 人口统计学
    demo_vars = ['age']
    for var in demo_vars:
        if var in df.columns:
            control_vars.append(var)
    
    # 体质积分
    valid_constitution_scores = [cs for cs in constitution_scores if cs in df.columns]
    
    if len(valid_constitution_scores) > 0 and len(control_vars) > 0:
        # 准备特征矩阵
        all_features = control_vars + valid_constitution_scores
        X = df[all_features].fillna(0)
        y = df[target]
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Logistic回归
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_scaled, y)
        
        # 提取标准化回归系数
        coefficients = lr.coef_[0]
        feature_coef = dict(zip(all_features, coefficients))
        
        # 只显示体质积分的贡献
        print("\n体质维度净贡献度（标准化回归系数绝对值）：")
        constitution_contribution = {}
        for cs in valid_constitution_scores:
            if cs in feature_coef:
                contribution = abs(feature_coef[cs])
                constitution_contribution[cs] = contribution
        
        # 排序
        sorted_contribution = sorted(constitution_contribution.items(), key=lambda x: x[1], reverse=True)
        for cs, contrib in sorted_contribution:
            print(f"{cs}: {contrib:.4f}")
        
        return sorted_contribution
    
    return []

def feature_engineering(input_path, output_path, target):
    """完整特征工程流程"""
    # 加载预处理后的数据
    df = pd.read_pickle(input_path)
    
    # 5.1.2 构建特征池
    df, features = build_feature_pool(df)
    
    # 5.1.3 特征筛选
    selected_features = select_features(df, features, target)
    
    # 5.1.4 体质贡献度分析
    analyze_constitution_contribution(df, target)
    
    # 标准化
    scaler = StandardScaler()
    df[selected_features] = scaler.fit_transform(df[selected_features])
    
    # 保存处理后的数据
    df.to_pickle(output_path)
    
    return df, selected_features
