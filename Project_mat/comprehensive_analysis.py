
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    # 加载数据
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    # 定义特征
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    # === 第一步：年龄和性别分组分析 ===
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    # === 第二步：关键指标筛选 ===
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    # 构建特征池
    basic_features = blood_indices + activity_indices
    
    # 派生特征
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    # 1. Spearman相关系数（与痰湿体质）
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    # 2. 互信息（与高血脂风险）
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    # 3. PLS联合分析
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target]))
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    # 4. 综合评分（熵权法）
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings))
    
    # 标准化
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
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
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    # 计算综合评分
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    # === 第三步：九种体质对发病风险的贡献度分析 ===
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    # 1. 相对风险分析
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) &gt; 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    # 卡方检验
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value &lt; 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    
    # 2. 基于积分的相关性分析
    print("\n--- 2. 基于体质积分的相关性分析 ---")
    constitution_corr = []
    for const in constitution_types:
        corr, _ = spearmanr(df[const], df[target])
        constitution_corr.append((const, corr))
    
    constitution_corr_sorted = sorted(constitution_corr, key=lambda x: abs(x[1]), reverse=True)
    print("\n九种体质与高血脂的Spearman相关系数（绝对值排序）：")
    for const, corr in constitution_corr_sorted:
        print(f"  {const}: {corr:.4f}")
    
    # 3. 多因素Logistic回归分析
    print("\n--- 3. 多因素Logistic回归净贡献分析 ---")
    control_vars = ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 
                   'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI',
                   '活动量表总分（ADL总分+IADL总分）', '年龄组', '性别']
    
    X_logistic = df[constitution_types + control_vars]
    y_logistic = df[target]
    
    scaler_logistic = StandardScaler()
    X_logistic_scaled = scaler_logistic.fit_transform(X_logistic)
    
    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(X_logistic_scaled, y_logistic)
    
    coefficients = logistic_model.coef_[0]
    feature_names = constitution_types + control_vars
    
    constitution_coefficients = []
    for const in constitution_types:
        idx = feature_names.index(const)
        constitution_coefficients.append((const, coefficients[idx]))
    
    constitution_coefficients_sorted = sorted(constitution_coefficients, 
                                               key=lambda x: abs(x[1]), reverse=True)
    
    print("\n九种体质的标准化回归系数（绝对值排序，控制协变量）：")
    for const, coef in constitution_coefficients_sorted:
        direction = "风险因子" if coef &gt; 0 else "保护因子"
        print(f"  {const}: {coef:.4f} ({direction})")
    
    # === 总结 ===
    print("\n" + "="*80)
    print("分析总结")
    print("="*80)
    
    print("\n1. 关键指标（前5）：")
    for i, indicator in enumerate(key_indicators[:5], 1):
        print(f"   {i}. {indicator}")
    
    print("\n2. 九种体质贡献度排名（基于综合分析）：")
    for i, (const, coef) in enumerate(constitution_coefficients_sorted, 1):
        direction = "↑" if coef &gt; 0 else "↓"
        print(f"   {i}. {const} {direction}")
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression

import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)}
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'L
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣',
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'I
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+I
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质',
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质',
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值:
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender,
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phleg
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*8
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C']
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] /
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] =
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']

import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）']
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL']
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HD
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features =
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spe
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2.
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]

import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y,
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    sc
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    

import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_sc
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))

import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]

import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:,
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(c
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings

import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m =
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.000
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P =
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum =
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:,
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] =
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spear
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")

import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1]
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:,
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,

import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score':
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indic
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    

import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_pre
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_pre
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr =
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(conting
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")

import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    

import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    
    print("\n--- 2. 基于体质积分的相关性分析 ---")
    constitution_c
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    
    print("\n--- 2. 基于体质积分的相关性分析 ---")
    constitution_corr = []
    for const in constitution_types:
        corr, _ = spearmanr
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    
    print("\n--- 2. 基于体质积分的相关性分析 ---")
    constitution_corr = []
    for const in constitution_types:
        corr, _ = spearmanr(df[const], df[target])
        constitution_corr.append((const, corr))
    

import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    
    print("\n--- 2. 基于体质积分的相关性分析 ---")
    constitution_corr = []
    for const in constitution_types:
        corr, _ = spearmanr(df[const], df[target])
        constitution_corr.append((const, corr))
    
    constitution_corr_sorted = sorted(constitution_corr, key=lambda x: abs(x
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    
    print("\n--- 2. 基于体质积分的相关性分析 ---")
    constitution_corr = []
    for const in constitution_types:
        corr, _ = spearmanr(df[const], df[target])
        constitution_corr.append((const, corr))
    
    constitution_corr_sorted = sorted(constitution_corr, key=lambda x: abs(x[1]), reverse=True)
    print("\
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    
    print("\n--- 2. 基于体质积分的相关性分析 ---")
    constitution_corr = []
    for const in constitution_types:
        corr, _ = spearmanr(df[const], df[target])
        constitution_corr.append((const, corr))
    
    constitution_corr_sorted = sorted(constitution_corr, key=lambda x: abs(x[1]), reverse=True)
    print("\n九种体质与高血脂的Spearman相关系数（绝对值排序）：")
    for const, corr in constitution_corr_sorted
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    
    print("\n--- 2. 基于体质积分的相关性分析 ---")
    constitution_corr = []
    for const in constitution_types:
        corr, _ = spearmanr(df[const], df[target])
        constitution_corr.append((const, corr))
    
    constitution_corr_sorted = sorted(constitution_corr, key=lambda x: abs(x[1]), reverse=True)
    print("\n九种体质与高血脂的Spearman相关系数（绝对值排序）：")
    for const, corr in constitution_corr_sorted:
        print(f"  {const}: {corr:.4f}")
    

import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    
    print("\n--- 2. 基于体质积分的相关性分析 ---")
    constitution_corr = []
    for const in constitution_types:
        corr, _ = spearmanr(df[const], df[target])
        constitution_corr.append((const, corr))
    
    constitution_corr_sorted = sorted(constitution_corr, key=lambda x: abs(x[1]), reverse=True)
    print("\n九种体质与高血脂的Spearman相关系数（绝对值排序）：")
    for const, corr in constitution_corr_sorted:
        print(f"  {const}: {corr:.4f}")
    
    print("\n--- 3. 多因素Logistic回归净贡献分析 ---")

import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    
    print("\n--- 2. 基于体质积分的相关性分析 ---")
    constitution_corr = []
    for const in constitution_types:
        corr, _ = spearmanr(df[const], df[target])
        constitution_corr.append((const, corr))
    
    constitution_corr_sorted = sorted(constitution_corr, key=lambda x: abs(x[1]), reverse=True)
    print("\n九种体质与高血脂的Spearman相关系数（绝对值排序）：")
    for const, corr in constitution_corr_sorted:
        print(f"  {const}: {corr:.4f}")
    
    print("\n--- 3. 多因素Logistic回归净贡献分析 ---")
    control_vars = ['TC（总胆固醇）', 'TG（甘油三酯）', 'L
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    
    print("\n--- 2. 基于体质积分的相关性分析 ---")
    constitution_corr = []
    for const in constitution_types:
        corr, _ = spearmanr(df[const], df[target])
        constitution_corr.append((const, corr))
    
    constitution_corr_sorted = sorted(constitution_corr, key=lambda x: abs(x[1]), reverse=True)
    print("\n九种体质与高血脂的Spearman相关系数（绝对值排序）：")
    for const, corr in constitution_corr_sorted:
        print(f"  {const}: {corr:.4f}")
    
    print("\n--- 3. 多因素Logistic回归净贡献分析 ---")
    control_vars = ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 
                   'HDL-C（高密度脂蛋白）', '空腹
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    
    print("\n--- 2. 基于体质积分的相关性分析 ---")
    constitution_corr = []
    for const in constitution_types:
        corr, _ = spearmanr(df[const], df[target])
        constitution_corr.append((const, corr))
    
    constitution_corr_sorted = sorted(constitution_corr, key=lambda x: abs(x[1]), reverse=True)
    print("\n九种体质与高血脂的Spearman相关系数（绝对值排序）：")
    for const, corr in constitution_corr_sorted:
        print(f"  {const}: {corr:.4f}")
    
    print("\n--- 3. 多因素Logistic回归净贡献分析 ---")
    control_vars = ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 
                   'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI',
                   '活动量表总分（ADL总分+I
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    
    print("\n--- 2. 基于体质积分的相关性分析 ---")
    constitution_corr = []
    for const in constitution_types:
        corr, _ = spearmanr(df[const], df[target])
        constitution_corr.append((const, corr))
    
    constitution_corr_sorted = sorted(constitution_corr, key=lambda x: abs(x[1]), reverse=True)
    print("\n九种体质与高血脂的Spearman相关系数（绝对值排序）：")
    for const, corr in constitution_corr_sorted:
        print(f"  {const}: {corr:.4f}")
    
    print("\n--- 3. 多因素Logistic回归净贡献分析 ---")
    control_vars = ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 
                   'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI',
                   '活动量表总分（ADL总分+IADL总分）', '年龄组', '性别']
    
    X_logistic = df[
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    
    print("\n--- 2. 基于体质积分的相关性分析 ---")
    constitution_corr = []
    for const in constitution_types:
        corr, _ = spearmanr(df[const], df[target])
        constitution_corr.append((const, corr))
    
    constitution_corr_sorted = sorted(constitution_corr, key=lambda x: abs(x[1]), reverse=True)
    print("\n九种体质与高血脂的Spearman相关系数（绝对值排序）：")
    for const, corr in constitution_corr_sorted:
        print(f"  {const}: {corr:.4f}")
    
    print("\n--- 3. 多因素Logistic回归净贡献分析 ---")
    control_vars = ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 
                   'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI',
                   '活动量表总分（ADL总分+IADL总分）', '年龄组', '性别']
    
    X_logistic = df[constitution_types + control_vars]
    y_logistic = df[target]
    
    scaler
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    
    print("\n--- 2. 基于体质积分的相关性分析 ---")
    constitution_corr = []
    for const in constitution_types:
        corr, _ = spearmanr(df[const], df[target])
        constitution_corr.append((const, corr))
    
    constitution_corr_sorted = sorted(constitution_corr, key=lambda x: abs(x[1]), reverse=True)
    print("\n九种体质与高血脂的Spearman相关系数（绝对值排序）：")
    for const, corr in constitution_corr_sorted:
        print(f"  {const}: {corr:.4f}")
    
    print("\n--- 3. 多因素Logistic回归净贡献分析 ---")
    control_vars = ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 
                   'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI',
                   '活动量表总分（ADL总分+IADL总分）', '年龄组', '性别']
    
    X_logistic = df[constitution_types + control_vars]
    y_logistic = df[target]
    
    scaler_logistic = StandardScaler()
    X_logistic_scaled = scaler_logistic.fit
import pandas as pd
import numpy as np
import warnings
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.constants import AGE_CONSTRAINTS

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("痰湿体质与高血脂发病风险综合分析")
    print("="*80)
    
    df = pd.read_excel('data/raw/附件1：样例数据.xlsx')
    print(f"\n数据加载完成，共 {len(df)} 个样本")
    
    blood_indices = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 
                    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    
    activity_indices = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
                       'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
                       '活动量表总分（ADL总分+IADL总分）']
    
    constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', 
                         '湿热质', '血瘀质', '气郁质', '特禀质']
    
    target = '高血脂症二分类标签'
    target_phlegm = '痰湿质'
    
    print("\n" + "="*80)
    print("第一步：年龄和性别分组分析")
    print("="*80)
    
    age_group_map = AGE_CONSTRAINTS['age_groups']
    print(f"\n年龄分组定义: {age_group_map}")
    
    print("\n--- 按年龄组分析 ---")
    for age_group in sorted(df['年龄组'].unique()):
        group = df[df['年龄组'] == age_group]
        print(f"\n年龄组 {age_group} ({age_group_map.get(age_group, '未知')}):")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
        print(f"  BMI平均值: {group['BMI'].mean():.2f}")
    
    print("\n--- 按性别分析 ---")
    gender_map = {1: '男', 0: '女'}
    for gender in sorted(df['性别'].unique()):
        group = df[df['性别'] == gender]
        print(f"\n{gender_map.get(gender, '未知')}:")
        print(f"  样本数: {len(group)}")
        print(f"  高血脂患病率: {group[target].mean():.4f}")
        print(f"  痰湿质平均得分: {group[target_phlegm].mean():.2f}")
    
    print("\n" + "="*80)
    print("第二步：关键指标筛选")
    print("="*80)
    
    basic_features = blood_indices + activity_indices
    
    df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
    df['AIP'] = np.log(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
    df['TC/HDL'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
    df['LDL/HDL'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
    df['TG/HDL'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
    
    derived_features = ['non-HDL-C', 'AIP', 'TC/HDL', 'LDL/HDL', 'TG/HDL']
    
    all_features = basic_features + derived_features
    
    print("\n--- 1. 与痰湿体质的Spearman相关性 ---")
    spearman_scores = []
    for feature in all_features:
        corr, _ = spearmanr(df[feature], df[target_phlegm])
        spearman_scores.append(abs(corr))
    
    spearman_df = pd.DataFrame({
        'feature': all_features,
        'spearman_corr': spearman_scores
    }).sort_values('spearman_corr', ascending=False)
    
    print("\n与痰湿体质相关性最高的前10个指标：")
    print(spearman_df.head(10).to_string(index=False))
    
    print("\n--- 2. 与高血脂风险的互信息 ---")
    X = df[all_features]
    y = df[target]
    mutual_info_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': all_features,
        'mutual_info': mutual_info_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n与高血脂风险互信息最高的前10个指标：")
    print(mi_df.head(10).to_string(index=False))
    
    print("\n--- 3. PLS联合结构载荷分析 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_combined = np.column_stack((df[target_phlegm], df[target])
    
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y_combined)
    
    pls_loadings = np.zeros(len(all_features))
    for i in range(len(all_features)):
        corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
        corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
        pls_loadings[i] = (abs(corr1) + abs(corr2)) / 2
    
    pls_df = pd.DataFrame({
        'feature': all_features,
        'pls_loading': pls_loadings
    }).sort_values('pls_loading', ascending=False)
    
    print("\nPLS联合载荷最高的前10个指标：")
    print(pls_df.head(10).to_string(index=False))
    
    print("\n--- 4. 综合评分（熵权法） ---")
    n = len(all_features)
    m = 3
    
    X_matrix = np.column_stack((spearman_scores, mutual_info_scores, pls_loadings)
    
    def normalize_matrix(X):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            min_val = np.min(X[:, j])
            max_val = np.max(X[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        return X_norm
    
    X_norm = normalize_matrix(X_matrix)
    
    P = np.zeros_like(X_norm)
    for j in range(m):
        col_sum = np.sum(X_norm[:, j])
        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X_norm[:, j] / col_sum
    
    e = np.zeros(m)
    k = 1 / np.log(n)
    for j in range(m):
        entropy = 0
        for i in range(n):
            if P[i, j] > 0:
                entropy += P[i, j] * np.log(P[i, j])
        e[j] = -k * entropy
    
    d = 1 - e
    w = d / np.sum(d)
    
    print(f"\n各维度权重：")
    print(f"  Spearman相关（痰湿）: {w[0]:.4f}")
    print(f"  互信息（风险）: {w[1]:.4f}")
    print(f"  PLS载荷（联合）: {w[2]:.4f}")
    
    total_scores = w[0] * X_norm[:, 0] + w[1] * X_norm[:, 1] + w[2] * X_norm[:, 2]
    
    feature_scores = pd.DataFrame({
        'feature': all_features,
        'spearman_score': spearman_scores,
        'mutual_info_score': mutual_info_scores,
        'pls_loading': pls_loadings,
        'total_score': total_scores
    }).sort_values('total_score', ascending=False)
    
    print("\n关键指标综合排名（前15）：")
    print(feature_scores.head(15).to_string(index=False))
    
    key_indicators = feature_scores.head(10)['feature'].tolist()
    print(f"\n能有效表征痰湿体质严重程度且能预警高血脂发病风险的关键指标：")
    for i, indicator in enumerate(key_indicators, 1):
        print(f"  {i}. {indicator}")
    
    print("\n" + "="*80)
    print("第三步：九种体质对发病风险的贡献度分析")
    print("="*80)
    
    print("\n--- 1. 基于体质标签的相对风险分析 ---")
    total_prevalence = df[target].mean()
    print(f"总体高血脂患病率: {total_prevalence:.4f}")
    
    relative_risks = []
    for i in range(1, 10):
        subset = df[df['体质标签'] == i]
        if len(subset) > 0:
            prevalence = subset[target].mean()
            rr = prevalence / total_prevalence
        else:
            rr = 1.0
        relative_risks.append(rr)
        print(f"{constitution_types[i-1]}: 患病率={prevalence:.4f}, 相对风险={rr:.4f}")
    
    contingency_table = pd.crosstab(df['体质标签'], df[target])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\n卡方检验: chi2={chi2:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("体质标签与高血脂发病风险存在统计学显著关联")
    else:
        print("体质标签与高血脂发病风险无统计学显著关联")
    
    print("\n--- 2. 基于体质积分的相关性分析 ---")
    constitution_corr = []
    for const in constitution_types:
        corr, _ = spearmanr(df[const], df[target])
        constitution_corr.append((const, corr))
    
    constitution_corr_sorted = sorted(constitution_corr, key=lambda x: abs(x[1]), reverse=True)
    print("\n九种体质与高血脂的Spearman相关系数（绝对值排序）：")
    for const, corr in constitution_corr_sorted:
        print(f"  {const}: {corr:.4f}")
    
    print("\n--- 3. 多因素Logistic回归净贡献分析 ---")
    control_vars = ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 
                   'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI',
                   '活动量表总分（ADL总分+IADL总分）', '年龄组', '性别']
    
    X_logistic = df[constitution_types + control_vars]
    y_logistic = df[target]
    
    scaler_logistic = StandardScaler()
    X_logistic_scaled = scaler_logistic.fit_transform(X_logistic)
    
    logistic_model = LogisticRegression(random_state=42)
