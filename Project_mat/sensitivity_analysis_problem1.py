import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载真实数据
def load_real_data():
    """加载真实数据"""
    import pickle
    data = pickle.load(open('/workspace/Project_mat/data/processed/preprocessed_data.pkl', 'rb'))
    
    # 重命名列以匹配分析需求
    rename_map = {
        'TC（总胆固醇）': 'TC',
        'TG（甘油三酯）': 'TG',
        'HDL-C（高密度脂蛋白）': 'HDL-C',
        'LDL-C（低密度脂蛋白）': 'LDL-C',
        '活动量表总分（ADL总分+IADL总分）': '活动量表总分'
    }
    data = data.rename(columns=rename_map)
    
    # 处理年龄数据
    if '年龄' not in data.columns:
        # 如果年龄组是整数，直接使用
        if pd.api.types.is_integer_dtype(data['年龄组']):
            data['年龄'] = data['年龄组'].astype(float)
        else:
            # 否则尝试提取数字
            data['年龄'] = data['年龄组'].astype(str).str.extract('(\d+)').astype(float)
    
    return data

# 计算综合评分
def calculate_comprehensive_score(data, weights=None):
    """计算综合评分"""
    if weights is None:
        # 默认权重
        weights = {
            'spearman': 0.1507,
            'mutual_info': 0.7091,
            'pls': 0.1401
        }
    
    # 计算Spearman相关系数（与痰湿质）
    spearman_scores = {}
    for col in ['TC', 'TG', '血尿酸', 'ADL总分', '活动量表总分', 'HDL-C', 'LDL-C', '空腹血糖', 'BMI']:
        corr, _ = spearmanr(data[col], data['痰湿质'])
        spearman_scores[col] = abs(corr)
    
    # 计算互信息（与高血脂）
    X = data[['TC', 'TG', '血尿酸', 'ADL总分', '活动量表总分', 'HDL-C', 'LDL-C', '空腹血糖', 'BMI']]
    y = data['高血脂症二分类标签']
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_dict = dict(zip(X.columns, mi_scores))
    
    # 计算PLS联合结构载荷
    pls = PLSRegression(n_components=1)
    X_pls = data[['TC', 'TG', '血尿酸', 'ADL总分', '活动量表总分', 'HDL-C', 'LDL-C', '空腹血糖', 'BMI']]
    Y_pls = data[['痰湿质', '高血脂症二分类标签']]
    pls.fit(X_pls, Y_pls)
    pls_scores = np.abs(pls.x_loadings_[:, 0])
    pls_dict = dict(zip(X_pls.columns, pls_scores))
    
    # 归一化
    def normalize(d):
        s = sum(d.values())
        return {k: v/s for k, v in d.items()}
    
    spearman_norm = normalize(spearman_scores)
    mi_norm = normalize(mi_dict)
    pls_norm = normalize(pls_dict)
    
    # 计算综合评分
    comprehensive = {}
    for col in X.columns:
        comprehensive[col] = (
            weights['spearman'] * spearman_norm.get(col, 0) +
            weights['mutual_info'] * mi_norm.get(col, 0) +
            weights['pls'] * pls_norm.get(col, 0)
        )
    
    # 归一化综合评分
    total = sum(comprehensive.values())
    comprehensive = {k: v/total for k, v in comprehensive.items()}
    
    return comprehensive, spearman_scores, mi_dict, pls_dict

# 敏感性分析
def sensitivity_analysis():
    """进行敏感性分析"""
    # 加载真实数据
    data = load_real_data()
    
    # 1. 权重敏感性分析
    print("1. 权重敏感性分析")
    weight_variations = [
        {'spearman': 0.10, 'mutual_info': 0.75, 'pls': 0.15},  # 增加互信息权重
        {'spearman': 0.20, 'mutual_info': 0.65, 'pls': 0.15},  # 增加Spearman权重
        {'spearman': 0.15, 'mutual_info': 0.60, 'pls': 0.25},  # 增加PLS权重
        {'spearman': 0.33, 'mutual_info': 0.33, 'pls': 0.34},  # 等权重
    ]
    
    weight_results = []
    for i, weights in enumerate(weight_variations):
        scores, _, _, _ = calculate_comprehensive_score(data, weights)
        top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        weight_results.append((i, weights, top5))
        print(f"  权重设置 {i+1}: {weights}")
        print(f"  前5指标: {top5}")
    
    # 2. 数据噪声敏感性分析
    print("\n2. 数据噪声敏感性分析")
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    noise_results = []
    
    for noise in noise_levels:
        noisy_data = data.copy()
        # 为关键指标添加噪声
        for col in ['TC', 'TG', '血尿酸', 'ADL总分']:
            noisy_data[col] = noisy_data[col] * (1 + np.random.normal(0, noise, len(noisy_data)))
        
        scores, _, _, _ = calculate_comprehensive_score(noisy_data)
        top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        noise_results.append((noise, top5))
        print(f"  噪声水平 {noise}: 前5指标: {top5}")
    
    # 3. 性别分组敏感性分析
    print("\n3. 性别分组敏感性分析")
    male_data = data[data['性别'] == 0]
    female_data = data[data['性别'] == 1]
    
    male_scores, _, _, _ = calculate_comprehensive_score(male_data)
    female_scores, _, _, _ = calculate_comprehensive_score(female_data)
    
    male_top5 = sorted(male_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    female_top5 = sorted(female_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"  男性前5指标: {male_top5}")
    print(f"  女性前5指标: {female_top5}")
    
    # 4. 年龄分组敏感性分析
    print("\n4. 年龄分组敏感性分析")
    # 年龄组映射
    age_group_map = {
        1: "40-49岁",
        2: "50-59岁",
        3: "60-69岁",
        4: "70-79岁",
        5: "80-89岁"
    }
    
    age_results = []
    for age_code, label in age_group_map.items():
        age_data = data[data['年龄组'] == age_code]
        if len(age_data) > 10:
            scores, _, _, _ = calculate_comprehensive_score(age_data)
            top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            age_results.append((label, top5))
            print(f"  {label}前5指标: {top5}")
    
    return {
        'weight_results': weight_results,
        'noise_results': noise_results,
        'gender_results': (male_top5, female_top5),
        'age_results': age_results
    }

# 生成敏感性分析图表
def generate_sensitivity_charts(results):
    """生成敏感性分析图表"""
    # 1. 权重敏感性分析图表
    plt.figure(figsize=(15, 10))
    
    # 权重变化对指标排名的影响
    weight_settings = results['weight_results']
    metrics = ['TC', 'TG', '血尿酸', 'ADL总分', '活动量表总分', 'HDL-C', 'LDL-C', '空腹血糖', 'BMI']
    
    # 准备数据
    weight_data = []
    for i, (idx, weights, top5) in enumerate(weight_settings):
        weight_label = f"设置{i+1}"
        for metric in metrics:
            score = 0
            for item in top5:
                if item[0] == metric:
                    score = item[1]
                    break
            weight_data.append({
                '权重设置': weight_label,
                '指标': metric,
                '综合评分': score
            })
    
    weight_df = pd.DataFrame(weight_data)
    
    plt.subplot(2, 2, 1)
    sns.barplot(data=weight_df, x='指标', y='综合评分', hue='权重设置')
    plt.title('不同权重设置对指标综合评分的影响')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='权重设置')
    plt.tight_layout()
    
    # 2. 噪声敏感性分析图表
    noise_results = results['noise_results']
    noise_data = []
    for noise, top5 in noise_results:
        for metric in metrics:
            score = 0
            for item in top5:
                if item[0] == metric:
                    score = item[1]
                    break
            noise_data.append({
                '噪声水平': f"{noise*100}%",
                '指标': metric,
                '综合评分': score
            })
    
    noise_df = pd.DataFrame(noise_data)
    
    plt.subplot(2, 2, 2)
    sns.lineplot(data=noise_df, x='噪声水平', y='综合评分', hue='指标', marker='o')
    plt.title('数据噪声对指标综合评分的影响')
    plt.xticks(rotation=45)
    plt.legend(title='指标', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 3. 性别差异分析图表
    male_top5, female_top5 = results['gender_results']
    gender_data = []
    
    for item in male_top5:
        gender_data.append({'性别': '男性', '指标': item[0], '综合评分': item[1]})
    for item in female_top5:
        gender_data.append({'性别': '女性', '指标': item[0], '综合评分': item[1]})
    
    gender_df = pd.DataFrame(gender_data)
    
    plt.subplot(2, 2, 3)
    sns.barplot(data=gender_df, x='指标', y='综合评分', hue='性别')
    plt.title('性别差异对指标重要性的影响')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='性别')
    plt.tight_layout()
    
    # 4. 年龄组差异分析图表
    age_results = results['age_results']
    age_data = []
    
    for label, top5 in age_results:
        for item in top5:
            age_data.append({'年龄组': label, '指标': item[0], '综合评分': item[1]})
    
    age_df = pd.DataFrame(age_data)
    
    plt.subplot(2, 2, 4)
    sns.barplot(data=age_df, x='年龄组', y='综合评分', hue='指标')
    plt.title('年龄组差异对指标重要性的影响')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='指标', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('/workspace/Project_mat/sensitivity_analysis_problem1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n敏感性分析图表已保存到 /workspace/Project_mat/sensitivity_analysis_problem1.png")

# 主函数
def main():
    print("开始问题一的敏感性分析...")
    results = sensitivity_analysis()
    generate_sensitivity_charts(results)
    print("\n敏感性分析完成！")

if __name__ == "__main__":
    main()
