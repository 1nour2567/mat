import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据"""
    try:
        df = pd.read_csv('/workspace/Project_mat/预处理后数据.txt', sep='\t', encoding='gbk')
    except:
        df = pd.read_csv('/workspace/Project_mat/预处理后数据.txt', sep='\t', encoding='gb18030')
    return df

def relative_risk_analysis(df):
    """基于主标签的相对风险分析"""
    print("\n" + "="*100)
    print("5.1.4 九种体质风险贡献度分析 - 第一部分：基于主标签的相对风险分析")
    print("="*100)
    
    constitution_names = {
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
    
    overall_prevalence = df['高血脂症二分类标签'].mean()
    print(f"\n总体高血脂患病率: {overall_prevalence*100:.2f}%")
    print(f"总样本数: {len(df)}")
    
    results = []
    
    for label in sorted(df['体质标签'].unique()):
        name = constitution_names[label]
        subset = df[df['体质标签'] == label]
        
        n_total = len(subset)
        n_cases = subset['高血脂症二分类标签'].sum()
        prevalence = subset['高血脂症二分类标签'].mean()
        relative_risk = prevalence / overall_prevalence
        
        results.append({
            '体质类型': name,
            '样本数': n_total,
            '确诊数': n_cases,
            '患病率': prevalence,
            '相对风险': relative_risk
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('相对风险', ascending=False)
    result_df['排名'] = range(1, len(result_df)+1)
    result_df = result_df[['排名', '体质类型', '样本数', '确诊数', '患病率', '相对风险']]
    
    print("\n各类体质高血脂患病情况：")
    pd.options.display.float_format = '{:.4f}'.format
    print(result_df.to_string(index=False))
    
    print("\n卡方独立性检验：")
    contingency_table = pd.crosstab(df['体质标签'], df['高血脂症二分类标签'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"卡方值: {chi2:.4f}")
    print(f"自由度: {dof}")
    print(f"P值: {p_value:.6f}")
    if p_value < 0.05:
        print("结论: 体质标签与高血脂发病风险存在统计学显著关联 (p < 0.05)")
    else:
        print("结论: 体质标签与高血脂发病风险无统计学显著关联 (p >= 0.05)")
    
    plt.figure(figsize=(12, 6))
    colors = ['red' if rr > 1 else 'blue' for rr in result_df['相对风险']]
    bars = plt.bar(result_df['体质类型'], result_df['相对风险'], color=colors, alpha=0.7)
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='总体平均')
    plt.title('各类体质相对风险（相对于总体平均）', fontsize=14, fontweight='bold')
    plt.ylabel('相对风险', fontsize=12)
    plt.xlabel('体质类型', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('/workspace/Project_mat/体质相对风险分析.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n可视化图表已保存：体质相对风险分析.png")
    
    return result_df

def logistic_regression_analysis(df):
    """基于积分的多因素贡献分析"""
    print("\n" + "="*100)
    print("5.1.4 九种体质风险贡献度分析 - 第二部分：基于积分的多因素贡献分析")
    print("="*100)
    
    constitution_features = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
    
    blood_lipid_features = [
        'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 'TC（总胆固醇）'
    ]
    
    activity_features = [
        'ADL总分', 'IADL总分'
    ]
    
    demographic_features = ['年龄组', '性别']
    
    print("\n模型变量说明：")
    print(f"- 体质积分: {len(constitution_features)}个变量")
    print(f"- 血脂因素: {len(blood_lipid_features)}个变量")
    print(f"- 活动能力: {len(activity_features)}个变量")
    print(f"- 人口统计: {len(demographic_features)}个变量")
    
    feature_groups = {
        '体质积分': constitution_features,
        '血脂因素': blood_lipid_features,
        '活动能力': activity_features,
        '人口统计': demographic_features
    }
    
    X = df[constitution_features + blood_lipid_features + activity_features + demographic_features].copy()
    y = df['高血脂症二分类标签']
    
    X = X.dropna()
    y = y.loc[X.index]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    
    coefficients = model.coef_[0]
    feature_names = X.columns.tolist()
    
    coef_df = pd.DataFrame({
        '特征名称': feature_names,
        '标准化系数': coefficients,
        '系数绝对值': np.abs(coefficients)
    })
    
    print("\nLogistic回归模型结果：")
    print(f"模型准确率: {model.score(X_scaled, y)*100:.2f}%")
    
    print("\n所有特征标准化系数（按绝对值排序）：")
    coef_df_sorted = coef_df.sort_values('系数绝对值', ascending=False)
    coef_df_sorted['排名'] = range(1, len(coef_df_sorted)+1)
    coef_df_sorted = coef_df_sorted[['排名', '特征名称', '标准化系数', '系数绝对值']]
    pd.options.display.float_format = '{:.6f}'.format
    print(coef_df_sorted.to_string(index=False))
    
    print("\n按特征分组的贡献度：")
    group_contributions = []
    for group_name, features in feature_groups.items():
        group_coef = coef_df[coef_df['特征名称'].isin(features)]
        total_abs = group_coef['系数绝对值'].sum()
        mean_abs = group_coef['系数绝对值'].mean()
        max_abs = group_coef['系数绝对值'].max()
        group_contributions.append({
            '特征组': group_name,
            '特征数': len(features),
            '系数绝对值总和': total_abs,
            '系数绝对值均值': mean_abs,
            '系数绝对值最大值': max_abs
        })
    
    group_df = pd.DataFrame(group_contributions)
    group_df = group_df.sort_values('系数绝对值总和', ascending=False)
    print(group_df.to_string(index=False))
    
    print("\n仅体质积分的贡献度（按绝对值排序）：")
    constitution_coef = coef_df[coef_df['特征名称'].isin(constitution_features)].copy()
    constitution_coef = constitution_coef.sort_values('系数绝对值', ascending=False)
    constitution_coef['排名'] = range(1, len(constitution_coef)+1)
    constitution_coef = constitution_coef[['排名', '特征名称', '标准化系数', '系数绝对值']]
    print(constitution_coef.to_string(index=False))
    
    plt.figure(figsize=(14, 8))
    top15 = coef_df_sorted.head(15)
    colors = []
    for feature in top15['特征名称']:
        if feature in constitution_features:
            colors.append('red')
        elif feature in blood_lipid_features:
            colors.append('blue')
        elif feature in activity_features:
            colors.append('green')
        else:
            colors.append('orange')
    
    bars = plt.barh(range(len(top15)), top15['系数绝对值'], color=colors, alpha=0.7)
    plt.yticks(range(len(top15)), top15['特征名称'])
    plt.xlabel('标准化系数绝对值', fontsize=12)
    plt.title('Top 15 特征贡献度（标准化系数绝对值）', fontsize=14, fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='体质积分'),
        Patch(facecolor='blue', label='血脂因素'),
        Patch(facecolor='green', label='活动能力'),
        Patch(facecolor='orange', label='人口统计')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('/workspace/Project_mat/多因素Logistic回归特征贡献度.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n可视化图表已保存：多因素Logistic回归特征贡献度.png")
    
    plt.figure(figsize=(12, 6))
    constitution_coef_sorted = constitution_coef.sort_values('系数绝对值', ascending=True)
    colors = ['red' if coef > 0 else 'blue' for coef in constitution_coef_sorted['标准化系数']]
    bars = plt.barh(range(len(constitution_coef_sorted)), constitution_coef_sorted['系数绝对值'], color=colors, alpha=0.7)
    plt.yticks(range(len(constitution_coef_sorted)), constitution_coef_sorted['特征名称'])
    plt.xlabel('标准化系数绝对值', fontsize=12)
    plt.title('九种体质积分净贡献度（控制混杂因素后）', fontsize=14, fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='正向关联（增加风险）'),
        Patch(facecolor='blue', label='负向关联（降低风险）')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('/workspace/Project_mat/九种体质净贡献度.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n可视化图表已保存：九种体质净贡献度.png")
    
    return coef_df_sorted, constitution_coef, group_df

def main():
    df = load_data()
    print("\n" + "="*100)
    print("          5.1.4 九种体质风险贡献度分析系统")
    print("="*100)
    print(f"\n✅ 数据加载成功！")
    print(f"   - 样本数量：{len(df)} 个")
    print(f"   - 特征数量：{len(df.columns)} 个")
    
    relative_risk_result = relative_risk_analysis(df)
    
    coef_df_sorted, constitution_coef, group_df = logistic_regression_analysis(df)
    
    relative_risk_result.to_csv('/workspace/Project_mat/体质相对风险分析_gbk.csv', index=False, encoding='gbk')
    relative_risk_result.to_csv('/workspace/Project_mat/体质相对风险分析_utf8bom.csv', index=False, encoding='utf-8-sig')
    
    coef_df_sorted.to_csv('/workspace/Project_mat/Logistic回归系数_gbk.csv', index=False, encoding='gbk')
    coef_df_sorted.to_csv('/workspace/Project_mat/Logistic回归系数_utf8bom.csv', index=False, encoding='utf-8-sig')
    
    constitution_coef.to_csv('/workspace/Project_mat/九种体质净贡献度_gbk.csv', index=False, encoding='gbk')
    constitution_coef.to_csv('/workspace/Project_mat/九种体质净贡献度_utf8bom.csv', index=False, encoding='utf-8-sig')
    
    group_df.to_csv('/workspace/Project_mat/特征组贡献度_gbk.csv', index=False, encoding='gbk')
    group_df.to_csv('/workspace/Project_mat/特征组贡献度_utf8bom.csv', index=False, encoding='utf-8-sig')
    
    print("\n" + "="*100)
    print("🎉 分析完成！所有结果已保存：")
    print("="*100)
    print("CSV文件（推荐使用_utf8bom版本）：")
    print("1. 体质相对风险分析_utf8bom.csv")
    print("2. Logistic回归系数_utf8bom.csv")
    print("3. 九种体质净贡献度_utf8bom.csv")
    print("4. 特征组贡献度_utf8bom.csv")
    print("\n可视化图表：")
    print("5. 体质相对风险分析.png")
    print("6. 多因素Logistic回归特征贡献度.png")
    print("7. 九种体质净贡献度.png")
    print("="*100)

if __name__ == "__main__":
    main()
