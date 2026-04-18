import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载预处理后的数据"""
    try:
        df = pd.read_csv('/workspace/Project_mat/预处理后数据.txt', sep='\t', encoding='gbk')
    except:
        df = pd.read_csv('/workspace/Project_mat/预处理后数据.txt', sep='\t', encoding='gb18030')
    return df

def define_feature_groups(df):
    """定义特征组"""
    # 血常规体检指标
    blood_routine_features = [
        'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 
        'TG（甘油三酯）', 'TC（总胆固醇）', 
        '空腹血糖', '血尿酸', 'BMI',
        '血脂异常项数', 'non_HDL_C', 'TC_HDL_ratio', 
        'LDL_HDL_ratio', 'TG_HDL_ratio', 'AIP'
    ]
    # 中老年人活动量表评分
    activity_features = [
        'ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
        'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
        '活动量表总分（ADL总分+IADL总分）'
    ]
    # 九种体质
    constitution_features = [
        '平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质',
        '湿热质', '血瘀质', '气郁质', '特禀质'
    ]
    return blood_routine_features, activity_features, constitution_features

def select_key_indicators_for_phlegm_dampness(df, features, target='痰湿质'):
    """筛选能有效表征痰湿体质严重程度的关键指标"""
    print("\n" + "="*80)
    print("第一部分：筛选表征痰湿体质严重程度的关键指标")
    print("="*80)
    
    X = df[features].dropna()
    y = df.loc[X.index, target]
    
    # 互信息
    mi = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({'特征名称': features, '互信息值': mi})
    mi_df = mi_df.sort_values('互信息值', ascending=False)
    
    # 随机森林特征重要性
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = pd.DataFrame({'特征名称': features, '随机森林重要性': rf.feature_importances_})
    rf_importance = rf_importance.sort_values('随机森林重要性', ascending=False)
    
    # 合并结果
    result = pd.merge(mi_df, rf_importance, on='特征名称')
    result['综合得分'] = (result['互信息值'] + result['随机森林重要性']) / 2
    result = result.sort_values('综合得分', ascending=False)
    result['排名'] = range(1, len(result)+1)
    result = result[['排名', '特征名称', '互信息值', '随机森林重要性', '综合得分']]
    
    print("\nTop 10 关键指标（痰湿体质严重程度）：")
    print(result.head(10).to_string(index=False))
    
    # 可视化
    plt.figure(figsize=(12, 6))
    top10 = result.head(10)
    sns.barplot(x='综合得分', y='特征名称', data=top10, palette='viridis')
    plt.title('Top 10 痰湿体质严重程度关键指标', fontsize=14, fontweight='bold')
    plt.xlabel('综合得分', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    plt.tight_layout()
    plt.savefig('/workspace/Project_mat/痰湿体质关键指标.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n可视化图表已保存：痰湿体质关键指标.png")
    
    return result

def select_key_indicators_for_hyperlipidemia_risk(df, features, target='高血脂症二分类标签'):
    """筛选能预警高血脂发病风险的关键指标"""
    print("\n" + "="*80)
    print("第二部分：筛选预警高血脂发病风险的关键指标")
    print("="*80)
    
    X = df[features].dropna()
    y = df.loc[X.index, target]
    
    # 互信息
    mi = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({'特征名称': features, '互信息值': mi})
    mi_df = mi_df.sort_values('互信息值', ascending=False)
    
    # 随机森林特征重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = pd.DataFrame({'特征名称': features, '随机森林重要性': rf.feature_importances_})
    rf_importance = rf_importance.sort_values('随机森林重要性', ascending=False)
    
    # 合并结果
    result = pd.merge(mi_df, rf_importance, on='特征名称')
    result['综合得分'] = (result['互信息值'] + result['随机森林重要性']) / 2
    result = result.sort_values('综合得分', ascending=False)
    result['排名'] = range(1, len(result)+1)
    result = result[['排名', '特征名称', '互信息值', '随机森林重要性', '综合得分']]
    
    print("\nTop 10 关键指标（高血脂发病风险）：")
    print(result.head(10).to_string(index=False))
    
    # 可视化
    plt.figure(figsize=(12, 6))
    top10 = result.head(10)
    sns.barplot(x='综合得分', y='特征名称', data=top10, palette='plasma')
    plt.title('Top 10 高血脂发病风险关键指标', fontsize=14, fontweight='bold')
    plt.xlabel('综合得分', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    plt.tight_layout()
    plt.savefig('/workspace/Project_mat/高血脂风险关键指标.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n可视化图表已保存：高血脂风险关键指标.png")
    
    return result

def analyze_nine_constitutions_contribution(df, constitution_features, target='高血脂症二分类标签'):
    """研究九种体质对发病风险的贡献度差异"""
    print("\n" + "="*80)
    print("第三部分：九种体质对发病风险的贡献度差异")
    print("="*80)
    
    X = df[constitution_features]
    y = df[target]
    
    # 随机森林分类器
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    contribution_df = pd.DataFrame({
        '体质类型': constitution_features,
        '贡献度': rf.feature_importances_
    }).sort_values('贡献度', ascending=False)
    contribution_df['排名'] = range(1, len(contribution_df)+1)
    contribution_df = contribution_df[['排名', '体质类型', '贡献度']]
    
    print("\n九种体质贡献度排名：")
    print(contribution_df.to_string(index=False))
    
    # 可视化
    plt.figure(figsize=(10, 6))
    sns.barplot(x='贡献度', y='体质类型', data=contribution_df, palette='Set2')
    plt.title('九种体质对高血脂发病风险的贡献度', fontsize=14, fontweight='bold')
    plt.xlabel('贡献度', fontsize=12)
    plt.ylabel('体质类型', fontsize=12)
    plt.tight_layout()
    plt.savefig('/workspace/Project_mat/九种体质贡献度.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n可视化图表已保存：九种体质贡献度.png")
    
    return contribution_df

def main():
    print("\n" + "="*80)
    print("          关键指标筛选与九种体质贡献度分析系统")
    print("="*80)
    
    # 加载数据
    df = load_data()
    print(f"\n✅ 数据加载成功！")
    print(f"   - 样本数量：{len(df)} 个")
    print(f"   - 特征数量：{len(df.columns)} 个")
    
    # 定义特征组
    blood_routine_features, activity_features, constitution_features = define_feature_groups(df)
    all_candidate_features = blood_routine_features + activity_features
    
    # 1. 筛选表征痰湿体质严重程度的关键指标
    phlegm_dampness_indicators = select_key_indicators_for_phlegm_dampness(df, all_candidate_features)
    
    # 2. 筛选预警高血脂发病风险的关键指标
    hyperlipidemia_indicators = select_key_indicators_for_hyperlipidemia_risk(df, all_candidate_features)
    
    # 3. 研究九种体质对发病风险的贡献度
    constitution_contribution = analyze_nine_constitutions_contribution(df, constitution_features)
    
    # 保存结果
    phlegm_dampness_indicators.to_csv('/workspace/Project_mat/痰湿体质关键指标.csv', index=False, encoding='gbk')
    hyperlipidemia_indicators.to_csv('/workspace/Project_mat/高血脂风险关键指标.csv', index=False, encoding='gbk')
    constitution_contribution.to_csv('/workspace/Project_mat/九种体质贡献度.csv', index=False, encoding='gbk')
    
    print("\n" + "="*80)
    print("🎉 分析完成！所有结果已保存：")
    print("="*80)
    print("1. 痰湿体质关键指标.csv")
    print("2. 高血脂风险关键指标.csv")
    print("3. 九种体质贡献度.csv")
    print("4. 痰湿体质关键指标.png")
    print("5. 高血脂风险关键指标.png")
    print("6. 九种体质贡献度.png")
    print("="*80)

if __name__ == "__main__":
    main()
