# 可视化模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams

rcParams['font.family'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'sans-serif']
rcParams['axes.unicode_minus'] = False

def plot_shap_values(model, X, features):
    """绘制美观的SHAP值图"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(14, 10), dpi=100)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    if isinstance(shap_values, list):
        shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_vals = shap_values
    
    shap.summary_plot(
        shap_vals, 
        X, 
        feature_names=features,
        plot_type='dot',
        cmap=LinearSegmentedColormap.from_list('custom', ['#4ECDC4', '#FF6B6B']),
        max_display=15,
        show=False
    )
    
    plt.gcf().set_size_inches(14, 10)
    plt.title('特征重要性分析 (SHAP)', fontsize=16, pad=30, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_radar_chart(df, features):
    """绘制美观的雷达图"""
    feature_means = df[features].mean().tolist()
    categories = features
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111, polar=True)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    values = feature_means + feature_means[:1]
    
    ax.plot(angles, values, 'o-', linewidth=4, color=colors[1], label='特征均值')
    ax.fill(angles, values, alpha=0.35, color=colors[1])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='medium')
    ax.set_ylim(bottom=0)
    
    ax.set_yticklabels([])
    ax.spines['polar'].set_color('#E0E0E0')
    ax.grid(color='#E0E0E0', linestyle='--', linewidth=0.8)
    
    ax.set_title('特征雷达图分析', fontsize=16, pad=40, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.tight_layout()
    plt.savefig('radar_chart.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_risk_distribution(df):
    """绘制美观的风险分布图"""
    risk_counts = df['risk_level'].value_counts().sort_index()
    total = len(df)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=100)
    
    colors = ['#96CEB4', '#FFEAA7', '#FF6B6B']
    risk_labels = risk_counts.index.tolist()
    
    bars = ax1.bar(
        risk_labels, 
        risk_counts.values,
        color=colors[:len(risk_labels)],
        edgecolor='white',
        linewidth=2,
        alpha=0.85
    )
    
    ax1.set_xlabel('风险等级', fontsize=13, fontweight='medium', labelpad=15)
    ax1.set_ylabel('样本数量', fontsize=13, fontweight='medium', labelpad=15)
    ax1.set_title('风险等级分布 - 柱状图', fontsize=15, pad=20, fontweight='bold')
    
    for bar, count in zip(bars, risk_counts.values):
        height = bar.get_height()
        percentage = (count / total) * 100
        ax1.text(
            bar.get_x() + bar.get_width()/2, 
            height + max(risk_counts.values)*0.01,
            f'{count}\n({percentage:.1f}%)',
            ha='center', 
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    
    wedges, texts, autotexts = ax2.pie(
        risk_counts.values,
        labels=risk_labels,
        colors=colors[:len(risk_labels)],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 12, 'fontweight': 'medium'},
        pctdistance=0.85
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    ax2.set_title('风险等级分布 - 饼图', fontsize=15, pad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('risk_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_age_risk_relationship(df):
    """绘制美观的年龄与风险关系图"""
    if 'age_group' not in df.columns or 'risk_probability' not in df.columns:
        return
    
    plt.figure(figsize=(12, 7), dpi=100)
    
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#FF6B6B']
    
    ax = sns.boxplot(
        x='age_group', 
        y='risk_probability', 
        data=df,
        palette=colors,
        width=0.6,
        linewidth=2,
        flierprops={'marker': 'o', 'markersize': 6, 'alpha': 0.5}
    )
    
    plt.title('年龄组与风险概率关系', fontsize=16, pad=25, fontweight='bold')
    plt.xlabel('年龄组', fontsize=13, fontweight='medium', labelpad=15)
    plt.ylabel('风险概率', fontsize=13, fontweight='medium', labelpad=15)
    
    plt.xticks(fontsize=11, rotation=0)
    plt.yticks(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('age_risk_relationship.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_results(input_path, model=None):
    """完整可视化流程"""
    df = pd.read_pickle(input_path)
    
    plot_risk_distribution(df)
    
    if 'age_group' in df.columns and 'risk_probability' in df.columns:
        plot_age_risk_relationship(df)
    
    features = []
    if model is not None:
        features = [col for col in df.columns if col not in ['risk_level', 'risk_probability', 'intervention_strategy']]
        X = df[features]
        plot_shap_values(model[0], X, features)
    
    if len(features) == 0:
        features = [col for col in df.columns if col not in ['risk_level', 'risk_probability', 'intervention_strategy']]
    
    if len(features) > 0:
        plot_radar_chart(df, features[:min(5, len(features))])
    
    return True