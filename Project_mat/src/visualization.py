# 可视化模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier

def plot_shap_values(model, X, features):
    """绘制SHAP值图"""
    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 绘制汇总图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=features)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.show()

def plot_radar_chart(df, features):
    """绘制雷达图"""
    # 计算各特征的均值
    feature_means = df[features].mean().tolist()
    
    # 雷达图参数
    categories = features
    N = len(categories)
    
    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合
    
    # 绘图
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    # 绘制数据
    values = feature_means + feature_means[:1]  # 闭合
    ax.plot(angles, values, 'o-', linewidth=2, label='Mean Values')
    ax.fill(angles, values, alpha=0.25)
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylabel('Value')
    ax.set_title('Feature Radar Chart')
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('radar_chart.png')
    plt.show()

def plot_risk_distribution(df):
    """绘制风险分布规律图"""
    plt.figure(figsize=(10, 6))
    sns.countplot(x='risk_level', data=df)
    plt.title('Risk Level Distribution')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('risk_distribution.png')
    plt.show()

def plot_age_risk_relationship(df):
    """绘制年龄与风险的关系图"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='age_group', y='risk_probability', data=df)
    plt.title('Age Group vs Risk Probability')
    plt.xlabel('Age Group')
    plt.ylabel('Risk Probability')
    plt.tight_layout()
    plt.savefig('age_risk_relationship.png')
    plt.show()

def visualize_results(input_path, model=None):
    """完整可视化流程"""
    # 加载数据
    df = pd.read_pickle(input_path)
    
    # 绘制风险分布图
    if 'risk_level' in df.columns:
        plot_risk_distribution(df)
    
    # 绘制年龄与风险关系图
    if 'age_group' in df.columns and 'risk_probability' in df.columns:
        plot_age_risk_relationship(df)
    
    # 绘制SHAP值图（如果提供了模型）
    features = []
    if model is not None:
        features = [col for col in df.columns if col not in ['risk_level', 'risk_probability', 'intervention_strategy']]
        valid_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        if valid_features:
            X = df[valid_features]
            plot_shap_values(model[0], X, valid_features)
    
    # 绘制雷达图
    if not features:
        features = [col for col in df.columns if col not in ['risk_level', 'risk_probability', 'intervention_strategy']]
    valid_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    if len(valid_features) > 0:
        plot_radar_chart(df, valid_features[:5])  # 选择前5个有效数值特征
    
    return True