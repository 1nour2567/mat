# PU (Positive-Unlabeled) 学习实现
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.three_layer_architecture import ClinicalRuleLayer, MODEL_FEATURES

def main():
    print("=" * 80)
    print("PU (Positive-Unlabeled) 学习实现")
    print("=" * 80)
    
    # 1. 加载预处理后的数据
    print("\n[步骤1] 加载数据...")
    data_path = "data/processed/preprocessed_data.pkl"
    df = pd.read_pickle(data_path)
    print(f"数据加载完成，形状: {df.shape}")
    
    # 2. 计算血脂异常项数并划分 P 和 U 集合
    print("\n[步骤2] 划分 P 和 U 集合...")
    df = ClinicalRuleLayer.apply_clinical_rules(df)[0]
    
    # 定义 P 集合（显性高风险正例集合）和 U 集合（未标注集合）
    P = df[df['血脂异常项数'] >= 1]
    U = df[df['血脂异常项数'] == 0]
    
    print(f"P 集合大小: {len(P)} ({len(P)/len(df)*100:.1f}%)")
    print(f"U 集合大小: {len(U)} ({len(U)/len(df)*100:.1f}%)")
    
    # 3. 准备训练数据
    print("\n[步骤3] 准备训练数据...")
    # 构建 s 标签：s=1 表示属于 P 集合，s=0 表示属于 U 集合
    df['s'] = 0
    df.loc[df['血脂异常项数'] >= 1, 's'] = 1
    
    # 使用预警门特征 χ^W (对应 MODEL_FEATURES)
    X = df[MODEL_FEATURES]
    y = df['s']
    
    # 4. 训练 LightGBM 模型估计 g(x) = P(s=1 | x)
    print("\n[步骤4] 训练 LightGBM 模型...")
    model, g_pred = train_g_model(X, y)
    
    # 5. 计算 c 值
    print("\n[步骤5] 计算 c 值...")
    c = calculate_c(df, g_pred)
    print(f"c 值: {c:.4f}")
    
    # 6. 计算潜在高风险概率 r(x)
    print("\n[步骤6] 计算潜在高风险概率...")
    df['g(x)'] = g_pred
    df['r(x)'] = calculate_risk_score(df, c)
    
    # 7. 分析结果
    print("\n[步骤7] 分析结果...")
    analyze_results(df, P, U)
    
    # 8. 保存结果
    output_path = "data/processed/pu_learning_results.pkl"
    df.to_pickle(output_path)
    print(f"\n结果已保存到: {output_path}")
    
    print("\n" + "=" * 80)
    print("PU 学习实现完成！")
    print("=" * 80)

def train_g_model(X, y):
    """
    训练 LightGBM 模型估计 g(x) = P(s=1 | x)
    """
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models = []
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        train_x, val_x = X.iloc[train_idx], X.iloc[val_idx]
        train_y, val_y = y.iloc[train_idx], y.iloc[val_idx]
        
        # 处理类别不平衡
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            is_unbalance=True,
            learning_rate=0.05,
            n_estimators=1000
        )
        model.fit(train_x, train_y)
        
        models.append(model)
        
        # 验证集预测
        val_pred = model.predict_proba(val_x)[:, 1]
        oof_preds[val_idx] = val_pred
        
        # 打印验证集AUC
        val_auc = roc_auc_score(val_y, val_pred)
        print(f"Fold {fold + 1} Validation AUC: {val_auc:.4f}")
    
    # 打印整体OOF AUC
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"Overall OOF AUC: {overall_auc:.4f}")
    
    return models, oof_preds

def calculate_c(df, g_pred):
    """
    计算 c 值：c = (1/|P|) * sum_{i∈P} g(x_i)
    """
    P_indices = df[df['s'] == 1].index
    g_p = g_pred[P_indices]
    c = np.mean(g_p)
    return c

def calculate_risk_score(df, c):
    """
    计算潜在高风险得分 r_i = min(1, g(x_i)/c)
    """
    risk_scores = []
    for _, row in df.iterrows():
        g_x = row['g(x)']
        r_x = g_x / c
        r_x = min(1, r_x)
        risk_scores.append(r_x)
    return risk_scores

def analyze_results(df, P, U):
    """
    分析 PU 学习结果
    """
    print("\n=== PU 学习结果分析 ===")
    
    # 分析 P 集合的 g(x) 和 r(x)
    P_results = df[df['s'] == 1]
    print(f"\nP 集合 (显性高风险):")
    print(f"  g(x) 均值: {P_results['g(x)'].mean():.4f}")
    print(f"  r(x) 均值: {P_results['r(x)'].mean():.4f}")
    
    # 分析 U 集合的 g(x) 和 r(x)
    U_results = df[df['s'] == 0]
    print(f"\nU 集合 (未标注):")
    print(f"  g(x) 均值: {U_results['g(x)'].mean():.4f}")
    print(f"  r(x) 均值: {U_results['r(x)'].mean():.4f}")
    
    # 分析 U 集合中高风险得分的分布
    high_risk_threshold = 0.5
    U_high_risk = U_results[U_results['r(x)'] >= high_risk_threshold]
    print(f"\nU 集合中潜在高风险 (r(x) ≥ {high_risk_threshold}):")
    print(f"  数量: {len(U_high_risk)}")
    print(f"  占比: {len(U_high_risk)/len(U_results)*100:.1f}%")
    
    # 分析特征重要性
    print("\n=== 特征重要性分析 ===")
    # 这里可以添加特征重要性的计算和展示

if __name__ == "__main__":
    main()