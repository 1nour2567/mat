#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.1.4 分析：九种体质风险贡献度分析
包含三个部分：
1. 主标签边际风险模型
2. 连续积分净贡献模型
3. 性别与年龄异质性的补充分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 机器学习库
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency, spearmanr


def load_data():
    """
    加载预处理后的数据
    """
    try:
        df = pd.read_pickle('/workspace/Project_mat/data/processed/preprocessed_data.pkl')
        print(f"数据加载成功！样本数: {len(df)}, 特征数: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("尝试从CSV加载...")
        try:
            df = pd.read_csv('/workspace/预处理后数据.csv')
            print(f"数据加载成功！样本数: {len(df)}, 特征数: {len(df.columns)}")
            return df
        except Exception as e2:
            print(f"CSV加载失败: {e2}")
            return None


def section_1_main_label_analysis(df):
    """
    第一部分：主标签边际风险模型
    """
    print("\n" + "=" * 80)
    print("5.1.4.1 主标签边际风险模型")
    print("=" * 80)

    # 查找相关特征
    constitution_scores = [col for col in df.columns if '质' in col and '体质' not in col]
    print(f"\n找到的体质积分特征: {constitution_scores}")

    # 查找体质标签（如果存在）
    constitution_label_col = None
    if '体质标签' in df.columns:
        constitution_label_col = '体质标签'
        print(f"找到体质标签列")
    elif '主体质' in df.columns:
        constitution_label_col = '主体质'
        print(f"找到主体质列")
    else:
        print("未找到明确的体质标签列，将尝试根据积分创建主要体质")
        if len(constitution_scores) > 0:
            df['主体质'] = df[constitution_scores].idxmax(axis=1)
            constitution_label_col = '主体质'
            print("已根据最大积分创建主体质标签")

    # 查找高血脂标签
    hyperlipidemia_col = None
    possible_names = ['高血脂症二分类标签', '高血脂', '血脂异常', 'hypertrophy']
    for name in possible_names:
        if name in df.columns:
            hyperlipidemia_col = name
            break

    if hyperlipidemia_col is None:
        # 尝试从血脂指标创建
        lipid_features = [col for col in df.columns if '胆' in col or '甘' in col or '脂' in col or 'HDL' in col or 'LDL' in col]
        if len(lipid_features) > 0:
            print(f"将根据血脂指标创建标签，找到血脂特征: {lipid_features[:5]}")
            # 简单创建一个二分类标签（示例）
            if 'TG（甘油三酯）' in df.columns:
                df['高血脂症二分类标签'] = (df['TG（甘油三酯）'] > df['TG（甘油三酯）'].median()).astype(int)
                hyperlipidemia_col = '高血脂症二分类标签'

    if hyperlipidemia_col is None:
        print("无法找到或创建高血脂标签，部分分析将跳过")
        return None, None

    print(f"使用高血脂标签: {hyperlipidemia_col}")

    if constitution_label_col is not None:
        # 1. 计算总体患病率
        overall_prevalence = df[hyperlipidemia_col].mean()
        print(f"\n1. 总体患病率: {overall_prevalence:.4f}")

        # 2. 计算各体质组的患病率和相对风险
        print("\n2. 各体质组分析:")
        contingency_data = []
        results = []

        for constitution in df[constitution_label_col].dropna().unique():
            subset = df[df[constitution_label_col] == constitution]
            if len(subset) < 5:
                continue

            n = len(subset)
            cases = subset[hyperlipidemia_col].sum()
            non_cases = n - cases
            prevalence = cases / n
            relative_risk = prevalence / overall_prevalence if overall_prevalence > 0 else 0

            results.append({
                '体质': constitution,
                '人数': n,
                '确诊数': cases,
                '患病率': prevalence,
                '相对风险': relative_risk
            })
            contingency_data.append([cases, non_cases])

            print(f"  {constitution}: 人数={n}, 确诊数={cases}, "
                  f"患病率={prevalence:.4f}, 相对风险={relative_risk:.4f}")

        # 3. 卡方独立性检验
        if len(contingency_data) >= 2:
            print("\n3. 卡方独立性检验:")
            chi2, p_value, dof, expected = chi2_contingency(contingency_data)
            print(f"   卡方统计量: {chi2:.4f}")
            print(f"   p值: {p_value:.4f}")
            print(f"   自由度: {dof}")

            if p_value < 0.05:
                print("   结论: 体质标签与高血脂风险存在统计学显著关联 (p < 0.05)")
            else:
                print("   结论: 体质标签与高血脂风险无统计学显著关联")

        # 保存结果
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('患病率', ascending=False)

        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 患病率图
        colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
        bars1 = ax1.barh(range(len(results_df)), results_df['患病率'], color=colors)
        ax1.set_yticks(range(len(results_df)))
        ax1.set_yticklabels(results_df['体质'])
        ax1.set_xlabel('患病率')
        ax1.set_title('各体质组高血脂患病率')
        ax1.axvline(overall_prevalence, color='red', linestyle='--', label='总体水平')
        ax1.legend()

        # 相对风险图
        bars2 = ax2.barh(range(len(results_df)), results_df['相对风险'], color=colors)
        ax2.set_yticks(range(len(results_df)))
        ax2.set_yticklabels(results_df['体质'])
        ax2.set_xlabel('相对风险')
        ax2.set_title('各体质组相对风险（相对于总体）')
        ax2.axvline(1.0, color='red', linestyle='--', label='总体水平=1')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('/workspace/Project_mat/图5_1_1_主标签边际风险分析.png', dpi=300, bbox_inches='tight')
        print("\n已保存: 图5_1_1_主标签边际风险分析.png")

        results_df.to_csv('/workspace/Project_mat/5_1_4_主标签分析结果.csv', index=False, encoding='utf-8-sig')
        print("已保存: 5_1_4_主标签分析结果.csv")

        return results_df, hyperlipidemia_col

    return None, hyperlipidemia_col


def section_2_continuous_score_model(df, hyperlipidemia_col):
    """
    第二部分：连续积分净贡献模型
    """
    print("\n" + "=" * 80)
    print("5.1.4.2 连续积分净贡献模型")
    print("=" * 80)

    # 查找体质积分特征
    constitution_scores = [col for col in df.columns if '质' in col and '体质' not in col]
    print(f"\n找到的体质积分特征: {constitution_scores}")

    # 查找控制变量
    control_vars = []

    # 血脂指标
    lipid_features = [col for col in df.columns if 'TG' in col or 'TC' in col or 'HDL' in col or 'LDL' in col]
    control_vars.extend(lipid_features[:5])

    # 活动能力
    activity_features = [col for col in df.columns if 'ADL' in col or '活动' in col]
    control_vars.extend(activity_features[:3])

    # 人口统计学
    demo_features = [col for col in df.columns if '年龄' in col or '性别' in col or '吸烟' in col or '饮酒' in col]
    control_vars.extend(demo_features[:4])

    # BMI
    if 'BMI' in df.columns:
        control_vars.append('BMI')

    # 过滤存在的特征
    valid_control_vars = [var for var in control_vars if var in df.columns]
    valid_constitution_scores = [cs for cs in constitution_scores if cs in df.columns]

    print(f"有效控制变量: {valid_control_vars}")
    print(f"有效体质积分: {valid_constitution_scores}")

    if len(valid_constitution_scores) == 0 or hyperlipidemia_col is None:
        print("缺少必要数据，无法进行分析")
        return None

    # 准备数据
    all_features = valid_control_vars + valid_constitution_scores
    data_analysis = df[all_features + [hyperlipidemia_col]].dropna().copy()

    X = data_analysis[all_features]
    y = data_analysis[hyperlipidemia_col]

    print(f"\n分析样本数: {len(data_analysis)}")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=all_features, index=data_analysis.index)

    # Logistic回归
    lr = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    lr.fit(X_scaled, y)

    # 提取系数
    coefficients = pd.DataFrame({
        '特征': all_features,
        '系数': lr.coef_[0],
        '绝对系数': np.abs(lr.coef_[0])
    })

    # 计算贡献度和贡献率
    constitution_coef = coefficients[coefficients['特征'].isin(valid_constitution_scores)].copy()
    total_contribution = constitution_coef['绝对系数'].sum()

    constitution_coef['绝对贡献度'] = constitution_coef['绝对系数']
    constitution_coef['贡献率'] = constitution_coef['绝对系数'] / total_contribution if total_contribution > 0 else 0
    constitution_coef = constitution_coef.sort_values('绝对贡献度', ascending=False)

    print("\n=== 标准化Logistic回归结果（体质部分） ===")
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(constitution_coef[['特征', '系数', '绝对贡献度', '贡献率']])

    # 解释
    print("\n=== 结果解释 ===")
    print(f"β_i > 0: 该体质维度升高会推动高血脂风险上升")
    print(f"β_i < 0: 该体质维度表现为相对保护方向")
    print(f"绝对贡献度: |β_i|，表示该体质的净贡献强度")
    print(f"贡献率: |β_i| / Σ|β_i|，表示该体质在九种体质中的占比")

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 系数图
    colors = ['red' if c > 0 else 'blue' for c in constitution_coef['系数']]
    bars1 = ax1.barh(range(len(constitution_coef)), constitution_coef['系数'], color=colors)
    ax1.set_yticks(range(len(constitution_coef)))
    ax1.set_yticklabels(constitution_coef['特征'])
    ax1.set_xlabel('标准化回归系数')
    ax1.set_title('各体质的标准化回归系数')
    ax1.axvline(0, color='black', linestyle='-', linewidth=1)
    ax1.text(0.05, 0.95, '红色：风险升高\n蓝色：保护作用', transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 贡献度图
    colors = plt.cm.RdBu_r(constitution_coef['贡献率'])
    bars2 = ax2.barh(range(len(constitution_coef)), constitution_coef['贡献率'], color=colors)
    ax2.set_yticks(range(len(constitution_coef)))
    ax2.set_yticklabels(constitution_coef['特征'])
    ax2.set_xlabel('贡献率')
    ax2.set_title('各体质的贡献率')

    # 添加数值标签
    for i, (idx, row) in enumerate(constitution_coef.iterrows()):
        ax2.text(row['贡献率'] + 0.01, i, f"{row['贡献率']:.1%}",
                va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('/workspace/Project_mat/图5_1_2_连续积分净贡献模型.png', dpi=300, bbox_inches='tight')
    print("\n已保存: 图5_1_2_连续积分净贡献模型.png")

    constitution_coef.to_csv('/workspace/Project_mat/5_1_4_连续积分分析结果.csv', index=False, encoding='utf-8-sig')
    print("已保存: 5_1_4_连续积分分析结果.csv")

    return constitution_coef


def section_3_heterogeneity_analysis(df, hyperlipidemia_col):
    """
    第三部分：性别与年龄异质性的补充分析
    """
    print("\n" + "=" * 80)
    print("5.1.4.3 性别与年龄异质性的补充分析")
    print("=" * 80)

    # 查找体质积分特征
    constitution_scores = [col for col in df.columns if '质' in col and '体质' not in col]
    valid_constitution_scores = [cs for cs in constitution_scores if cs in df.columns]

    if len(valid_constitution_scores) == 0 or hyperlipidemia_col is None:
        print("缺少必要数据，无法进行分析")
        return None

    # 查找分组变量
    gender_col = None
    age_group_col = None

    for col in df.columns:
        if '性别' in col:
            gender_col = col
        elif '年龄组' in col or '年龄分组' in col:
            age_group_col = col

    print(f"性别列: {gender_col}")
    print(f"年龄组列: {age_group_col}")

    # 控制变量
    control_vars = []
    lipid_features = [col for col in df.columns if 'TG' in col or 'TC' in col or 'HDL' in col or 'LDL' in col][:3]
    control_vars.extend(lipid_features)
    if 'BMI' in df.columns:
        control_vars.append('BMI')

    valid_control_vars = [var for var in control_vars if var in df.columns]

    all_features = valid_control_vars + valid_constitution_scores

    results_by_group = {}

    # 1. 性别分组分析
    if gender_col is not None and df[gender_col].nunique() >= 2:
        print(f"\n--- 性别分组分析 ---")
        gender_groups = df[gender_col].dropna().unique()

        for gender in gender_groups:
            subset = df[df[gender_col] == gender]
            if len(subset) < 50:
                continue

            print(f"\n分析性别组: {gender}, 样本数: {len(subset)}")

            data_sub = subset[all_features + [hyperlipidemia_col]].dropna()
            if len(data_sub) < 30:
                continue

            X = data_sub[all_features]
            y = data_sub[hyperlipidemia_col]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_scaled, y)

            coef_df = pd.DataFrame({
                '特征': all_features,
                f'{gender}_系数': lr.coef_[0]
            })

            results_by_group[f'gender_{gender}'] = coef_df

    # 2. 年龄组分析
    if age_group_col is not None and df[age_group_col].nunique() >= 3:
        print(f"\n--- 年龄组分组分析 ---")
        age_groups = sorted(df[age_group_col].dropna().unique())

        for age_group in age_groups:
            subset = df[df[age_group_col] == age_group]
            if len(subset) < 50:
                continue

            print(f"\n分析年龄组: {age_group}, 样本数: {len(subset)}")

            data_sub = subset[all_features + [hyperlipidemia_col]].dropna()
            if len(data_sub) < 30:
                continue

            X = data_sub[all_features]
            y = data_sub[hyperlipidemia_col]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_scaled, y)

            coef_df = pd.DataFrame({
                '特征': all_features,
                f'{age_group}_系数': lr.coef_[0]
            })

            results_by_group[f'age_{age_group}'] = coef_df

    # 合并结果并只显示体质部分
    if results_by_group:
        merged = None
        for group_name, coef_df in results_by_group.items():
            if merged is None:
                merged = coef_df.copy()
            else:
                merged = merged.merge(coef_df, on='特征', how='left')

        # 只保留体质特征
        constitution_results = merged[merged['特征'].isin(valid_constitution_scores)].copy()

        print("\n=== 分组分析结果汇总（体质系数） ===")
        print(constitution_results.to_string())

        # 可视化
        n_groups = len([col for col in constitution_results.columns if col != '特征'])
        if n_groups > 0:
            fig, ax = plt.subplots(figsize=(12, 8))

            x = np.arange(len(constitution_results))
            width = 0.8 / n_groups

            for i, col in enumerate([c for c in constitution_results.columns if c != '特征']):
                ax.barh(x + i * width, constitution_results[col], width, label=col)

            ax.set_yticks(x + width * (n_groups - 1) / 2)
            ax.set_yticklabels(constitution_results['特征'])
            ax.set_xlabel('标准化回归系数')
            ax.set_title('不同性别/年龄组的体质贡献度对比')
            ax.legend()
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)

            plt.tight_layout()
            plt.savefig('/workspace/Project_mat/图5_1_3_性别年龄异质性分析.png', dpi=300, bbox_inches='tight')
            print("\n已保存: 图5_1_3_性别年龄异质性分析.png")

        constitution_results.to_csv('/workspace/Project_mat/5_1_4_异质性分析结果.csv', index=False, encoding='utf-8-sig')
        print("已保存: 5_1_4_异质性分析结果.csv")

        return constitution_results

    return None


def main():
    """
    主函数：执行完整的5.1.4分析
    """
    print("=" * 80)
    print("5.1.4 九种体质风险贡献度分析 - 完整流程")
    print("=" * 80)

    # 加载数据
    df = load_data()
    if df is None:
        return

    # 第一部分：主标签边际风险模型
    results_section1, hyperlipidemia_col = section_1_main_label_analysis(df)

    # 第二部分：连续积分净贡献模型
    results_section2 = section_2_continuous_score_model(df, hyperlipidemia_col)

    # 第三部分：性别与年龄异质性的补充分析
    results_section3 = section_3_heterogeneity_analysis(df, hyperlipidemia_col)

    # 总结
    print("\n" + "=" * 80)
    print("分析完成总结")
    print("=" * 80)
    print("\n已生成文件：")
    print("1. 图5_1_1_主标签边际风险分析.png")
    print("2. 图5_1_2_连续积分净贡献模型.png")
    print("3. 图5_1_3_性别年龄异质性分析.png")
    print("4. 5_1_4_主标签分析结果.csv")
    print("5. 5_1_4_连续积分分析结果.csv")
    print("6. 5_1_4_异质性分析结果.csv")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
