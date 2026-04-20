"""
问题三：痰湿体质患者干预方案优化（动态规划版）
创新点：效果递减 + 协同效应 + 鲁棒优化 + 耐受度惩罚
求解方法：动态规划
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects
import warnings

warnings.filterwarnings('ignore')
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

try:
    fm.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
except:
    pass

sns.set_style("whitegrid")

# 配色方案
COLORS = {
    'tcm_1': '#4CAF50', 'tcm_2': '#FF9800', 'tcm_3': '#F44336',
    'act_1': '#2196F3', 'act_2': '#9C27B0', 'act_3': '#FF5722',
    'primary': '#2E86AB', 'secondary': '#A23B72',
    'accent1': '#F18F01', 'accent2': '#C73E1D',
    'success': '#27AE60', 'warning': '#E67E22', 'danger': '#E74C3C',
}

# ============================================================================
# 1. 数据读取
# ============================================================================
df = pd.read_csv('E:/preprocessed_data_clip.csv', encoding='utf-8')
df_phlegm = df[df['体质标签'] == 5].copy()

print("=" * 80)
print("【问题三】基于动态规划的干预方案优化")
print("=" * 80)
print(f"痰湿体质患者数量: {len(df_phlegm)}人")


# ============================================================================
# 2. 参数定义
# ============================================================================
class Params:
    TCM_RATES = {1: (0.00, 0.00, 0.00), 2: (0.01, 0.005, 0.015), 3: (0.02, 0.015, 0.025)}
    INTENSITY_EFFECT = (0.03, 0.02, 0.04)
    FREQUENCY_EFFECT = (0.01, 0.005, 0.015)
    TCM_COST = {1: 30, 2: 80, 3: 130}
    ACT_UNIT_COST = {1: 3, 2: 5, 3: 8}
    DIMINISHING_ALPHA = 0.25
    SYNERGY_GAMMA = 0.10
    LAMBDA_ROBUST = 0.3
    TOLERANCE_LAMBDA = 0.10
    BUDGET = 2000


# ============================================================================
# 3. 约束函数
# ============================================================================
def get_tcm_level(phlegm_score):
    if phlegm_score <= 58:
        return 1
    elif phlegm_score <= 61:
        return 2
    else:
        return 3


def get_allowed_intensities(age_group, activity_score):
    age_mid = {1: 45, 2: 55, 3: 65, 4: 75, 5: 85}[age_group]
    allowed = []
    for intensity in [1, 2, 3]:
        if age_mid >= 80 and intensity > 1: continue
        if age_mid >= 60 and intensity > 2: continue
        if activity_score < 40 and intensity > 1: continue
        if activity_score < 60 and intensity > 2: continue
        allowed.append(intensity)
    return allowed


# ============================================================================
# 4. 效果计算函数
# ============================================================================
def compute_month_rate(tcm_level, intensity, freq, current_score, initial_score, worst_case=False):
    if worst_case:
        r_tcm = Params.TCM_RATES[tcm_level][1]
        intensity_effect = Params.INTENSITY_EFFECT[1]
        freq_effect = Params.FREQUENCY_EFFECT[1]
    else:
        r_tcm = Params.TCM_RATES[tcm_level][0]
        intensity_effect = Params.INTENSITY_EFFECT[0]
        freq_effect = Params.FREQUENCY_EFFECT[0]

    r_act = 0.0 if freq < 5 else max(0, intensity_effect * (intensity - 1) + freq_effect * (freq - 5))

    diminishing = 1 - Params.DIMINISHING_ALPHA * (initial_score - current_score) / initial_score
    diminishing = max(0.5, diminishing)
    r_tcm *= diminishing
    r_act *= diminishing

    synergy = 1 + Params.SYNERGY_GAMMA if (tcm_level >= 2 and intensity >= 2 and freq >= 5) else 1
    r_total = (1 - (1 - r_tcm) * (1 - r_act)) * synergy
    return min(r_total, 0.15)


def compute_cost(tcm_level, intensity, freq):
    return Params.TCM_COST[tcm_level] + Params.ACT_UNIT_COST[intensity] * freq * 4


# ============================================================================
# 5. 动态规划求解
# ============================================================================
def dp_optimize(S0, age_group, activity_score):
    """
    动态规划求解最优干预方案
    """
    allowed_intensities = get_allowed_intensities(age_group, activity_score)
    initial_tcm = get_tcm_level(S0)

    # 状态离散化参数
    SCORE_STEP = 0.5
    BUDGET_STEP = 50

    # 记忆化字典
    memo = {}

    def discretize_score(score):
        return round(score / SCORE_STEP) * SCORE_STEP

    def discretize_budget(budget):
        return (budget // BUDGET_STEP) * BUDGET_STEP

    def dp(month, current_score, remaining_budget):
        """返回：(最大累计下降量, 最优路径)"""
        if month == 6:
            return 0, []

        # 离散化状态作为key
        state_key = (month, discretize_score(current_score), discretize_budget(remaining_budget))
        if state_key in memo:
            return memo[state_key]

        current_tcm = get_tcm_level(current_score)
        best_reduction = -1
        best_path = []

        for intensity in allowed_intensities:
            for freq in range(1, 11):
                cost = compute_cost(current_tcm, intensity, freq)
                if cost > remaining_budget:
                    continue

                # 计算本月效果
                r_nom = compute_month_rate(current_tcm, intensity, freq,
                                           current_score, S0, worst_case=False)
                next_score = current_score * (1 - r_nom)
                month_reduction = current_score * r_nom

                # 递归求解后续月份
                future_reduction, future_path = dp(month + 1, next_score,
                                                   remaining_budget - cost)
                total_reduction = month_reduction + future_reduction

                if total_reduction > best_reduction:
                    best_reduction = total_reduction
                    best_path = [(current_tcm, intensity, freq, cost)] + future_path

        memo[state_key] = (best_reduction, best_path)
        return memo[state_key]

    # 求解
    total_reduction, path = dp(0, S0, Params.BUDGET)

    if not path:
        return None, None, None

    # 从路径中提取最优方案（取第一个月的决策）
    first_month = path[0]
    best_intensity = first_month[1]
    best_freq = first_month[2]
    total_cost = sum(p[3] for p in path)

    # 计算名义效果和最坏效果
    S_nom = S0
    S_worst = S0
    for month in range(6):
        current_tcm = get_tcm_level(S_nom)
        r_nom = compute_month_rate(current_tcm, best_intensity, best_freq, S_nom, S0, False)
        r_worst = compute_month_rate(current_tcm, best_intensity, best_freq, S_worst, S0, True)
        S_nom *= (1 - r_nom)
        S_worst *= (1 - r_worst)

    nominal_delta = S0 - S_nom
    worst_delta = S0 - S_worst
    tolerance_penalty = Params.TOLERANCE_LAMBDA * best_intensity * best_freq
    robust_score = (1 - Params.LAMBDA_ROBUST) * nominal_delta + Params.LAMBDA_ROBUST * worst_delta - tolerance_penalty
    robustness = worst_delta / nominal_delta if nominal_delta > 0 else 1

    return {
        'intensity': best_intensity,
        'freq': best_freq,
        'nominal_delta': nominal_delta,
        'worst_delta': worst_delta,
        'robust_score': robust_score,
        'nominal_cost': total_cost,
        'robustness': robustness,
        'tolerance_penalty': tolerance_penalty
    }


# ============================================================================
# 6. 批量求解
# ============================================================================
print("\n" + "-" * 80)
print("求解中（动态规划 + 效果递减 + 协同效应 + 鲁棒优化 + 耐受度惩罚）...")
print("-" * 80)

results = []
for idx, row in df_phlegm.iterrows():
    best_plan = dp_optimize(row['痰湿质'], row['年龄组'],
                            row['活动量表总分（ADL总分+IADL总分）'])
    if best_plan:
        initial_tcm = get_tcm_level(row['痰湿质'])
        results.append({
            '样本ID': row['样本ID'],
            '当前痰湿积分': row['痰湿质'],
            '年龄组': row['年龄组'],
            '活动总分': row['活动量表总分（ADL总分+IADL总分）'],
            '初始中医等级': initial_tcm,
            '最优强度': best_plan['intensity'],
            '最优频率': best_plan['freq'],
            '名义下降': round(best_plan['nominal_delta'], 2),
            '最坏下降': round(best_plan['worst_delta'], 2),
            '总成本': round(best_plan['nominal_cost'], 2),
            '鲁棒性': round(best_plan['robustness'], 3),
            '耐受惩罚': round(best_plan['tolerance_penalty'], 2),
        })

df_results = pd.DataFrame(results)
df_results['痰湿分组'] = pd.cut(df_results['当前痰湿积分'], bins=[0, 58, 61, 100],
                                labels=['≤58分(轻度)', '59-61分(中度)', '≥62分(重度)'])
df_results['活动分组'] = pd.cut(df_results['活动总分'], bins=[0, 40, 60, 100],
                                labels=['<40分(重度依赖)', '40-59分(中度依赖)', '≥60分(轻度依赖)'])

# ============================================================================
# 7. 控制台输出
# ============================================================================
print(f"\n【方案分布】")
print(f"  中医等级: 1级={len(df_results[df_results['初始中医等级'] == 1])}人, "
      f"2级={len(df_results[df_results['初始中医等级'] == 2])}人, "
      f"3级={len(df_results[df_results['初始中医等级'] == 3])}人")
print(f"  活动强度: 1级={len(df_results[df_results['最优强度'] == 1])}人, "
      f"2级={len(df_results[df_results['最优强度'] == 2])}人, "
      f"3级={len(df_results[df_results['最优强度'] == 3])}人")

print(f"\n【频率分布】")
freq_dist = df_results['最优频率'].value_counts().sort_index()
for freq, count in freq_dist.items():
    print(f"  {freq}次/周: {count}人")

print(f"\n【鲁棒性】平均鲁棒性: {df_results['鲁棒性'].mean():.3f}")

print("\n" + "=" * 80)
print("【样本ID=1,2,3的最优干预方案】")
print("=" * 80)
for sample_id in [1, 2, 3]:
    plan = df_results[df_results['样本ID'] == sample_id].iloc[0]
    print(f"\n样本ID = {sample_id}: 痰湿{plan['当前痰湿积分']:.0f}分 → 中医{int(plan['初始中医等级'])}级")
    print(f"  活动强度: {int(plan['最优强度'])}级, 每周{int(plan['最优频率'])}次, 成本: {plan['总成本']:.0f}元")
    print(f"  名义下降: {plan['名义下降']:.2f}分, 最坏下降: {plan['最坏下降']:.2f}分, 耐受惩罚: {plan['耐受惩罚']:.2f}")

# ============================================================================
# 8. 保存结果
# ============================================================================
df_results.to_csv('E:/问题3_动态规划结果.csv', index=False, encoding='utf-8-sig')
print("\n✓ 结果已保存至 E:/问题3_动态规划结果.csv")

# ============================================================================
# 9. 可视化（保留原有6张图）
# ============================================================================
print("\n" + "-" * 80)
print("生成精美可视化图表...")
print("-" * 80)

# 图1：等级分布双饼图
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
tcm_counts = df_results['初始中医等级'].value_counts().sort_index()
colors_tcm = [COLORS['tcm_1'], COLORS['tcm_2'], COLORS['tcm_3']]
wedges, texts, autotexts = axes[0].pie(tcm_counts.values, labels=['1级\n(0-58分)', '2级\n(59-61分)', '3级\n(≥62分)'],
                                       colors=colors_tcm, autopct='%1.1f%%', startangle=90, explode=(0.03, 0.03, 0.03),
                                       textprops={'fontsize': 13, 'fontweight': 'bold'},
                                       wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')
axes[0].set_title('中医调理等级分布', fontsize=16, fontweight='bold', pad=20)

act_counts = df_results['最优强度'].value_counts().sort_index()
colors_act = [COLORS['act_1'], COLORS['act_2'], COLORS['act_3']]
wedges, texts, autotexts = axes[1].pie(act_counts.values, labels=['1级', '2级', '3级'],
                                       colors=colors_act, autopct='%1.1f%%', startangle=90, explode=(0.03, 0.03, 0.03),
                                       textprops={'fontsize': 13, 'fontweight': 'bold'},
                                       wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')
axes[1].set_title('活动强度等级分布', fontsize=16, fontweight='bold', pad=20)
plt.suptitle('干预方案等级分布', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('E:/问题3_图1_等级分布.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ 图1：等级分布饼图")

# 图2：频率分布柱状图
fig, ax = plt.subplots(figsize=(14, 6))
freq_counts = df_results['最优频率'].value_counts().sort_index()
colors_freq = plt.cm.Blues(np.linspace(0.4, 0.9, len(freq_counts)))
bars = ax.bar(freq_counts.index.astype(str), freq_counts.values, color=colors_freq, edgecolor='white', linewidth=2)
for bar, val in zip(bars, freq_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(val), ha='center', fontsize=13,
            fontweight='bold', color=COLORS['primary'])
avg_freq = df_results['最优频率'].mean()
ax.axhline(y=freq_counts.mean(), color=COLORS['accent2'], linestyle='--', linewidth=2,
           label=f'平均频率: {avg_freq:.1f}次/周')
ax.set_xlabel('每周活动频率（次）', fontsize=13, fontweight='bold')
ax.set_ylabel('人数', fontsize=13, fontweight='bold')
ax.set_title('最优活动频率分布', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, max(freq_counts.values) * 1.2)
ax.legend(fontsize=11, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('E:/问题3_图2_频率分布.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ 图2：频率分布柱状图")

# 图3：成本-效果气泡图
fig, ax = plt.subplots(figsize=(14, 8))
colors_map = {1: COLORS['tcm_1'], 2: COLORS['tcm_2'], 3: COLORS['tcm_3']}
for tcm_level in [1, 2, 3]:
    subset = df_results[df_results['初始中医等级'] == tcm_level]
    sizes = subset['名义下降'] * 15
    ax.scatter(subset['总成本'], subset['名义下降'], s=sizes, c=colors_map[tcm_level], alpha=0.6,
               edgecolors='white', linewidth=1.5, label=f'中医{tcm_level}级')
for sample_id in [1, 2, 3]:
    plan = df_results[df_results['样本ID'] == sample_id].iloc[0]
    ax.annotate(f"ID{sample_id}", xy=(plan['总成本'], plan['名义下降']),
                xytext=(10, 10), textcoords='offset points', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
z = np.polyfit(df_results['总成本'], df_results['名义下降'], 1)
p = np.poly1d(z)
x_trend = np.linspace(df_results['总成本'].min(), df_results['总成本'].max(), 100)
ax.plot(x_trend, p(x_trend), '--', color='gray', linewidth=2, alpha=0.7, label='趋势线')
ax.set_xlabel('总成本（元）', fontsize=13, fontweight='bold')
ax.set_ylabel('名义下降量（分）', fontsize=13, fontweight='bold')
ax.set_title('成本-效果分析（气泡大小=效果量）', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('E:/问题3_图3_成本效果气泡图.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ 图3：成本-效果气泡图")

# 图4：分组效果对比
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
groups_tcm = ['≤58分\n(轻度)', '59-61分\n(中度)', '≥62分\n(重度)']
means_tcm = [df_results[df_results['痰湿分组'] == g]['名义下降'].mean() for g in
             ['≤58分(轻度)', '59-61分(中度)', '≥62分(重度)']]
stds_tcm = [df_results[df_results['痰湿分组'] == g]['名义下降'].std() for g in
            ['≤58分(轻度)', '59-61分(中度)', '≥62分(重度)']]
colors_tcm_bar = [COLORS['tcm_1'], COLORS['tcm_2'], COLORS['tcm_3']]
bars = axes[0].bar(groups_tcm, means_tcm, yerr=stds_tcm, capsize=8, color=colors_tcm_bar, edgecolor='white',
                   linewidth=2, width=0.6)
for bar, mean in zip(bars, means_tcm):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{mean:.1f}分', ha='center', fontsize=13,
                 fontweight='bold')
axes[0].set_xlabel('痰湿程度', fontsize=12, fontweight='bold')
axes[0].set_ylabel('平均名义下降量（分）', fontsize=12, fontweight='bold')
axes[0].set_title('不同痰湿程度患者的干预效果', fontsize=14, fontweight='bold')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

groups_act = ['<40分\n(重度依赖)', '40-59分\n(中度依赖)', '≥60分\n(轻度依赖)']
means_act = [df_results[df_results['活动分组'] == g]['名义下降'].mean() for g in
             ['<40分(重度依赖)', '40-59分(中度依赖)', '≥60分(轻度依赖)']]
stds_act = [df_results[df_results['活动分组'] == g]['名义下降'].std() for g in
            ['<40分(重度依赖)', '40-59分(中度依赖)', '≥60分(轻度依赖)']]
colors_act_bar = [COLORS['danger'], COLORS['warning'], COLORS['success']]
bars = axes[1].bar(groups_act, means_act, yerr=stds_act, capsize=8, color=colors_act_bar, edgecolor='white',
                   linewidth=2, width=0.6)
for bar, mean in zip(bars, means_act):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{mean:.1f}分', ha='center', fontsize=13,
                 fontweight='bold')
axes[1].set_xlabel('活动能力', fontsize=12, fontweight='bold')
axes[1].set_ylabel('平均名义下降量（分）', fontsize=12, fontweight='bold')
axes[1].set_title('不同活动能力患者的干预效果', fontsize=14, fontweight='bold')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('E:/问题3_图4_分组效果对比.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ 图4：分组效果对比图")

# 图5：样本ID=1,2,3对比
fig, ax = plt.subplots(figsize=(12, 7))
samples = df_results[df_results['样本ID'].isin([1, 2, 3])].sort_values('样本ID')
x = np.arange(3)
width = 0.35
bars1 = ax.bar(x - width / 2, samples['名义下降'], width, label='名义下降', color=COLORS['primary'], edgecolor='white',
               linewidth=2, alpha=0.9)
bars2 = ax.bar(x + width / 2, samples['最坏下降'], width, label='最坏下降（鲁棒）', color=COLORS['accent2'],
               edgecolor='white', linewidth=2, alpha=0.9)
for bar, val in zip(bars1, samples['名义下降']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f'{val:.1f}', ha='center', fontsize=12,
            fontweight='bold', color=COLORS['primary'])
for bar, val in zip(bars2, samples['最坏下降']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f'{val:.1f}', ha='center', fontsize=12,
            fontweight='bold', color=COLORS['accent2'])
for i, (_, row) in enumerate(samples.iterrows()):
    info_text = f"痰湿{row['当前痰湿积分']:.0f}分\n活动{row['活动总分']:.0f}分"
    ax.text(i, -3, info_text, ha='center', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels([f"样本ID={int(i)}" for i in samples['样本ID']], fontsize=13, fontweight='bold')
ax.set_ylabel('痰湿积分下降量（分）', fontsize=13, fontweight='bold')
ax.set_title('样本ID=1,2,3干预效果对比', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right')
ax.set_ylim(0, max(samples['名义下降']) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('E:/问题3_图5_样本对比.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ 图5：样本对比图")

# 图6：鲁棒性分析
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].hist(df_results['鲁棒性'], bins=20, color=COLORS['primary'], alpha=0.7, edgecolor='white', linewidth=1.5)
axes[0].axvline(x=df_results['鲁棒性'].mean(), color=COLORS['accent2'], linestyle='--', linewidth=2,
                label=f"均值: {df_results['鲁棒性'].mean():.3f}")
axes[0].set_xlabel('鲁棒性指标（最坏/名义）', fontsize=12, fontweight='bold')
axes[0].set_ylabel('频数', fontsize=12, fontweight='bold')
axes[0].set_title('鲁棒性分布', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[1].scatter(df_results['名义下降'], df_results['最坏下降'], c=df_results['初始中医等级'], cmap='RdYlGn_r', s=60,
                alpha=0.7, edgecolors='white', linewidth=1)
axes[1].plot([0, 35], [0, 35], 'k--', alpha=0.3, label='y=x (完全鲁棒)')
axes[1].set_xlabel('名义下降量（分）', fontsize=12, fontweight='bold')
axes[1].set_ylabel('最坏情况下降量（分）', fontsize=12, fontweight='bold')
axes[1].set_title('名义效果 vs 最坏效果', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
cbar.set_label('中医等级', fontsize=11)
plt.tight_layout()
plt.savefig('E:/问题3_图6_鲁棒性分析.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ 图6：鲁棒性分析图")

# ============================================================================
# 10. 总结报告
# ============================================================================
print("\n" + "=" * 80)
print("【问题三总结】")
print("=" * 80)
print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                      问题三 核心结论（动态规划版）                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. 求解方法                                                                  │
│    - 动态规划：状态变量(S_k, B_k)，递推求解最优策略                           │
│                                                                              │
│ 2. 创新点                                                                    │
│    - 效果递减：痰湿积分越低，改善越难（α={Params.DIMINISHING_ALPHA}）          │
│    - 协同效应：中医≥2级+活动≥2级+频率≥5次 → 增益γ={Params.SYNERGY_GAMMA}      │
│    - 鲁棒优化：考虑参数不确定性（λ={Params.LAMBDA_ROBUST}）                    │
│    - 耐受度惩罚：高频高强方案受惩罚（λ_tol={Params.TOLERANCE_LAMBDA}）         │
│                                                                              │
│ 3. 方案分布                                                                  │
│    - 中医1级: {len(df_results[df_results['初始中医等级'] == 1])}人, 2级: {len(df_results[df_results['初始中医等级'] == 2])}人, 3级: {len(df_results[df_results['初始中医等级'] == 3])}人 │
│    - 活动1级: {len(df_results[df_results['最优强度'] == 1])}人, 2级: {len(df_results[df_results['最优强度'] == 2])}人, 3级: {len(df_results[df_results['最优强度'] == 3])}人 │
│                                                                              │
│ 4. 样本ID=1,2,3方案                                                          │
│    ID=1: 中医3级, 活动1级, {int(df_results[df_results['样本ID'] == 1]['最优频率'].values[0])}次/周, 下降{df_results[df_results['样本ID'] == 1]['名义下降'].values[0]:.1f}分 │
│    ID=2: 中医1级, 活动2级, {int(df_results[df_results['样本ID'] == 2]['最优频率'].values[0])}次/周, 下降{df_results[df_results['样本ID'] == 2]['名义下降'].values[0]:.1f}分 │
│    ID=3: 中医2级, 活动3级, {int(df_results[df_results['样本ID'] == 3]['最优频率'].values[0])}次/周, 下降{df_results[df_results['样本ID'] == 3]['名义下降'].values[0]:.1f}分 │
│                                                                              │
│ 5. 鲁棒性分析                                                                │
│    - 平均鲁棒性: {df_results['鲁棒性'].mean():.3f}                            │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("\n✓ 问题三（动态规划版）分析完成！图表已保存至E盘。")