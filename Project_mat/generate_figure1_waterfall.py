#!/usr/bin/env python3
# 生成Figure_1 双目标综合评分堆叠瀑布图

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==================== 数据准备 ====================
# 构建综合评分DataFrame
# 假设的特征列表（包含要展示的所有指标）
features_to_plot = [
    '痰湿质得分', '痰湿质得分×BMI', '痰湿质得分×TG', '痰湿质得分/HDL-C',
    '痰湿质得分×LDL-C', '血脂异常项数', '痰湿质得分×AIP', 'TG/HDL比值',
    'TG', 'AIP', 'TC', '血尿酸', 'ADL总分', 'HDL-C', '活动量表总分',
    'non-HDL-C缩尾', 'AIP缩尾', 'TC/HDL缩尾', 'LDL/HDL缩尾', 'TG/HDL缩尾'
]

# 为每个特征定义类型（用于配色标注）
type_map = {
    '痰湿质得分': '中医体质',
    '痰湿质得分×BMI': '中西医交叉',
    '痰湿质得分×TG': '中西医交叉',
    '痰湿质得分/HDL-C': '中西医交叉',
    '痰湿质得分×LDL-C': '中西医交叉',
    '血脂异常项数': '西医派生',
    '痰湿质得分×AIP': '中西医交叉',
    'TG/HDL比值': '西医派生',
    'TG': '西医基础',
    'AIP': '西医派生',
    'TC': '西医基础',
    '血尿酸': '西医基础',
    'ADL总分': '活动量表',
    'HDL-C': '西医基础',
    '活动量表总分': '活动量表',
    'non-HDL-C缩尾': '西医派生',
    'AIP缩尾': '西医派生',
    'TC/HDL缩尾': '西医派生',
    'LDL/HDL缩尾': '西医派生',
    'TG/HDL缩尾': '西医派生'
}

# 假设的各维度得分（根据实际数据调整）
spearman_scores = {
    '痰湿质得分': 0.85,
    '痰湿质得分×BMI': 0.72,
    '痰湿质得分×TG': 0.68,
    '痰湿质得分/HDL-C': 0.65,
    '痰湿质得分×LDL-C': 0.62,
    '血脂异常项数': 0.55,
    '痰湿质得分×AIP': 0.58,
    'TG/HDL比值': 0.52,
    'TG': 0.48,
    'AIP': 0.45,
    'TC': 0.42,
    '血尿酸': 0.38,
    'ADL总分': 0.35,
    'HDL-C': 0.32,
    '活动量表总分': 0.30,
    'non-HDL-C缩尾': 0.40,
    'AIP缩尾': 0.43,
    'TC/HDL缩尾': 0.38,
    'LDL/HDL缩尾': 0.35,
    'TG/HDL缩尾': 0.42
}

mi_scores = {
    '痰湿质得分': 0.35,
    '痰湿质得分×BMI': 0.65,
    '痰湿质得分×TG': 0.70,
    '痰湿质得分/HDL-C': 0.68,
    '痰湿质得分×LDL-C': 0.60,
    '血脂异常项数': 0.80,
    '痰湿质得分×AIP': 0.72,
    'TG/HDL比值': 0.75,
    'TG': 0.82,
    'AIP': 0.85,
    'TC': 0.78,
    '血尿酸': 0.65,
    'ADL总分': 0.45,
    'HDL-C': 0.55,
    '活动量表总分': 0.40,
    'non-HDL-C缩尾': 0.75,
    'AIP缩尾': 0.83,
    'TC/HDL缩尾': 0.70,
    'LDL/HDL缩尾': 0.65,
    'TG/HDL缩尾': 0.78
}

pls_scores = {
    '痰湿质得分': 0.45,
    '痰湿质得分×BMI': 0.60,
    '痰湿质得分×TG': 0.65,
    '痰湿质得分/HDL-C': 0.62,
    '痰湿质得分×LDL-C': 0.58,
    '血脂异常项数': 0.55,
    '痰湿质得分×AIP': 0.68,
    'TG/HDL比值': 0.50,
    'TG': 0.52,
    'AIP': 0.55,
    'TC': 0.58,
    '血尿酸': 0.48,
    'ADL总分': 0.40,
    'HDL-C': 0.45,
    '活动量表总分': 0.38,
    'non-HDL-C缩尾': 0.55,
    'AIP缩尾': 0.58,
    'TC/HDL缩尾': 0.52,
    'LDL/HDL缩尾': 0.48,
    'TG/HDL缩尾': 0.54
}

# 权重（根据熵权法计算）
w_spearman = 0.1507
w_mi = 0.7091
w_pls = 0.1401

# 初始化存储
data = []
for feat in features_to_plot:
    # 获取各维度原始得分
    sp = spearman_scores.get(feat, 0)
    mi = mi_scores.get(feat, 0)
    pls = pls_scores.get(feat, 0)
    
    # 极差标准化（简化处理，假设值已在0-1之间）
    norm_sp = sp
    norm_mi = mi
    norm_pls = pls
    
    # 加权贡献
    contrib_sp = w_spearman * norm_sp
    contrib_mi = w_mi * norm_mi
    contrib_pls = w_pls * norm_pls
    total = contrib_sp + contrib_mi + contrib_pls
    
    data.append({
        '特征': feat,
        '类型': type_map.get(feat, '其他'),
        'Spearman贡献': contrib_sp,
        '互信息贡献': contrib_mi,
        'PLS贡献': contrib_pls,
        '综合评分': total
    })

df_waterfall = pd.DataFrame(data).sort_values('综合评分', ascending=False).reset_index(drop=True)

# ==================== 绘制堆叠瀑布图 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 7))

# 颜色方案（学术风格）
colors = {
    'Spearman贡献': '#2E86AB',   # 深蓝
    '互信息贡献': '#A23B72',     # 紫红
    'PLS贡献': '#F18F01'         # 橙黄
}

# 准备堆叠数据
x = np.arange(len(df_waterfall))
bottom = np.zeros(len(df_waterfall))

# 逐层堆叠
layer1 = df_waterfall['Spearman贡献'].values
layer2 = df_waterfall['互信息贡献'].values
layer3 = df_waterfall['PLS贡献'].values

bars1 = ax.bar(x, layer1, color=colors['Spearman贡献'], label='Spearman贡献（痰湿表征）', width=0.7)
bars2 = ax.bar(x, layer2, bottom=layer1, color=colors['互信息贡献'], label='互信息贡献（风险预警）', width=0.7)
bars3 = ax.bar(x, layer3, bottom=layer1+layer2, color=colors['PLS贡献'], label='PLS贡献（联合结构）', width=0.7)

# 添加综合评分数值标注在柱子顶部
for i, (total, feat) in enumerate(zip(df_waterfall['综合评分'], df_waterfall['特征'])):
    ax.text(i, total + 0.01, f'{total:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ========== 关键区域标注 ==========
# 1. 标注"单一目标高分区"（痰湿质得分）
idx_tanshi = df_waterfall[df_waterfall['特征']=='痰湿质得分'].index[0]
ax.annotate('单一目标高分\n(仅痰湿表征)', 
            xy=(idx_tanshi, df_waterfall.loc[idx_tanshi, '综合评分']),
            xytext=(idx_tanshi-0.8, df_waterfall.loc[idx_tanshi, '综合评分']+0.15),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1),
            fontsize=9, ha='center', color='#333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9E6', edgecolor='#CCCCCC'))

# 2. 标注"中西医协同效应区"（交叉特征区域）
cross_indices = df_waterfall[df_waterfall['类型']=='中西医交叉'].index
if len(cross_indices) > 0:
    left = min(cross_indices) - 0.4
    right = max(cross_indices) + 0.4
    y_max = df_waterfall['综合评分'].max() * 0.9
    ax.annotate('', xy=(left, y_max), xytext=(right, y_max),
                arrowprops=dict(arrowstyle='<->', color='#D62828', lw=2))
    ax.text((left+right)/2, y_max+0.02, '中西医协同效应区', 
            ha='center', fontsize=11, fontweight='bold', color='#D62828',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#D62828'))

# ========== 坐标轴与标题设置 ==========
ax.set_xticks(x)
ax.set_xticklabels(df_waterfall['特征'], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('综合评分（加权贡献）', fontsize=12, fontweight='bold')
ax.set_xlabel('指标名称（按综合评分降序排列）', fontsize=12, fontweight='bold')
ax.set_title('图1 基于熵权-多准则决策的双目标综合评分Top15指标瀑布图\n(柱内堆叠表示三个维度的加权贡献)', 
             fontsize=14, fontweight='bold', pad=20)

# 图例
ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')

# 添加背景网格（提升可读性）
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# 添加类型颜色条（在x轴标签下方用小色块标注）
for i, (feat, ftype) in enumerate(zip(df_waterfall['特征'], df_waterfall['类型'])):
    color_map_type = {'中医体质':'#8ECAE6', '中西医交叉':'#219EBC', 
                      '西医派生':'#FFB703', '西医基础':'#FB8500', '活动量表':'#8D99AE'}
    ax.add_patch(plt.Rectangle((i-0.35, -0.08), 0.7, 0.04, 
                               facecolor=color_map_type.get(ftype, '#ccc'), 
                               transform=ax.get_xaxis_transform(), clip_on=False))

plt.tight_layout()
plt.savefig('Figure1_瀑布图_修正版.png', dpi=300, bbox_inches='tight')
plt.show()

print("图表已生成并保存为: Figure1_瀑布图_修正版.png")
print("\n综合评分Top15指标:")
print(df_waterfall[['特征', '类型', '综合评分']])