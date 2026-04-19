#!/usr/bin/env python3
# Figure 3: 年龄组体质贡献度热图
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 来自问题一分析报告的真实数据
constitutions = ['气虚质', '平和质', '阳虚质', '特禀质',
                 '血瘀质', '气郁质', '湿热质', '阴虚质', '痰湿质']
age_groups = ['40-49', '50-59', '60-69', '70-79', '80-89']

data_matrix = np.array([
    [0.6112, 0.0116, 0.1842, 0.0491, 0.4127],  # 气虚质
    [0.3298, 0.0655, 0.0495, 0.2928, 0.0564],  # 平和质
    [0.0441, 0.2158, 0.2094, 0.0447, 0.1177],  # 阳虚质
    [0.0264, 0.0679, 0.3347, 0.1152, 0.0827],  # 特禀质
    [0.0678, 0.1546, 0.0310, 0.2794, 0.2548],  # 血瘀质
    [0.0219, 0.1897, 0.0277, 0.0647, 0.1721],  # 气郁质
    [0.0207, 0.1067, 0.0330, 0.0255, 0.0756],  # 湿热质
    [0.2648, 0.0676, 0.0505, 0.0094, 0.0304],  # 阴虚质
    [0.0343, 0.0455, 0.0811, 0.0042, 0.0233]   # 痰湿质
])

# 创建图表
fig, ax = plt.subplots(figsize=(12, 9))

# 绘制热图
im = sns.heatmap(data_matrix, cmap='Reds', ax=ax,
                 xticklabels=age_groups, yticklabels=constitutions,
                 annot=True, fmt='.4f', linewidths=0.5,
                 vmin=0, vmax=0.65)

# 标记每个年龄组的Top 1
top1_idx = [0, 2, 3, 1, 0]  # 每个年龄组贡献最大的体质索引
for i, j in enumerate(top1_idx):
    ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False,
                               edgecolor='black', linewidth=3, linestyle='-'))

# 添加标题和标签
ax.set_title('Figure 3: 九种体质对发病风险的贡献度 - 年龄组差异',
             fontsize=14, fontweight='bold')
ax.set_xlabel('年龄组', fontsize=12)
ax.set_ylabel('体质类型', fontsize=12)

# 在图上方加生命周期规律注释
ax.text(0.5, 1.12,
        '青年期(气虚) → 中年期(阳虚) → 老年前期(特禀) → 老年期(平和) → 高龄期(气虚)',
        ha='center', va='bottom', color='#d62728', fontsize=12,
        transform=ax.get_xaxis_transform(), fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('/workspace/Project_mat/Figure_3_年龄组热图.png',
            dpi=300, bbox_inches='tight')
print('Figure 3 saved successfully!')
plt.show()

