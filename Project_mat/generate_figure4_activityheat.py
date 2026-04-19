#!/usr/bin/env python3
# Figure 4: 活动能力年龄热图
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 来自问题一分析报告的真实数据
activity_indicators = ['ADL总分', 'IADL总分', 'ADL吃饭', 'ADL用厕',
                       'ADL步行', 'IADL理财', 'IADL服药', 'IADL交通']
age_groups = ['40-49', '50-59', '60-69', '70-79', '80-89']

data_matrix = np.array([
    [0.1412, 0.1171, 0.15,    0.1047, 0.1229],  # ADL总分
    [0.1059, 0.1055, 0.3593, 0.1510, 0.1827],  # IADL总分
    [0.0585, 0.1268, 0.3223, 0.1215, 0.1354],  # ADL吃饭
    [0.1607, 0.1966, 0.1961, 0.0998, 0.14],     # ADL用厕
    [0.0689, 0.1362, 0.17,    0.1325, 0.12],     # ADL步行
    [0.0797, 0.1512, 0.1765, 0.0760, 0.1144],  # IADL理财
    [0.0904, 0.0599, 0.3090, 0.0724, 0.1183],  # IADL服药
    [0.0461, 0.0748, 0.1748, 0.0703, 0.1191]   # IADL交通
])

# 创建图表
fig, ax = plt.subplots(figsize=(12, 9))

# 绘制热图
im = ax.imshow(data_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=0.4)

# 设置标签
ax.set_xticks(np.arange(len(age_groups)))
ax.set_xticklabels(age_groups, fontsize=11)
ax.set_yticks(np.arange(len(activity_indicators)))
ax.set_yticklabels(activity_indicators, fontsize=11)

# 在格子内写具体数值
for i in range(len(activity_indicators)):
    for j in range(len(age_groups)):
        ax.text(j, i, f'{data_matrix[i, j]:.4f}',
                ha='center', va='center', color='black', fontsize=9)

# 标注功能需求漂移路径 - 用不同颜色框出重要区域
# 40-59岁区域（ADL用厕、ADL步行）
rect1 = plt.Rectangle((-0.5, 3), 2, 2, linewidth=3,
                      edgecolor='#1f77b4', facecolor='none', label='40-59岁重点')
ax.add_patch(rect1)

# 60-69岁区域（IADL总分、ADL吃饭、IADL服药）
rect2 = plt.Rectangle((1.5, 1), 1, 2, linewidth=3,
                      edgecolor='#d62728', facecolor='none', label='60-69岁关键')
ax.add_patch(rect2)

# 70岁+区域重点标注
ax.axvspan(2.5, 4.5, color='purple', alpha=0.1)

# 添加标题和标签
ax.set_title('Figure 4: 活动能力预警效能的年龄-指标热力图',
             fontsize=14, fontweight='bold')
ax.set_xlabel('年龄组', fontsize=12)
ax.set_ylabel('活动指标', fontsize=12)

# 添加图例
handles = [
    plt.Line2D([], [], color='#1f77b4', lw=3, label='40-59岁ADL主导'),
    plt.Line2D([], [], color='#d62728', lw=3, label='60-69岁IADL爆发')
]
ax.legend(handles=handles, loc='upper right')

# 添加功能需求漂移注释
ax.text(0.5, -0.12,
        '功能需求漂移路径: ADL基础自理 → IADL工具性活动爆发 → 综合总分回归主导',
        ha='center', va='top', color='#d62728', fontsize=10,
        transform=ax.get_xaxis_transform(), fontweight='bold')

# 添加颜色条
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('综合评分', rotation=-90, va='bottom')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/workspace/Project_mat/Figure_4_活动能力热图.png',
            dpi=300, bbox_inches='tight')
print('Figure 4 saved successfully!')
plt.show()

