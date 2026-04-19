#!/usr/bin/env python3
# 生成活动能力预警效能的年龄-指标热力日历图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 数据准备
age_groups = ['40-49岁', '50-59岁', '60-69岁', '70-79岁', '80-89岁']

# 核心活动指标
activities = ['ADL总分', 'IADL总分', 'ADL吃饭', 'ADL用厕', 'ADL步行', 'IADL理财', 'IADL服药', 'IADL交通']

# 活动能力预警效能数据
# 格式：[40-49, 50-59, 60-69, 70-79, 80-89]
data = np.array([
    [0.1412, 0.0, 0.0, 0.1607, 0.0, 0.0, 0.0, 0.0],  # ADL总分
    [0.0, 0.0, 0.3593, 0.1510, 0.1827, 0.0, 0.0, 0.0],  # IADL总分
    [0.0, 0.0, 0.3223, 0.0, 0.1354, 0.0, 0.0, 0.0],  # ADL吃饭
    [0.1607, 0.1966, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ADL用厕
    [0.0, 0.1362, 0.0, 0.1325, 0.0, 0.0, 0.0, 0.0],  # ADL步行
    [0.0, 0.1512, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # IADL理财
    [0.0, 0.0, 0.3090, 0.0, 0.0, 0.0, 0.0, 0.0],  # IADL服药
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # IADL交通
])

# 活动量表总分数据（单独处理，因为它是一个重要指标）
activity_scale_score = [0.1642, 0.0, 0.0, 0.1685, 0.2136]

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))

# 创建热力图
im = ax.imshow(data, cmap='Reds', vmin=0, vmax=0.4)

# 设置坐标轴标签
ax.set_xticks(np.arange(len(age_groups)))
ax.set_xticklabels(age_groups, fontsize=11)
ax.set_yticks(np.arange(len(activities)))
ax.set_yticklabels(activities, fontsize=11)
ax.set_xlabel('年龄组', fontsize=12)
ax.set_ylabel('核心活动指标', fontsize=12)
ax.set_title('活动能力预警效能的年龄-指标热力日历图', fontsize=14, fontweight='bold')

# 在每个格子内写入数值
for i in range(len(activities)):
    for j in range(len(age_groups)):
        if data[i, j] > 0:
            text = ax.text(j, i, f'{data[i, j]:.4f}',
                          ha="center", va="center", color="black", fontsize=9)

# 添加活动量表总分数据（在顶部添加一行）
for j in range(len(age_groups)):
    if activity_scale_score[j] > 0:
        text = ax.text(j, -0.5, f'{activity_scale_score[j]:.4f}',
                      ha="center", va="center", color="black", fontsize=9, fontweight='bold')

# 在顶部添加活动量表总分标签
ax.text(-1, -0.5, '活动量表总分', ha="right", va="center", fontsize=11, fontweight='bold')

# 添加颜色条
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('综合评分', fontsize=10)

# 标注功能需求漂移路径
# 40-59岁：ADL基础自理主导
ax.annotate('ADL基础自理主导', xy=(0.5, 6.5), xytext=(0.5, 7.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            ha='center', fontsize=10, fontweight='bold')

# 60-69岁：IADL工具性活动爆发式重要
ax.annotate('IADL工具性活动爆发式重要', xy=(2, 3), xytext=(2, 7.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            ha='center', fontsize=10, fontweight='bold')

# 70岁+：综合总分回归主导
ax.annotate('综合总分回归主导', xy=(3.5, 0.5), xytext=(3.5, 7.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            ha='center', fontsize=10, fontweight='bold')

# 调整布局
plt.tight_layout()

# 保存图表
output_path = '/workspace/Project_mat/活动能力预警效能热力日历图.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'图表已保存到：{output_path}')

plt.show()