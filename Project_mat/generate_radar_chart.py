#!/usr/bin/env python3
# 生成跨人群的极坐标雷达图

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 关键指标数据
data = {
    '总体人群': {
        'TC（总胆固醇）': 0.1588,
        'TG（甘油三酯）': 0.1484,
        '血尿酸': 0.1305,
        'ADL总分': 0.1060,
        '活动量表总分（ADL总分+IADL总分）': 0.0751,
        'ADL吃饭': 0.0643,
        'HDL-C（高密度脂蛋白）': 0.0549,
        'ADL用厕': 0.0543,
        'LDL-C（低密度脂蛋白）': 0.0444,
        'ADL洗澡': 0.0435
    },
    '男性': {
        'TC（总胆固醇）': 0.3921,
        'AIP': 0.3705,
        'TG（甘油三酯）': 0.3564,
        '血尿酸': 0.2158,
        'HDL-C（高密度脂蛋白）': 0.1798,
        'LDL-C（低密度脂蛋白）': 0.1575,  # 补充女性的第5名
        'ADL总分': 0.1060,  # 补充总体的第4名
        '活动量表总分（ADL总分+IADL总分）': 0.0751,  # 补充总体的第5名
        'ADL吃饭': 0.0643,  # 补充总体的第6名
        'ADL用厕': 0.0543  # 补充总体的第8名
    },
    '女性': {
        'TC（总胆固醇）': 0.3831,
        'AIP': 0.3691,
        'TG（甘油三酯）': 0.3457,
        '血尿酸': 0.1798,
        'LDL-C（低密度脂蛋白）': 0.1575,
        'HDL-C（高密度脂蛋白）': 0.1798,  # 补充男性的第5名
        'ADL总分': 0.1060,  # 补充总体的第4名
        '活动量表总分（ADL总分+IADL总分）': 0.0751,  # 补充总体的第5名
        'ADL吃饭': 0.0643,  # 补充总体的第6名
        'ADL用厕': 0.0543  # 补充总体的第8名
    }
}

# 合并所有指标，取Top 10
all_metrics = set()
for group in data.values():
    all_metrics.update(group.keys())

# 计算每个指标在各人群中的最大值，用于归一化
max_values = {}
for metric in all_metrics:
    max_val = 0
    for group in data.values():
        if metric in group:
            max_val = max(max_val, group[metric])
    max_values[metric] = max_val

# 对每个指标进行归一化（0-1）
normalized_data = {}
for group, metrics in data.items():
    normalized_data[group] = {}
    for metric, value in metrics.items():
        normalized_data[group][metric] = value / max_values[metric]

# 选择Top 10指标（基于总体人群的评分）
top_metrics = sorted(data['总体人群'].items(), key=lambda x: x[1], reverse=True)[:10]
top_metrics = [metric for metric, _ in top_metrics]

# 确保所有人群都有这些指标的数据
for group in normalized_data:
    for metric in top_metrics:
        if metric not in normalized_data[group]:
            # 如果该人群没有该指标，设置为0
            normalized_data[group][metric] = 0

# 准备雷达图数据
angles = np.linspace(0, 2 * np.pi, len(top_metrics), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

# 为每个人群准备数据
groups = ['总体人群', '男性', '女性']
colors = ['black', 'blue', 'red']
line_styles = ['-', '--', '-.']
line_widths = [2, 1.5, 1.5]

# 创建雷达图
plt.figure(figsize=(12, 10))
ax = plt.subplot(111, polar=True)

for i, (group, color, linestyle, linewidth) in enumerate(zip(groups, colors, line_styles, line_widths)):
    values = [normalized_data[group][metric] for metric in top_metrics]
    values += values[:1]  # 闭合图形
    ax.plot(angles, values, color=color, linestyle=linestyle, linewidth=linewidth, label=group)
    ax.fill(angles, values, color=color, alpha=0.1)

# 设置角度标签
ax.set_thetagrids(np.degrees(angles[:-1]), top_metrics, fontsize=10, rotation=0)

# 设置极轴范围
ax.set_ylim(0, 1)

# 添加标题
plt.title('不同人群关键指标雷达图', size=16, pad=20)

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# 添加网格
ax.grid(True)

# 调整布局
plt.tight_layout()

# 保存图形
output_path = '/workspace/Project_mat/cross_population_radar_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"雷达图已保存到: {output_path}")

# 显示图形
plt.show()

# 生成归一化数据表格
df_normalized = pd.DataFrame(normalized_data).T
df_normalized = df_normalized[top_metrics]
print("\n归一化数据:")
print(df_normalized)

# 保存数据到CSV
csv_path = '/workspace/Project_mat/cross_population_radar_data.csv'
df_normalized.to_csv(csv_path)
print(f"\n数据已保存到: {csv_path}")
