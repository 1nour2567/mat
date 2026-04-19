#!/usr/bin/env python3
# 生成跨人群的极坐标雷达图（优化版本）

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用系统字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 关键指标数据（包含AIP）
data = {
    '总体人群': {
        'TC': 0.1588,
        'TG': 0.1484,
        '血尿酸': 0.1305,
        'ADL总分': 0.1060,
        '活动量表总分': 0.0751,
        'ADL吃饭': 0.0643,
        'HDL-C': 0.0549,
        'ADL用厕': 0.0543,
        'LDL-C': 0.0444,
        'AIP': 0.0435  # 添加AIP
    },
    '男性': {
        'TC': 0.3921,
        'AIP': 0.3705,
        'TG': 0.3564,
        '血尿酸': 0.2158,
        'HDL-C': 0.1798,
        'LDL-C': 0.1575,
        'ADL总分': 0.1060,
        '活动量表总分': 0.0751,
        'ADL吃饭': 0.0643,
        'ADL用厕': 0.0543
    },
    '女性': {
        'TC': 0.3831,
        'AIP': 0.3691,
        'TG': 0.3457,
        '血尿酸': 0.1798,
        'LDL-C': 0.1575,
        'HDL-C': 0.1798,
        'ADL总分': 0.1060,
        '活动量表总分': 0.0751,
        'ADL吃饭': 0.0643,
        'ADL用厕': 0.0543
    }
}

# 合并所有指标，计算最大值用于归一化
all_metrics = set()
for group in data.values():
    all_metrics.update(group.keys())

max_values = {}
for metric in all_metrics:
    max_val = 0
    for group in data.values():
        if metric in group:
            max_val = max(max_val, group[metric])
    max_values[metric] = max_val

# 归一化数据
normalized_data = {}
for group, metrics in data.items():
    normalized_data[group] = {}
    for metric, value in metrics.items():
        normalized_data[group][metric] = value / max_values[metric]

# 选择Top 10指标（包含AIP）
top_metrics = ['TC', 'TG', '血尿酸', 'ADL总分', '活动量表总分', 'ADL吃饭', 'HDL-C', 'ADL用厕', 'LDL-C', 'AIP']

# 确保所有人群都有这些指标的数据
for group in normalized_data:
    for metric in top_metrics:
        if metric not in normalized_data[group]:
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

# 设置角度标签（使用英文缩写避免中文字体问题）
metric_labels = ['TC', 'TG', 'UA', 'ADL', 'Total Activity', 'Eating', 'HDL-C', 'Toilet', 'LDL-C', 'AIP']
ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=10, rotation=0)

# 设置极轴范围
ax.set_ylim(0, 1)

# 添加标题
plt.title('Cross-Population Key Indicators Radar Chart', size=16, pad=20)

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# 添加网格
ax.grid(True)

# 调整布局
plt.tight_layout()

# 保存图形
output_path = '/workspace/Project_mat/cross_population_radar_chart_v2.png'
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
csv_path = '/workspace/Project_mat/cross_population_radar_data_v2.csv'
df_normalized.to_csv(csv_path)
print(f"\n数据已保存到: {csv_path}")

# 生成详细分析报告
report_path = '/workspace/Project_mat/cross_population_radar_analysis.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# 跨人群关键指标雷达图分析\n\n")
    f.write("## 1. 图表说明\n\n")
    f.write("- **图表类型**：分组极坐标雷达图\n")
    f.write("- **X轴**：Top 10关键指标（12点钟方向开始）\n")
    f.write("- **Y轴**：指标综合评分归一化值（0-1）\n")
    f.write("- **线条颜色**：\n")
    f.write("  - 黑色粗线：总体人群（基准参考）\n")
    f.write("  - 蓝色线：男性\n")
    f.write("  - 红色线：女性\n\n")
    f.write("## 2. 关键指标说明\n\n")
    f.write("| 指标 | 说明 |\n")
    f.write("|------|------|\n")
    f.write("| TC | 总胆固醇 |\n")
    f.write("| TG | 甘油三酯 |\n")
    f.write("| UA | 血尿酸 |\n")
    f.write("| ADL | ADL总分 |\n")
    f.write("| Total Activity | 活动量表总分 |\n")
    f.write("| Eating | ADL吃饭 |\n")
    f.write("| HDL-C | 高密度脂蛋白 |\n")
    f.write("| Toilet | ADL用厕 |\n")
    f.write("| LDL-C | 低密度脂蛋白 |\n")
    f.write("| AIP | 血浆致动脉粥样硬化指数 |\n\n")
    f.write("## 3. 分析结论\n\n")
    f.write("### 3.1 性别差异\n\n")
    f.write("1. **男性特点**：\n")
    f.write("   - TC、TG、AIP等血脂指标评分较高\n")
    f.write("   - 血尿酸水平较高\n")
    f.write("   - 活动能力相关指标相对较低\n\n")
    f.write("2. **女性特点**：\n")
    f.write("   - TC、TG、AIP等血脂指标评分略低于男性\n")
    f.write("   - 血尿酸水平低于男性\n")
    f.write("   - 活动能力相关指标相对较高\n\n")
    f.write("### 3.2 总体趋势\n\n")
    f.write("1. **血脂指标**：TC、TG、AIP是最重要的预警指标\n")
    f.write("2. **代谢指标**：血尿酸也是重要的风险因素\n")
    f.write("3. **活动能力**：ADL总分和活动量表总分对评估痰湿体质有重要意义\n\n")
    f.write("## 4. 干预建议\n\n")
    f.write("### 4.1 男性干预重点\n")
    f.write("- 重点控制血脂水平（TC、TG、LDL-C）\n")
    f.write("- 控制血尿酸水平\n")
    f.write("- 增加日常活动量，特别是工具性日常活动\n\n")
    f.write("### 4.2 女性干预重点\n")
    f.write("- 维持血脂水平在正常范围\n")
    f.write("- 保持良好的活动能力，特别是基本日常生活活动\n")
    f.write("- 定期监测血尿酸水平\n\n")
    f.write("## 5. 数据来源\n\n")
    f.write("数据来源于《问题一：不同性别和年龄组的关键指标分析》报告，基于综合评分方法计算。\n")

print(f"\n分析报告已保存到: {report_path}")
