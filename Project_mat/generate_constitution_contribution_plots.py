#!/usr/bin/env python3
# 生成九种体质贡献度的性别背靠背条形图和年龄组热图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 数据准备
constitutions = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']

# 性别贡献度数据
gender_data = {
    '男': [0.0282, 0.1305, 0.1628, 0.0581, 0.0725, 0.0517, 0.0553, 0.0063, 0.0367],
    '女': [0.1578, 0.1370, 0.1095, 0.0330, 0.0754, 0.0672, 0.0409, 0.0339, 0.0080]
}

# 年龄组贡献度数据
age_groups = ['40-49岁', '50-59岁', '60-69岁', '70-79岁', '80-89岁']
age_data = np.array([
    [0.3298, 0.6112, 0.0441, 0.2648, 0.0343, 0.0207, 0.0678, 0.0219, 0.0264],  # 40-49岁
    [0.0655, 0.0116, 0.2158, 0.0676, 0.0455, 0.1067, 0.1546, 0.1897, 0.0679],  # 50-59岁
    [0.0495, 0.1842, 0.2094, 0.0505, 0.0811, 0.0330, 0.0310, 0.0277, 0.3347],  # 60-69岁
    [0.2928, 0.0491, 0.0447, 0.0094, 0.0042, 0.0255, 0.2794, 0.0647, 0.1152],  # 70-79岁
    [0.0564, 0.4127, 0.1177, 0.0304, 0.0233, 0.0756, 0.2548, 0.1721, 0.0827],  # 80-89岁
])

# 找出每个年龄组贡献度最高的体质
top_constitutions_per_age = ['气虚质', '阳虚质', '特禀质', '平和质', '气虚质']

# 创建组合图表
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 1, hspace=0.3)

# 图2-1：性别背靠背条形图（上半部分）
ax1 = fig.add_subplot(gs[0, 0])
y_pos = np.arange(len(constitutions))
width = 0.4

# 绘制男性（左侧，蓝色）
male_bars = ax1.barh(y_pos - width/2, [-x for x in gender_data['男']], height=width, color='#1f77b4', label='男性')
# 绘制女性（右侧，红色）
female_bars = ax1.barh(y_pos + width/2, gender_data['女'], height=width, color='#d62728', label='女性')

# 设置属性
ax1.set_yticks(y_pos)
ax1.set_yticklabels(constitutions, fontsize=12)
ax1.set_xlabel('体质贡献度', fontsize=12)
ax1.set_title('图2-1：九种体质贡献度性别背靠背条形图', fontsize=14, fontweight='bold')
ax1.set_xlim(-0.2, 0.2)
ax1.axvline(0, color='black', linewidth=0.5)
ax1.legend()

# 标注数值
for i, (male, female) in enumerate(zip(gender_data['男'], gender_data['女'])):
    ax1.text(-male - 0.008, i - width/2, f'{male:.4f}', va='center', ha='right', fontsize=9)
    ax1.text(female + 0.008, i + width/2, f'{female:.4f}', va='center', ha='left', fontsize=9)

# 高亮痰湿质行
tan_shi_idx = constitutions.index('痰湿质')
ax1.add_patch(Rectangle((-0.2, tan_shi_idx - width), 0.4, width*2, 
                      fill=False, edgecolor='#ff7f0e', linewidth=2))

# 在平和质行右侧加注星号
ping_he_idx = constitutions.index('平和质')
ax1.text(0.17, ping_he_idx + width/2, '*', fontsize=16, color='black', ha='center')

# 添加图注
ax1.text(0.5, -0.12, '*女性平和质高贡献可能源于围绝经期激素波动掩盖体质偏颇，提示' +
        '"虚假平和"现象。', transform=ax1.transAxes, ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# 图2-2：年龄组贡献度热图（下半部分）
ax2 = fig.add_subplot(gs[1, 0])

# 计算总贡献度并排序
total_contributions = np.sum(age_data, axis=0)
sorted_indices = np.argsort(total_contributions)[::-1]
sorted_constitutions = [constitutions[i] for i in sorted_indices]
sorted_age_data = age_data[:, sorted_indices]

# 创建热图
im = ax2.imshow(sorted_age_data.T, cmap='Reds', vmin=0, vmax=0.7)

# 设置坐标轴标签
ax2.set_xticks(np.arange(len(age_groups)))
ax2.set_xticklabels(age_groups, fontsize=11)
ax2.set_yticks(np.arange(len(sorted_constitutions)))
ax2.set_yticklabels(sorted_constitutions, fontsize=11)
ax2.set_xlabel('年龄组', fontsize=12)
ax2.set_ylabel('九种体质（按总贡献度降序排列）', fontsize=12)
ax2.set_title('图2-2：年龄组体质贡献度热图', fontsize=14, fontweight='bold')

# 在每个格子内写入数值
for i in range(len(sorted_constitutions)):
    for j in range(len(age_groups)):
        text = ax2.text(j, i, f'{sorted_age_data.T[i, j]:.4f}',
                       ha="center", va="center", color="black", fontsize=9)

# 用黑色粗框标出每个年龄组的Top1体质
for j, age_group in enumerate(age_groups):
    top_constitution = top_constitutions_per_age[j]
    if top_constitution in sorted_constitutions:
        i = sorted_constitutions.index(top_constitution)
        ax2.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, 
                              fill=False, edgecolor='black', linewidth=3))

# 添加颜色条
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('体质贡献度', fontsize=10)

# 旋转并对齐x轴标签
plt.setp(ax2.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

# 保存图表
output_path = '/workspace/Project_mat/体质贡献度综合图表.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'图表已保存到：{output_path}')

plt.show()
